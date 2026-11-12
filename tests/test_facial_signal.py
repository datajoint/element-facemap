"""Regression tests for FacialSignal.make(), pinning three bugs fixed together:

1. The component axis was taken from the wrong dimension of motMask/motSVD
   (a `zip` over axis 0 instead of indexing the last axis), so a real
   500-component result inserted 128 rows of (160, 500) masks instead of 500
   rows of (128, 160) -- silently, no exception.
2. `"motSv" in dataset` is True for an empty list, so a short recording (one
   that fits in a single facemap chunk, with `motSv == []`) raised
   IndexError on `dataset["motSv"][roi_idx]`.
3. MovieSVD was never populated: the loop that builds it appended to
   `motion_svd_entries` (MotionSVD's list) using the field name `motmask`
   (MotionSVD's field), so `MovieSVD.insert(movie_svd_entries)` always
   inserted an empty list and the rows silently landed in the wrong table's
   entries instead of raising a name mismatch.

No real Facemap run is needed: `_proc.npy` is just a pickled dict with a
documented shape, so these are synthetic fixtures matching that shape,
combined with a real database round trip through FacialSignal.make() itself.
"""
import datetime
import numpy as np
import pytest


def _make_dataset(n_frames=6, n_components=4, ly=5, lx=4, with_movie=False, empty_motsv=False):
    """Build a dict shaped like a real facemap `_proc.npy`, fullSVD-only (one
    ROI, no separately-defined ROIs) -- the default facemap configuration.
    """
    rng = np.random.default_rng(0)
    motSVD = rng.random((n_frames, n_components)).astype("float32")
    motMask = rng.random((ly, lx, n_components)).astype("float32")
    motSv = rng.random(n_components).astype("float32")

    dataset = dict(
        fullSVD=True,
        rois=None,
        Lx=[lx],
        Ly=[ly],
        motion=[np.zeros(n_frames, dtype="float32")],
        motSVD=[motSVD],
        motMask_reshape=[motMask],
        motSv=[] if empty_motsv else [motSv],
        sbin=4,
        avgframe_reshape=rng.random((ly, lx)).astype("float32"),
        avgmotion_reshape=rng.random((ly, lx)).astype("float32"),
    )

    movSVD = movMask = movSv = None
    if with_movie:
        movSVD = rng.random((n_frames, n_components)).astype("float32")
        movMask = rng.random((ly, lx, n_components)).astype("float32")
        movSv = rng.random(n_components).astype("float32")
        dataset.update(movSVD=[movSVD], movMask_reshape=[movMask], movSv=[movSv])

    return dataset, motSVD, motMask, motSv, movSVD, movMask, movSv


def _insert_facemap_task(pipeline, session_key, tmp_path, dataset, recording_id=0):
    """FacemapTask reads its output_dir with find_full_path(), which accepts an
    already-absolute existing path directly -- no root-directory resolution
    needed for a self-contained test.
    """
    fbe = pipeline.fbe
    pipeline.lab.Device.insert1(
        dict(device="test-cam", modality="video"), skip_duplicates=True
    )
    fbe.VideoRecording.insert1(
        dict(session_key, recording_id=recording_id, device="test-cam"),
        skip_duplicates=True,
    )
    fbe.VideoRecording.File.insert1(
        dict(session_key, recording_id=recording_id, file_id=0, file_path="video.avi"),
        skip_duplicates=True,
    )
    np.save(tmp_path / "test_proc.npy", dataset, allow_pickle=True)
    fbe.FacemapTask.insert1(
        dict(
            session_key,
            recording_id=recording_id,
            facemap_task_id=0,
            facemap_output_dir=str(tmp_path),
            task_mode="load",
            facemap_params={},
            do_mot_svd=1,
            do_mov_svd=int(bool(dataset.get("movSVD"))),
        ),
        skip_duplicates=True,
    )
    fbe.FacemapProcessing.insert1(
        dict(session_key, recording_id=recording_id, facemap_task_id=0,
             processing_time=datetime.datetime.now()),
        skip_duplicates=True,
    )
    return dict(session_key, recording_id=recording_id, facemap_task_id=0)


def test_motion_svd_axis_is_correct(pipeline, session_key, tmp_path):
    """Bug 1: singular_value/motmask/projection must come from the component
    axis, not axis 0."""
    dataset, motSVD, motMask, motSv, *_ = _make_dataset()
    key = _insert_facemap_task(pipeline, session_key, tmp_path, dataset)

    pipeline.fbe.FacialSignal().make(key)

    rows = (pipeline.fbe.FacialSignal.MotionSVD() & key).fetch(
        "component_id", "singular_value", "motmask", "projection", order_by="component_id"
    )
    component_ids, singular_values, motmasks, projections = rows
    n_components = motSVD.shape[-1]
    assert list(component_ids) == list(range(n_components))
    for idx in range(n_components):
        assert singular_values[idx] == pytest.approx(motSv[idx])
        np.testing.assert_array_equal(motmasks[idx], motMask[..., idx])
        np.testing.assert_array_equal(projections[idx], motSVD[:, idx])


def test_empty_motsv_falls_back_to_nan_instead_of_indexerror(pipeline, session_key, tmp_path):
    """Bug 2: `motSv == []` (the whole-recording list, not a per-ROI element)
    must fall back to NaN, not raise IndexError."""
    dataset, motSVD, motMask, _, *_ = _make_dataset(empty_motsv=True)
    key = _insert_facemap_task(pipeline, session_key, tmp_path, dataset, recording_id=1)

    pipeline.fbe.FacialSignal().make(key)  # must not raise

    singular_values, motmasks = (pipeline.fbe.FacialSignal.MotionSVD() & key).fetch(
        "singular_value", "motmask", order_by="component_id"
    )
    assert all(np.isnan(v) for v in singular_values)
    for idx in range(motSVD.shape[-1]):
        np.testing.assert_array_equal(motmasks[idx], motMask[..., idx])


def test_movie_svd_actually_populates(pipeline, session_key, tmp_path):
    """Bug 3: with do_mov_svd on, MovieSVD must receive rows -- with the right
    field name (movmask) and without stealing MotionSVD's entries."""
    dataset, motSVD, motMask, motSv, movSVD, movMask, movSv = _make_dataset(with_movie=True)
    key = _insert_facemap_task(pipeline, session_key, tmp_path, dataset, recording_id=2)

    pipeline.fbe.FacialSignal().make(key)

    mot_rows = (pipeline.fbe.FacialSignal.MotionSVD() & key).fetch(as_dict=True)
    mov_rows = (pipeline.fbe.FacialSignal.MovieSVD() & key).fetch(as_dict=True)

    n_components = motSVD.shape[-1]
    assert len(mot_rows) == n_components, "MotionSVD must still populate when MovieSVD also runs"
    assert len(mov_rows) == n_components, "MovieSVD must receive its own rows, not stay empty"

    mov_rows.sort(key=lambda r: r["component_id"])
    for idx, row in enumerate(mov_rows):
        assert row["singular_value"] == pytest.approx(movSv[idx])
        np.testing.assert_array_equal(row["movmask"], movMask[..., idx])
        np.testing.assert_array_equal(row["projection"], movSVD[:, idx])
