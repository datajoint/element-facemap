import datajoint as dj
import cv2
import inspect
import importlib
import numpy as np
from glob import glob
from pathlib import Path
from datetime import datetime
from element_interface.utils import find_full_path, find_root_directory

schema = dj.schema()

_linking_module = None


def activate(
    facemap_schema_name, *, create_schema=True, create_tables=True, linking_module=None
):
    """
    activate(schema_name, *, create_schema=True, create_tables=True,
             linking_module=None)
    :param schema_name: schema name on the database server to activate the
                        `facemap` element
    :param create_schema: when True (default), create schema in the database if it
                            does not yet exist.
    :param create_tables: when True (default), create schema in the database if it
                            does not yet exist.
    :param linking_module: a module (or name) containing the required dependencies
                            to activate the `session` element:
        Upstream tables:
            + Session: parent table to VideoRecording, identifying a recording session
            + Device: parent table to VideoRecording, identifying video recording device
        Functions:
            + get_facemap_root_data_dir() -> list
                Retrieve the root data director(y/ies) with face
                recordings for all subject/sessions.
                :return: a string for full path to the root data directory
            + get_facemap_processed_data_dir(session_key: dict) -> str
                Optional function to retrive the desired output directory
                for Facemap files for a given session. If unspecified,
                output stored in the session video folder, per Facemap default
                :return: a string for the absolute path of output directory
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'dependency' must be a module's name or a module"
    assert hasattr(
        linking_module, "get_facemap_root_data_dir"
    ), "The linking module must specify a lookup function for a root data directory"

    global _linking_module
    _linking_module = linking_module

    # activate
    schema.activate(
        facemap_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# -------------- Functions required by element-facemap ---------------


def get_facemap_root_data_dir() -> list:
    """
    It is recommended that all paths in DataJoint Elements stored as relative
    paths, with respect to some user-configured "root" director(y/ies). The
    root(s) may vary between data modalities and user machines
    get_fm_root_data_dir() -> list
        This user-provided function retrieves the possible root data
        director(y/ies) containing continuous face data for all subjects
        and sessions (e.g. acquired video raw files)
        :return: a string for full path to the behavioral root data directory,
         or list of strings for possible root data directories
    """
    root_directories = _linking_module.get_facemap_root_data_dir()
    if isinstance(root_directories, (str, Path)):
        root_directories = [root_directories]

    if hasattr(_linking_module, "get_facemap_processed_data_dir"):
        root_directories.append(_linking_module.get_facemap_processed_data_dir())

    return root_directories


def get_facemap_processed_data_dir() -> str:
    """
    If specified by the user, this function provides Facemapp with an output
    directory for processed files. If unspecified, output files will be stored
    in the session directory 'videos' folder, per Facemap default
    get_fm_processed_data_dir -> str
        This user-provided function specifies where Facemap output files
        will be stored.
    """
    if hasattr(_linking_module, "get_facemap_processed_data_dir"):
        return _linking_module.get_facemap_processed_data_dir()
    else:
        return get_facemap_root_data_dir()[0]


def get_facemap_video_files(video_key: dict) -> str:
    """
    Retrieve the list of video files associated with a given video recording
    :param video_key: key of a video recording
    :return: list of Video files' full file-paths
    """
    return _linking_module.get_facemap_video_files(video_key)


# ----------------------------- Table declarations ----------------------


@schema
class VideoRecording(dj.Manual):
    definition = """
    -> Session
    recording_id                : int
    ---
    -> Device
    """

    # One VideoRecording can be saved in multiple files
    class File(dj.Part):
        definition = """
        -> master
        file_id     : smallint
        ---
        file_path   : varchar(255)  # filepath of video, relative to root directory
        """


@schema
class RecordingInfo(dj.Imported):
    definition = """
    -> VideoRecording
    ---
    px_height             : int       # height in pixels
    px_width              : int       # width in pixels
    nframes               : int       # number of frames
    fps                   : int       # frames per second in Hz
    recording_duration    : float     # video duration in seconds
    recording_time = NULL : datetime  # time at the beginning of the recording
    """

    @property
    def key_source(self):
        return VideoRecording & VideoRecording.File

    def make(self, key):
        file_paths = (VideoRecording.File & key).fetch("file_path")

        nframes = 0
        px_height, px_width, fps = None, None, None

        for file_path in file_paths:
            file_path = (
                find_full_path(get_facemap_root_data_dir(), file_path)
            ).as_posix()

            cap = cv2.VideoCapture(file_path)
            info = (
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FPS)),
            )
            if px_height is not None:
                assert (px_height, px_width, fps) == info
            px_height, px_width, fps = info
            nframes += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        self.insert1(
            {
                **key,
                "px_height": px_height,
                "px_width": px_width,
                "nframes": nframes,
                "fps": fps,  # usually user-defined and wrong
                "recording_duration": nframes / fps,  # see caution above
            }
        )


@schema
class FacemapTask(dj.Manual):
    definition = """
    # Configuration for a facemap analysis task on a particular VideoRecording
    -> VideoRecording
    facemap_task_id             : smallint
    ---
    facemap_output_dir=''       : varchar(255)  # output directory - storing the results of Facemap analysis
    task_mode='load'            : enum('load', 'trigger')
    facemap_params              : longblob  # content of facemap's _proc.npy as dict
    do_mot_svd=1                : bool
    do_mov_svd=0                : bool
    task_description=''         : varchar(128)
    """

    def infer_output_dir(self, key, relative=True, mkdir=True):
        video_file = (VideoRecording.File & key).fetch("file_path", limit=1)[0]
        video_dir = find_full_path(get_facemap_root_data_dir(), video_file).parent
        root_dir = find_root_directory(get_facemap_root_data_dir(), video_dir)

        paramset_key = (FacemapTask & key).fetch1("facemap_task_id")
        processed_dir = Path(get_facemap_processed_data_dir())
        output_dir = (
            processed_dir / video_dir.relative_to(root_dir) / f"facemap_{paramset_key}"
        )

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir.relative_to(processed_dir) if relative else output_dir


@schema
class FacemapProcessing(dj.Computed):
    definition = """
    # Processing Procedure
    -> FacemapTask
    ---
    processing_time     : datetime  # time of generation of the facemap results
    package_version=''  : varchar(16)
    """

    # Process only the VideoRecordings that have their Info inserted.
    @property
    def key_source(self):
        return FacemapTask & VideoRecording.File

    def make(self, key):
        task_mode = (FacemapTask & key).fetch1("task_mode")

        output_dir = (FacemapTask & key).fetch1("facemap_output_dir")
        if not output_dir:
            output_dir = FacemapTask().infer_output_dir(key, relative=True, mkdir=True)
            # update processing_output_dir
            FacemapTask.update1({**key, "facemap_output_dir": output_dir.as_posix()})

        if task_mode == "trigger":
            from facemap.process import run as facemap_run

            params = (FacemapTask & key).fetch1("facemap_params")

            video_files = (FacemapTask * VideoRecording.File & key).fetch("file_path")
            video_files = [
                [
                    find_full_path(get_facemap_root_data_dir(), video_file).as_posix()
                    for video_file in video_files
                ]
            ]

            output_dir = find_full_path(get_facemap_root_data_dir(), output_dir)
            facemap_run(
                video_files,
                sbin=params["sbin"],
                proc=params,
                savepath=output_dir.as_posix(),
                motSVD=params["motSVD"],
                movSVD=params["movSVD"],
            )

        _, creation_time = get_loader_result(key, FacemapTask)
        key = {**key, "processing_time": creation_time}

        self.insert1(key)


@schema
class FacialSignal(dj.Imported):
    definition = """
    # PCA analysis results obtained with Facemap
    -> FacemapProcessing
    """

    class Region(dj.Part):
        definition = """
        -> master
        roi_no        : int         # region no
        ---
        roi_name=''   : varchar(16) # user-friendly name of the roi
        xrange        : longblob    # 1d np.array - x pixel indices
        yrange        : longblob    # 1d np.array - y pixel indices
        xrange_bin    : longblob    # 1d np.array - binned x pixel indices
        yrange_bin    : longblob    # 1d np.array - binned y pixel indices
        motion        : longblob    # 1d np.array - absolute motion energies (nframes)
        """

    class MotionSVD(dj.Part):
        definition = """
        -> master.Region
        pc_no               : int         # principle component (PC) no
        ---
        singular_value=null : float       # singular value corresponding to the PC
        motmask             : longblob    # PC (y, x)
        projection          : longblob    # projections onto the principle component (nframes)
        """

    class MovieSVD(dj.Part):
        definition = """
        -> master.Region
        pc_no               : int         # principle component (PC) no
        ---
        singular_value=null : float       # singular value corresponding to the PC
        movmask             : longblob    # PC (y, x)
        projection          : longblob    # projections onto the principle component (nframes)
        """

    class Summary(dj.Part):
        definition = """
        -> master
        ---
        sbin          : int         # spatial bin size
        avgframe      : longblob    # 2d np.array - average binned frame
        avgmotion     : longblob    # 2d nd.array - average binned motion frame
        """

    def make(self, key):
        dataset, _ = get_loader_result(key, FacemapTask)
        # Only motion SVD region type is supported.
        dataset["rois"] = [x for x in dataset["rois"] if x["rtype"] == "motion SVD"]

        self.insert1(key)

        self.Region.insert(
            [
                dict(
                    key,
                    roi_no=i,
                    xrange=dataset["rois"][i]["xrange"],
                    yrange=dataset["rois"][i]["yrange"],
                    xrange_bin=dataset["rois"][i]["xrange_bin"]
                    if "xrange_bin" in dataset["rois"][i]
                    else None,
                    yrange_bin=dataset["rois"][i]["yrange_bin"]
                    if "yrange_bin" in dataset["rois"][i]
                    else None,
                    motion=dataset["motion"][i + 1],
                )
                for i in range(len(dataset["rois"]))
                if dataset["rois"][i]["rtype"] == "motion SVD"
            ]
        )

        # MotionSVD
        if any(np.any(x) for x in dataset.get("motSVD", [False])):
            entry = [
                dict(
                    key,
                    roi_no=roi_no,
                    pc_no=i,
                    singular_value=dataset["motSv"][i] if "motSv" in dataset else None,
                    motmask=dataset["motMask_reshape"][roi_no + 1][:, :, i],
                    projection=dataset["motSVD"][roi_no + 1][i],
                )
                for roi_no in range(len(dataset["rois"]))
                for i in range(dataset["motSVD"][roi_no + 1].shape[1])
            ]
            self.MotionSVD.insert(entry)

        # MovieSVD
        if any(np.any(x) for x in dataset.get("movSVD", [False])):
            entry = [
                dict(
                    key,
                    roi_no=roi_no,
                    pc_no=i,
                    singular_value=dataset["movSv"][i] if "movSv" in dataset else None,
                    movmask=dataset["movMask_reshape"][roi_no + 1][:, :, i],
                    projection=dataset["movSVD"][roi_no + 1][i],
                )
                for roi_no in range(len(dataset["rois"]))
                for i in range(dataset["movSVD"][roi_no + 1].shape[1])
            ]
            self.MovieSVD.insert(entry)

        # Summary
        self.Summary.insert1(
            dict(
                key,
                sbin=dataset["sbin"],
                avgframe=dataset["avgframe"][0],
                avgmotion=dataset["avgmotion"][0],
            )
        )


# ---------------- HELPER FUNCTIONS ----------------


def get_loader_result(key, table):
    """
    Retrieve the results from the facemap loader
        :param key: the `key` to one entry of FacemapTask
        :param table: the class defining the table to retrieve
         the loaded results from (e.g. FacemapTask)
        :return: output dictionary in the _proc.npy and the creation date time
    """
    output_dir = (table & key).fetch1("facemap_output_dir")

    output_path = find_full_path(get_facemap_root_data_dir(), output_dir)
    output_file = glob(output_path.as_posix() + "/*_proc.npy")[0]

    loaded_dataset = np.load(output_file, allow_pickle=True).item()
    creation_time = datetime.fromtimestamp(Path(output_file).stat().st_ctime)

    return loaded_dataset, creation_time
