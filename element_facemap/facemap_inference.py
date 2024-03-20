import pickle
import shutil
import importlib
import inspect
import datajoint as dj
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from element_interface.utils import find_full_path, find_root_directory

from . import facial_behavior_estimation as fbe
from .facial_behavior_estimation import (
    get_facemap_root_data_dir,
    get_facemap_processed_data_dir,
)

schema = dj.schema()

_linking_module = None


def activate(
    facemap_model_schema_name: str,
    fbe_schema_name: str = None,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module=None,
):
    """Activate schema.

    Args:
        facemap_model_schema_name (str): Schema name on the database server to activate the
            `facemap_inference` schema of element-facemap
        fbe_schema_name (str):  Schema name on the database server to activate the 'facial_behavioral_estimation
        create_schema (bool): When True (default), create schema in the database if it
            does not yet exist.
        create_tables (bool): When True (default), create tables in the database if
            they do not yet exist.
        linking_module (str): A module name or a module containing the required
            dependencies to activate the `facial_behavior_estimation` module:

    Dependencies:
    Upstream tables:
        + Session: A parent table to VideoRecording, identifying a recording session.
        + Equipment: A parent table to VideoRecording, identifying video recording equipment.
        + VideoRecording: A parent table to FacemapInferenceTask, identifying videos to be used in inference.
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

    fbe.activate(
        fbe_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        linking_module=linking_module,
    )
    schema.activate(
        facemap_model_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# ----------------------------- Table declarations ----------------------
@schema
class BodyPart(dj.Lookup):
    """Body parts tracked by Facemap models.

    Attributes:
        body_part (str): Body part short name.
        body_part_description (str, optional): Detailed body part description.

    """

    definition = """
    body_part                : varchar(32)
    ---
    body_part_description='' : varchar(1000)
    """

    # Facemap Default BodyPart list
    contents = [
        ("eye(back)", ""),
        ("eye(bottom)", ""),
        ("eye(front)", ""),
        ("eye(top)", ""),
        ("lowerlip", ""),
        ("mouth", ""),
        ("nose(bottom)", ""),
        ("nose(r)", ""),
        ("nose(tip)", ""),
        ("nose(top)", ""),
        ("nosebridge", ""),
        ("paw", ""),
        ("whisker(I)", ""),
        ("whisker(III)", ""),
        ("whisker(II)", ""),
    ]


@schema
class FacemapModel(dj.Manual):
    """Trained Models stored for facial pose inference.

    Attributes:
        model_id (int): User specified unique model ID associated with a model.
        model_name (str): Name of model.
        model_description (str, optional): Detailed model description.
    """

    definition = """
    model_id                  : int              # user assigned ID associated with a unique model
    ---
    model_name                : varchar(64)      # name of model 
    model_description=''      : varchar(1000)    # optional model description
    """

    class BodyPart(dj.Part):
        """Body parts associated with a given model

        Attributes:
            body_part (str): Body part name.
            body_part_description (str): Detailed body part description.
        """

        definition = """
        -> master
        -> BodyPart
        """

    class File(dj.Part):
        """Relative paths of facemap models with respect to facemap_root_data_dir

        Attributes:
            FacemapModel (foreign key): Primary key from FacemapModel.
            model_file (attach): Facemap model file.

        """

        definition = """
        -> master
        ---
        model_file: attach      # model file attachment. Stored as binary in database.
        """

    @classmethod
    def insert_new_model(
        cls,
        model_id: int,
        model_name: str,
        model_description: str,
        full_model_path: str,
    ):
        """Insert a new model into the FacemapModel table and relevant part tables.

        Args:
            model_id (int): User specified unique model ID associated with a model.
            model_name (str): Name of model.
            model_description (str): Detailed model description.
            full_model_path (str): Full path to the model file.
        """
        cls.insert1(
            dict(
                model_id=model_id,
                model_name=model_name,
                model_description=model_description,
            )
        )

        cls.BodyPart.insert(
            [
                dict(
                    model_id=model_id,
                    body_part=part,
                )
                for part in BodyPart.fetch("body_part")
            ]
        )

        cls.File.insert1(
            dict(
                model_id=model_id,
                model_file=full_model_path,
            ),
        )


@schema
class FacemapInferenceTask(dj.Manual):
    """A pairing of video recordings and Facemap model.

    Attributes:
        fbe.VideoRecording (foreign key): Primary key from VideoRecording table.
        FacemapModel (foreign key): Primary key from FacemapModel table.
        facemap_inference_output_dir (str): output dir storing the results of pose analysis.
        task_mode (str): One of 'load' (load computed analysis results) or 'trigger' (trigger computation).
        bbox (longblob, nullable) : Bounding box for cropping the video [x1, x2, y1, y2]. If not set, entire frame is used.
        task_description (str, optional) : Task description.
    """

    definition = """
    # Staging table for pairing of recording and Facemap model.
    -> fbe.VideoRecording
    -> FacemapModel
    ---
    facemap_inference_output_dir    : varchar(255)  # Output directory of processed results of Facemap inference analysis relative to root directory.
    task_description=''             : varchar(128)  # Optional. Additional task description
    task_mode='load'                : enum('load', 'trigger') 
    bbox=null                       : longblob  # list containing bounding box for cropping the video [x1, x2, y1, y2]
    """

    @classmethod
    def infer_output_dir(cls, key, relative=False, mkdir=False):
        """Infer an output directory for an entry in FacemapInferenceTask table.

        Args:
            key (dict): Primary key from the FacemapInferenceTask table.
            relative (bool, optional): If True, facemap_inference_output_dir is returned
            relative to facemap_root_dir. Defaults to True.
            mkdir (bool, optional): If True, create facemap_inference_output_dir. Defaults to True.

        Returns:
            dir (str): A default output directory for inference results (facemap_inference_output_dir
                in FacemapInferenceTask) based on the following convention:
                processed_dir / relative_video_dir / {facemap_recordingid}_{model_id}
                e.g.: sub1/sess1/video_files/facemap_recording_id0_model0
        """
        video_file = (fbe.VideoRecording.File & key).fetch("file_path", limit=1)[0]
        video_dir = find_full_path(get_facemap_root_data_dir(), video_file).parent
        root_dir = find_root_directory(get_facemap_root_data_dir(), video_dir)

        processed_dir = Path(get_facemap_processed_data_dir())
        output_dir = (
            processed_dir
            / video_dir.relative_to(root_dir)
            / f"facemap_recordingid{key['recording_id']}_model{key['model_id']}"
        )

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def generate(
        cls,
        key,
        task_description: str = "",
        task_mode: str = "load",
        bbox: list = [],
    ):
        """Generate a unique pose estimation task for each of the relative_video_paths

        Args:
            key (dict): Primary key from FacemapInferenceTask table
                e.g.: {subject="sub1",session_id=0,recording_id=0,model_id=0}
            relative_video_paths (list): list of relative videos in VideoRecording.File table
            task_mode (str, optional): 'load' or 'trigger. Defaults to 'load'.
            bbox (list, optional): Bounding box for processing. Defaults to [].
        """
        facemap_inference_output_dir = cls.infer_output_dir(key)

        cls.insert1(
            dict(
                **key,
                facemap_inference_output_dir=facemap_inference_output_dir,
                task_description=task_description,
                task_mode=task_mode,
                bbox=bbox,
            ),
        )

    insert_facemap_inference_task = generate


@schema
class FacemapInference(dj.Computed):
    """Perform facemap pose estimation.

    Attributes:
        FacemapInferenceTask (foreign key): Primary key from FacemapInferenceTask.
        inference_completion_time (datetime): Inference completion datetime.
        inference_run_duration (datetime): Duration to inference completion.
        total_frame_count (int): Number of frames in all video files.
    """

    definition = """
    -> FacemapInferenceTask
    ---
    inference_completion_time: datetime  # time of generation of this set of facemap results
    inference_run_duration: float # seconds
    total_frame_count: int          # frame count across all video files          
    """

    class BodyPartPosition(dj.Part):
        """Position of individual body parts by frame index.

        Attributes:
            FacemapInference (foreign key): Primary key from FacemapInference.
            FacemapModel.BodyPart (foreign key): Primary key from FacemapModel.BodyPart.
            x_pos (longblob): X position.
            y_pos (longblob): Y position.
            likelihood (longblob): Model confidence."""

        definition = """ # uses facemap h5 output for body part position
        -> master
        -> FacemapModel.BodyPart
        ---
        x_pos       : longblob      # x position
        y_pos       : longblob      # y position
        likelihood  : longblob      # model evaluated likelihood
        """

    def make(self, key):
        """.populate() method will launch training for each FacemapInferenceTask"""
        # ID model and directories
        task_mode, output_dir = (FacemapInferenceTask & key).fetch1(
            "task_mode", "facemap_inference_output_dir"
        )

        if not output_dir:
            output_dir = FacemapInferenceTask.infer_output_dir(
                key, relative=True, mkdir=True
            )
            # update facemap_inference_output_dir
            FacemapInferenceTask.update1(
                {**key, "facemap_inference_output_dir": output_dir.as_posix()}
            )

        output_dir = find_full_path(fbe.get_facemap_root_data_dir(), output_dir)
        video_files = (FacemapInferenceTask * fbe.VideoRecording.File & key).fetch(
            "file_path"
        )

        video_files = [
            find_full_path(fbe.get_facemap_root_data_dir(), video_file)
            for video_file in video_files
        ]
        vid_name = Path(video_files[0]).stem
        facemap_result_path = output_dir / f"{vid_name}_FacemapPose.h5"
        full_metadata_path = output_dir / f"{vid_name}_FacemapPose_metadata.pkl"

        # Load or Trigger Facemap Pose Estimation Inference
        if (
            facemap_result_path.exists() & full_metadata_path.exists()
        ) or task_mode == "load":  # Load results and do not rerun processing
            (
                body_part_position_entry,
                inference_duration,
                total_frame_count,
                creation_time,
            ) = _load_facemap_results(key, facemap_result_path, full_metadata_path)
            self.insert1(
                {
                    **key,
                    "inference_completion_time": creation_time,
                    "inference_run_duration": inference_duration,
                    "total_frame_count": total_frame_count,
                }
            )
            self.BodyPartPosition.insert(body_part_position_entry)
            return

        elif task_mode == "trigger":
            from facemap.pose import pose as facemap_pose, model_loader

            bbox = (FacemapInferenceTask & key).fetch1("bbox") or []

            # Fetch model(.pt) file attachment to present working directory
            facemap_model_name = (
                FacemapModel.File & f'model_id="{key["model_id"]}"'
            ).fetch1("model_file")

            facemap_model_path = Path.cwd() / facemap_model_name
            models_root_dir = model_loader.get_models_dir()

            # Create Symbolic Links to raw video data files from outbox directory
            video_symlinks = []
            for video_file in video_files:
                video_symlink = output_dir / video_file.name
                if video_symlink.exists():
                    video_symlink.unlink()
                video_symlink.symlink_to(video_file)
                video_symlinks.append(video_symlink.as_posix())

            # copy this model file to the facemap model root directory (~/.facemap/models/)
            shutil.copy(facemap_model_path, models_root_dir)

            # Instantiate Pose object, with filenames specified as video files, and bounding specified in params
            # Assumes GUI to be none as we are running CLI implementation
            pose = facemap_pose.Pose(
                filenames=[video_symlinks],
                model_name=facemap_model_path.stem,
                bbox=bbox,
                bbox_set=bool(bbox),
            )
            pose.run()

            (
                body_part_position_entry,
                inference_duration,
                total_frame_count,
                creation_time,
            ) = _load_facemap_results(key, facemap_result_path, full_metadata_path)
            self.insert1(
                {
                    **key,
                    "inference_completion_time": creation_time,
                    "inference_run_duration": inference_duration,
                    "total_frame_count": total_frame_count,
                }
            )
            self.BodyPartPosition.insert(body_part_position_entry)

    @classmethod
    def get_trajectory(cls, key: dict, body_parts: list = "all") -> pd.DataFrame:
        """Returns a pandas dataframe of coordinates of the specified body_part(s)

        Args:
            key (dict): A DataJoint query specifying one FacemapInferenceEstimation entry.
            body_parts (list, optional): Body parts as a list. If "all", all joints

        Returns:
            df: multi index pandas dataframe with Facemap model name, body_parts
                and x/y coordinates of each body part for a camera_id, similar to
                output of facemap inference data.
        """
        model_name = (FacemapModel & f'model_id={key["model_id"]}').fetch1("model_name")

        if body_parts == "all":
            body_parts = (cls.BodyPartPosition & key).fetch("body_part")
        elif not isinstance(body_parts, list):
            body_parts = list(body_parts)

        df = None
        for body_part in body_parts:
            result_dict = (
                cls.BodyPartPosition
                & {"body_part": body_part}
                & {"recording_id": key["recording_id"]}
                & {"session_id": key["session_id"]}
            ).fetch("x_pos", "y_pos", "likelihood", as_dict=True)[0]
            x_pos = result_dict["x_pos"].tolist()
            y_pos = result_dict["y_pos"].tolist()
            likelihood = result_dict["likelihood"].tolist()
            a = np.vstack((x_pos, y_pos, likelihood))
            a = a.T
            pdindex = pd.MultiIndex.from_product(
                [[model_name], [body_part], ["x", "y", "likelihood"]],
                names=["model", "bodyparts", "coords"],
            )
            frame = pd.DataFrame(a, columns=pdindex, index=range(0, a.shape[0]))
            df = pd.concat([df, frame], axis=1)
        return df


def _load_facemap_results(key, facemap_result_path, full_metadata_path):
    """Load facemap results from h5 and metadata files."""

    from facemap import utils

    with open(full_metadata_path, "rb") as f:
        metadata = pickle.load(f)
    keypoints_data = utils.load_keypoints(metadata["bodyparts"], facemap_result_path)

    # Facemap inference result is a 3D nested array with D1 - (x,y likelihood), D2 - bodyparts, D3 - frame count
    # body parts are ordered the same way as stored
    pose_x_coord = keypoints_data[0, :, :]  # (bodyparts, frames)
    pose_y_coord = keypoints_data[1, :, :]  # (bodyparts, frames)
    pose_likelihood = keypoints_data[2, :, :]  # (bodyparts, frames)

    body_part_position_entries = []
    for b_idx, bodypart in enumerate(metadata["bodyparts"]):
        body_part_position_entries.append(
            {
                **key,
                "body_part": bodypart,
                "x_pos": pose_x_coord[b_idx],
                "y_pos": pose_y_coord[b_idx],
                "likelihood": pose_likelihood[b_idx],
            }
        )

    creation_time = datetime.fromtimestamp(full_metadata_path.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    inference_duration = metadata["total_frames"] * metadata["inference_speed"]

    return (
        body_part_position_entries,
        inference_duration,
        metadata["total_frames"],
        creation_time,
    )
