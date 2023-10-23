import importlib
import inspect
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List, Tuple
import datajoint as dj
import numpy as np
import pandas as pd
from element_interface.utils import find_full_path, find_root_directory

import os
import pickle
import shutil
from . import facial_behavior_estimation as fbe

schema = dj.schema()

_linking_module = None


def activate(
    facemap_model_schema_name: str,
    fbe_schema_name: str = None,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate schema.

    Args:
        facemap_model_schema_name (str): Schema name on the database server to activate the
            `facemap_pose_model` schema of element-facemap
        fbe_schema_name (str):  Schema name on the database server to activate the 'facial_behavioral_estimation
        create_schema (bool): When True (default), create schema in the database if it
            does not yet exist.
        create_tables (bool): When True (default), create tables in the database if
            they do not yet exist.
        linking_module (str): A module name or a module containing the required
            dependencies to activate the `facial_behavior_estimation` module:

    Dependencies:
    Upstream tables:
        + Session: A parent table to VideoRecording, identifying a recording session
        + Equipment: A parent table to VideoRecording, identifying video recording equipment
        + VideoRecording: A parent table to FacemapPoseEstimationTask, identifying videos to be used in inference
    Functions:
        + get_facemap_root_data_dir() -> list
            Retrieves the root data directory(s) with face recordings for all
            subject/sessions. Returns a string for the full path to the root data directory.
        + get_facemap_processed_data_dir(session_key: dict) -> str
            Optional function to retrieve the desired output directory
            for Facemap files for a given session. If unspecified,
            the output is stored in the video folder for the session, which is the default behavior of Facemap.
            Returns a string of the absolute path of the output directory.
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

    # activate facial behavioral extimation (fbe) schema
    fbe.activate(
        fbe_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        linking_module=linking_module,
    )

    # activate facial pose model schema
    schema.activate(
        facemap_model_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# ----------------------------- Table declarations ----------------------
@schema
class BodyPart(dj.Lookup):
    """Body parts tracked by Facemap models

    Attributes:
        body_part ( varchar(32) ): Body part short name.
        body_part_description ( varchar(1000),optional ): Full description

    """

    definition = """
    body_part                : varchar(32)
    ---
    body_part_description='' : varchar(1000)
    """
    
    # Facemap Default BodyPart list
    contents = [ 
        ("eye(back)", ''),
        ("eye(bottom)", ''),
        ("eye(front)", ''),
        ("eye(top)", ''),
        ("lowerlip", ''),
        ("mouth", ''),
        ("nose(bottom)", ''),
        ("nose(r)", ''),
        ("nose(tip)", ''),
        ("nose(top)", ''),
        ("nosebridge", ''),
        ("paw", ''),
        ("whisker(I)", ''),
        ("whisker(III)", ''),
        ("whisker(II)", ''),
    ]
    
@schema
class FacemapModel(dj.Manual):
    """Trained Models stored for facial pose inference

    Attributes:
        model_id(int) : User specified ID associated with a unique model 
        model_name( varchar(64) ): Name of model, filepath.stem
    """

    definition = """
    model_id   : int                    # user assigned ID associated with a unique model
    ---
    model_name : varchar(64)            # name of model 
    model_description: varchar(1000)    # optional model description
    """

    class BodyPart(dj.Part):
        """Body parts associated with a given model

        Attributes:
            body_part ( varchar(32) ): Body part name, (location specfication)
            body_part_description ( varchar(1000) ): Optional. Longer description."""

        definition = """
        -> master
        -> BodyPart
        """

    class File(dj.Part):
        """Relative paths of facemap models with respect to facemap_root_data_dir

        Attributes:
            FacemapModel (foreign key): Facemap model primary key.
            model_file ( attach ): filepath of facemap model, relative to root data dir
        """

        definition = """
        -> master
        ---
        model_file: attach            # model file attachment
        """
    @classmethod
    def insert_new_model(cls, model_id: int, model_name: str, model_description: str, full_model_path: str):
        facemap_model_insert = dict(
            model_id=model_id, model_name=model_name, model_description=model_description
        )
        FacemapModel.insert1(facemap_model_insert)

        body_part_insert = []
        for bp in BodyPart.contents:
            body_part_insert.append(dict(model_id=model_id, body_part=bp))
        
        # Insert into parent BodyPart table if no entries are present
        if len(cls.BodyPart()) == 0:
            cls.BodyPart.insert(body_part_insert)
        file_insert = dict(model_id=model_id, model_file=full_model_path)

        cls.BodyPart.insert(body_part_insert)
        cls.File.insert1(file_insert)

@schema
class FacemapPoseEstimationTask(dj.Manual):
    """Staging table for pairing of video recordings and Facemap parameters before processing.

    Attributes:
        fbe.VideoRecording (foreign key) : Primary key for VideoRecording table.
        FacemapModel (foreign key) : Primary key for the facemap model table
        pose_estimation_output_dir ( varchar(255), optional) : output dir storing the results
            of pose estimation analysis.
        task_mode (enum) : Default trigger. Load or trigger analysis.
        bbox (longblob) : Bounding box for cropping the video [x1, x2, y1, y2]. If not set, entire frame is used.
        task_description ( varchar(128), optional) : Task description.
    """

    definition = """
    # Staging table for pairing of recording and Facemap parameters before processing.
    -> fbe.VideoRecording
    -> FacemapModel
    ---
    pose_estimation_output_dir=''   : varchar(255)  # output dir - stores results of Facemap Pose estimation analysis
    task_mode='trigger'             : enum('load', 'trigger')
    bbox=None                       : longblob  # list containing bounding box for cropping the video [x1, x2, y1, y2]
    task_description=''             : varchar(128)    
    """

    @classmethod
    def infer_output_dir(cls, key, relative=True, mkdir=True):
        video_file = (fbe.VideoRecording.File & key).fetch("file_path", limit=1)[0]
        video_dir = find_full_path(fbe.get_facemap_root_data_dir(), video_file).parent
        root_dir = find_root_directory(fbe.get_facemap_root_data_dir(), video_dir)

        model_id = (FacemapPoseEstimationTask & key).fetch1("model_id")
        processed_dir = Path(fbe.get_facemap_processed_data_dir())
        output_dir = (
            processed_dir / video_dir.relative_to(root_dir) / f"facemap_{model_id}"
        )

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir.relative_to(processed_dir) if relative else output_dir

    
    @classmethod
    def generate(cls, key, model_id: int, relative_video_paths: list, task_mode: str = "trigger", bbox: list = []):
        """Generate a unique pose estimation task for each of the relative_video_paths

        Args:
            model_id (int): User Specified model identification number 
            session_key (dict):  
            relative_video_paths (list): _description_
            task_mode (str, optional): _description_. Defaults to "trigger".
            bbox (list, optional): _description_. Defaults to [].
        """
        video_paths = [find_full_path(fbe.get_facemap_root_data_dir(), rpath) for rpath in relative_video_paths]
        for vid_path in video_paths:
            device_id = (fbe.VideoRecording & key).fetch('device_id')
            vrec_key = (fbe.VideoRecording & key).fetch('key')

            model_key = (FacemapModel & f"model_id={model_id}").fetch1("KEY")
            pose_estimation_output_dir = cls.infer_output_dir(vrec_key)

            facemap_pose_estimation_task_insert = {
                **vrec_key,
                **model_key,
                "pose_estimation_output_dir": pose_estimation_output_dir,
                "task_mode": task_mode,
                "bbox": bbox,
            }
            cls.insert1(
                facemap_pose_estimation_task_insert
            )
    insert_pose_estimation_task = generate
    
@schema
class FacemapPoseEstimation(dj.Computed):
    """Results of facemap pose estimation

    Attributes:
        FacemapPoseEstimationTask (foreign key): Pose Estimation Task key.
        post_estimation_time (datetime): time of generation of this set of facemap results.
        execution_duration (datetime): duration of model
    """

    definition = """
    -> FacemapPoseEstimationTask
    ---
    pose_estimation_time: datetime  # time of generation of this set of facemap results
    pose_estimation_duration: float # seconds
    total_frame_count: int          # frame count across all video files          
    """

    class BodyPartPosition(dj.Part):
        """Position of individual body parts by frame index

        Attributes:
            PoseEstimation (foreign key): Pose Estimation key.
            FacemapModel.BodyPart (foreign key): Body Part key.
            body_part (longblob): Body part for positional likelihood
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
        """.populate() method will launch training for each PoseEstimationTask"""
        # ID model and directories
        task_mode, output_dir = (FacemapPoseEstimationTask & key).fetch1(
            "task_mode", "pose_estimation_output_dir"
        )

        output_dir = find_full_path(fbe.get_facemap_root_data_dir(), output_dir)
        video_files = (FacemapPoseEstimationTask * fbe.VideoRecording.File & key).fetch(
            "file_path"
        )

        video_files = [
            find_full_path(fbe.get_facemap_root_data_dir(), video_file)
            for video_file in video_files
        ]
        vid_name = Path(video_files[0]).stem
        facemap_result_path = output_dir / f"{vid_name}_FacemapPose.h5"
        full_metadata_path = output_dir / f"{vid_name}_FacemapPose_metadata.pkl"

        # Create Symbolic Links to raw video data files from outbox directory
        video_symlinks = []
        for video_file in video_files:
            video_symlink = output_dir / video_file.name
            if video_symlink.exists():
                video_symlink.unlink()
            video_symlink.symlink_to(video_file)
            video_symlinks.append(video_symlink.as_posix())

        # Trigger Facemap Pose Estimation Inference
        if task_mode == "trigger":
            # Triggering facemap for pose estimation requires:
            # - model_path: full path to the directory containing the trained model
            # - video_filepaths: full paths to the video files for inference
            # - analyze_video_params: optional parameters to analyze video (uses facemap default params)

            from facemap.pose import pose as facemap_pose, model_loader

            # If output files have been created, load the output
            if (
                facemap_result_path.exists() & full_metadata_path.exists()
            ):  # Load results and do not rerun processing
                (
                    body_part_position_entry,
                    inference_duration,
                    total_frame_count,
                    creation_time,
                ) = _load_facemap_results(key, facemap_result_path, full_metadata_path)
                self.insert1(
                    {
                        **key,
                        "pose_estimation_time": creation_time,
                        "pose_estimation_duration": inference_duration,
                        "total_frame_count": total_frame_count,
                    }
                )
                self.BodyPartPosition.insert(body_part_position_entry)
                return

            bbox = (FacemapPoseEstimationTask & key).fetch1("bbox") or []
                
            # Model Name of interest should be specified by user during facemap task params manual update
            model_id = (FacemapPoseEstimationTask & key).fetch("model_id")

            # Fetch model(.pt) file attachment to present working directory
            facemap_model_name = (FacemapModel.File & f'model_id="{model_id}"').fetch1(
                "model_file"
            )

            facemap_model_path = Path.cwd() / facemap_model_name
            models_root_dir = model_loader.get_models_dir()

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

        elif task_mode == "load":
            if (
                facemap_result_path.exists() & full_metadata_path.exists()
            ):  # Load preprocessed inference results
                (
                    body_part_position_entry,
                    inference_duration,
                    total_frame_count,
                    creation_time,
                ) = _load_facemap_results(key, facemap_result_path, full_metadata_path)

        self.insert1(
            {
                **key,
                "pose_estimation_time": creation_time,
                "pose_estimation_duration": inference_duration,
                "total_frame_count": total_frame_count,
            }
        )
        self.BodyPartPosition.insert(body_part_position_entry)

    @classmethod
    def get_trajectory(cls, key: dict, body_parts: list = "all") -> pd.DataFrame:
        """Returns a pandas dataframe of coordinates of the specified body_part(s)

        Args:
            key (dict): A DataJoint query specifying one FacemapPoseEstimation entry.
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
