import datajoint as dj
import inspect
import importlib
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from element_interface.utils import find_full_path, dict_to_uuid, find_root_directory

from . import facial_behavior_estimation as fbe
from . import facemap_inference
from .facial_behavior_estimation import (
    get_facemap_root_data_dir,
    get_facemap_processed_data_dir,
)

schema = dj.schema()
_linking_module = None


def activate(
    facemap_train_schema_name: str,
    fbe_schema_name: str = None,
    facemap_inference_schema_name: str = None,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activate this schema.

    Args:
        facemap_train_schema_name (str): schema name on the database server to activate
            the `facemap_train` schema of element-facemap
        fbe_schema_name (str):  Schema name on the database server to activate the 'facial_behavioral_estimation
        facemap_inference_schema_name (str): Schema name on the database server to activate the
            `facemap_inference` schema of element-facemap
        create_schema (bool): when True (default), create schema in the database if it
                            does not yet exist.
        create_tables (bool): when True (default), create schema tables in the database
                             if they do not yet exist.i
        linking_module (str): a module (or name) containing the required dependencies.

    Dependencies:
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

    # activate facemap inference schema
    facemap_inference.activate(
        facemap_inference_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        linking_module=linking_module,
    )

    # activate facemap train schema
    schema.activate(
        facemap_train_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )


# ----------------------------- Table declarations ----------------------


@schema
class FacemapTrainFileSet(dj.Manual):
    """Collection of files associated with a given training set.

    Attributes:
        file_set_id (int): Unique ID for each collection of training files."""

    definition = """ # Set of vids in training set
    file_set_id: int
    """

    class VideoFile(dj.Part):
        """Video File IDs and paths in a given FacemapTrainFileSet

        Attributes:
            FacemapTrainFileSet (foreign key)   : FacemapTrainFileSet key.
            video_file_id (int)                 : Video File index
            video_file_path ( varchar(255) )    : Path to file on disk relative to root.
        """

        definition = """ # Paths of training files (e.g., .avi, .mp4 video files)
        -> master
        video_file_id: int
        ---
        video_file_path: varchar(255)
        """

    class KeypointsFile(dj.Part):
        """Keypoints File and paths in a given FacemapTrainFileSet

        Attributes:
            FacemapTrainFileSet (foreign key)   : FacemapTrainFileSet key.
            file_path ( varchar(255) )          : Path to file on disk relative to root.
        """

        definition = """ # Paths of training files (e.g.: .h5 keypoints data file)
        -> master
        ---
        file_path: varchar(255)
        """


@schema
class FacemapTrainParamSet(dj.Lookup):
    """Parameters used to train a model

    Attributes:
        paramset_idx (smallint): Index uniqely identifying each paramset.
        paramset_desc ( varchar(128) ): Description of paramset.
        param_set_hash (uuid): Hash identifying this paramset.
        params (longblob): Dictionary of all applicable parameters.
        Note: param_set_hash must be unique."""

    definition = """
    # Parameters to specify a facemap model training instance
    paramset_idx                  : smallint
    ---
    paramset_desc                 : varchar(128) # Optional description of parameterset  
    param_set_hash                : uuid         # hash identifying this paramset
                                                unique index (param_set_hash)
    params                        : longblob     # required model training parameters
    """

    required_parameters = (
        "weight_decay",
        "bbox",
        "learning_rate",
        "epochs",
        "batch_size",
    )

    @classmethod
    def insert_new_params(
        cls, paramset_desc: str, params: dict, paramset_idx: int = None
    ):
        """
        Insert a new set of training parameters into FacemapTrainParamSet.

        Args:
            paramset_desc (str): Description of parameter set to be inserted
            params (dict): Dictionary including all settings to specify model training.
            paramset_idx (int): optional, integer to represent parameters.
        """

        for required_param in cls.required_parameters:
            assert required_param in params, (
                "Missing required parameter: " + required_param
            )

        if paramset_idx is None:
            paramset_idx = (
                dj.U().aggr(cls, n="max(paramset_idx)").fetch1("n") or 0
            ) + 1

        paramset_hash = dict_to_uuid(params)  # store to avoid recompute
        param_query = cls & {"param_set_hash": paramset_hash}

        # If the specified param-set already exists
        if param_query:
            existing_paramset_idx = param_query.fetch1("paramset_idx")
            if existing_paramset_idx == int(paramset_idx):  # If existing_idx same:
                print(f"Paramset already exists at index {paramset_idx}")
                return  # job done
        else:
            cls.insert1(
                dict(
                    paramset_idx=paramset_idx,
                    paramset_desc=paramset_desc,
                    param_set_hash=paramset_hash,
                    params=params,
                ),
            )  # if duplicate, will raise duplicate error


@schema
class FacemapModelTrainingTask(dj.Manual):
    """Staging table for pairing videosets and training parameter sets

    Attributes:
        FacemapTrainFileSet (foreign key): FacemapTrainFileSet Key.
        FacemapTrainParamSet (foreign key): TrainingParamSet key.
        train_output_dir( varchar(255) ): Relative output directory for trained model
        selected_frame_ind (blob) : Array of frames to run training on, if not specified all frames used.
        refined_model_name ( varchar(32) ): Name for retrained model
        retrain_model_id (smallint): Model index, of FacemapModel table, to be used for retraining
        model_description ( varchar(255) ): Optional. Model Description for insertion into FacemapModel

    """

    definition = """      # Specification for a facemap model training instance
    -> FacemapTrainFileSet                  # video(s) and files for training
    -> FacemapTrainParamSet                 # Initially specified ROIs
    ---
    train_output_dir                        : varchar(255)  # Trained model output directory
    selected_frame_ind=null                 : blob          # Optional, array of frames to run training on   
    refined_model_prefix=''                 : varchar(128)  # Specify prefix of finetuned/trained model filepath
    -> [nullable]facemap_inference.FacemapModel.proj(retrain_model_id='model_id')  # Specify retrain_model_id
    model_description=None                  : varchar(255)  # Optional, model desc for insertion into FacemapModel     
    """

    @classmethod
    def infer_output_dir(self, key, relative=True, mkdir=True):
        video_file = (FacemapTrainFileSet.VideoFile & key).fetch(
            "video_file_path", limit=1
        )[0]
        video_dir = find_full_path(get_facemap_root_data_dir(), video_file).parent
        root_dir = find_root_directory(get_facemap_root_data_dir(), video_dir)

        processed_dir = Path(get_facemap_processed_data_dir())
        output_dir = (
            processed_dir
            / video_dir.relative_to(root_dir)
            / f"facemap_train_fileset{key['file_set_id']}_paramset{key['paramset_idx']}"
        )

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def insert_facemap_training_task(
        cls,
        file_set_id,
        paramset_idx,
        refined_model_prefix="",
        model_description=None,
        retrain_model_id=None,
        selected_frame_ind=None,
    ):
        key = {"file_set_id": file_set_id, "paramset_idx": paramset_idx}
        inferred_output_dir = cls().infer_output_dir(key, relative=True, mkdir=True)
        cls.insert1(
            dict(
                **key,
                train_output_dir=inferred_output_dir.as_posix(),
                refined_model_prefix=refined_model_prefix,
                model_description=model_description,
                selected_frame_ind=selected_frame_ind,
                retrain_model_id=retrain_model_id,
            ),
        )


@schema
class FacemapModelTraining(dj.Computed):
    """Automated Model training

    Attributes:
        FacemapModelTrainingTask (foreign key): FacemapModelTrainingTask key.
        train_model_time (datetime): Time of creation of newly trained model
        facemap_model_reference (smallint): Reference to index of facemap_inference.FacemapModel

    """

    definition = """
    -> FacemapModelTrainingTask
    ---
    train_model_time        : datetime      # Time of creation of train model file
    """

    class RetrainedModelFile(dj.Part):
        """Stores newly trained models

        Attributes:
            FacemapModelTraining (foreign key):
        """

        definition = """
        -> master
        -> facemap_inference.FacemapModel.proj(retrain_model_id='model_id')  # link to facemap model table
        ---
        retrain_model_file: attach          # retrained model file attachment 
        """

    def make(self, key):
        from facemap.pose import pose
        from facemap import utils
        import torch
        import cv2

        train_output_dir = (FacemapModelTrainingTask & key).fetch1("train_output_dir")
        output_dir = find_full_path(get_facemap_root_data_dir(), train_output_dir)

        video_files = [
            find_full_path(get_facemap_root_data_dir(), fp).as_posix()
            for fp in (FacemapTrainFileSet.VideoFile & key).fetch("video_file_path")
        ]

        # manually specified .h5 keypoints file
        keypoints_file = [
            find_full_path(get_facemap_root_data_dir(), fp).as_posix()
            for fp in (FacemapTrainFileSet.KeypointsFile & key).fetch("file_path")
        ]

        if len(keypoints_file) > 0:
            keypoints_file = keypoints_file[
                0
            ]  # if multiple keypoints files are specified, select first file

        # Create a pose model object, specifying the video files
        train_model = pose.Pose(filenames=[video_files])  # facemap expects list of list
        train_model.pose_prediction_setup()  # Sets default facemap model as train_model.net, handles empty bbox
        retrain_model_id = (FacemapModelTrainingTask & key).fetch1("retrain_model_id")

        if (
            retrain_model_id is not None
        ):  # Retrain an existing model from the facemap_inference.FacemapModel table
            # Fetch model file attachment so that model_file (.pth) is availible in Path.cwd()
            model_file = (
                facemap_inference.FacemapModel.File & {"model_id": retrain_model_id}
            ).fetch1("model_file")

            # Set train_model object to load preexisting model
            train_model.model_name = model_file

            # Overwrite default train_model.net
            train_model.net.load_state_dict(
                torch.load(model_file, map_location=train_model.device)
            )

            # link model to torch device
            train_model.net.to(train_model.device)

        # Convert videos to images for train input
        pre_selected_frame_ind = (FacemapModelTrainingTask & key).fetch1(
            "selected_frame_ind"
        )

        # Currently, only support single video training
        assert len(video_files) == 1
        video_file = video_files[0]

        # Load video capture to iterate through frames and convert to grayscale
        cap = cv2.VideoCapture(video_file)
        if len(pre_selected_frame_ind) == 0:  # set selected frames to all frames
            selected_frame_indices = np.arange(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        else:
            selected_frame_indices = pre_selected_frame_ind
        frames = []
        for frame_ind in selected_frame_indices:
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_ind:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
            ret, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ret:
                frames.append(gray_frame)
            else:
                print("Error reading frame")
        image_data = np.array(frames)

        # Load image frames from video
        # image_data = utils.load_images_from_video(video_file, selected_frame_ind)

        keypoints_data = utils.load_keypoints(
            list(zip(*facemap_inference.BodyPart.contents))[0], keypoints_file
        )

        # Model Parameters (fetch from TrainingParamSet as dict)
        training_params = (
            FacemapTrainParamSet & f'paramset_idx={key["paramset_idx"]}'
        ).fetch1("params")
        refined_model_prefix = (FacemapModelTrainingTask & key).fetch1(
            "refined_model_prefix"
        )  # default = "refined_model"

        # Train model using train function defined in Pose class
        train_model.train(
            image_data,
            keypoints_data.T,  # needs to be transposed
            int(training_params["epochs"]),
            int(training_params["batch_size"]),
            float(training_params["learning_rate"]),
            int(training_params["weight_decay"]),
            bbox=training_params["bbox"],
        )

        # Save Refined Model
        refined_model_name = f"{refined_model_prefix}_refined_model.pth"
        model_output_path = output_dir / refined_model_name
        train_model.save_model(model_output_path)

        model_description = (FacemapModelTrainingTask & key).fetch1("model_description")

        # Insert newly trained model results into FacemapModel table
        try:
            model_ids = facemap_inference.FacemapModel.fetch("model_id")
            model_id = max(model_ids) + 1
        except ValueError:  # case that nothing has been inserted
            model_id = 0

        facemap_inference.FacemapModel().insert_new_model(
            model_id,
            f"{refined_model_prefix}_refined_model.pth",
            model_description,
            model_output_path,
        )

        train_model_time = datetime.fromtimestamp(
            model_output_path.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")

        self.insert1(
            dict(
                **key,
                train_model_time=train_model_time,
            )
        )

        self.RetrainedModelFile.insert1(
            dict(
                **key,
                retrain_model_id=model_id,
                retrain_model_file=model_output_path,
            ),
        )
