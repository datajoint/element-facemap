import datajoint as dj
import inspect
import importlib
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from element_interface.utils import find_full_path, dict_to_uuid, find_root_directory

from . import facial_behavior_estimation as fbe
from . import facial_pose_model as facemap_pose


schema = dj.schema()
_linking_module = None


def activate(
    facemap_train_schema_name: str,
    fbe_schema_name: str = None,
    facemap_model_schema_name: str = None, 
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
        facemap_model_schema_name (str): Schema name on the database server to activate the
            `facemap_pose_model` schema of element-facemap
        create_schema (bool): when True (default), create schema in the database if it
                            does not yet exist.
        create_tables (bool): when True (default), create schema tables in the database
                             if they do not yet exist.
        linking_module (str): a module (or name) containing the required dependencies.

    Dependencies:
    Upstream tables:
        + Session: A parent table to VideoRecording, identifying a recording session
        + Equipment: A parent table to VideoRecording, identifying video recording equipment
        + VideoRecording: A parent table to FacemapInferenceTask, identifying videos to be used in inference
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
        linking_module, "get_dlc_root_data_dir"
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
    facemap_pose.activate(
        facemap_model_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
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
        video_set_id (int): Unique ID for each collection of videos."""

    definition = """ # Set of vids in training set
    video_set_id: int
    """

    class File(dj.Part):
        """File IDs and paths in a given TrainVideoSet

        Attributes:
            VideoSet (foreign key): VideoSet key.
            file_path ( varchar(255) ): Path to file on disk relative to root."""

        definition = """ # Paths of training files (e.g., .avi, .mp4 video files, .h5 keypoints data file)
        -> master
        file_id: int
        ---
        file_path: varchar(255)
        """


@schema
class FacemapTrainParamSet(dj.Lookup):
    """Parameters used to train a model, initial ROIs from (_proc.npy)

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
    paramset_desc                 : varchar(128) # Description of parameterset used for 
    param_set_hash                : uuid      # hash identifying this paramset
                                                unique index (param_set_hash)
    params                        : longblob  # numpy array of initially selected ROIs
    """

    @classmethod
    def insert_new_params(
        cls, paramset_desc: str, params: dict, paramset_idx: int = None
    ):
        """
        Insert a new set of training parameters into dlc.TrainingParamSet.

        Args:
            paramset_desc (str): Description of parameter set to be inserted 
            params (dict): Dictionary including all settings to specify model training.
                        Must include shuffle & trainingsetindex b/c not in config.yaml.
                        project_path and video_sets will be overwritten by config.yaml.
                        Note that trainingsetindex is 0-indexed
            paramset_idx (int): optional, integer to represent parameters.
        """

        if paramset_idx is None:
            paramset_idx = (
                dj.U().aggr(cls, n="max(paramset_idx)").fetch1("n") or 0
            ) + 1

        param_dict = {
            "paramset_idx": paramset_idx,
            "paramset_desc": paramset_desc,
            "params": params,
            "param_set_hash": dict_to_uuid(params),
        }
        param_query = cls & {"param_set_hash": param_dict["param_set_hash"]}
        # If the specified param-set already exists
        if param_query:
            existing_paramset_idx = param_query.fetch1("paramset_idx")
            if existing_paramset_idx == int(paramset_idx):  # If existing_idx same:
                return  # job done
        else:
            cls.insert1(param_dict)  # if duplicate, will raise duplicate error



@schema
class FacemapModelTrainingTask(dj.Manual):
    """Staging table for pairing videosets and training parameter sets

    Attributes:
        FacemapTrainVideoSet (foreign key): FacemapTrainVideoSet Key.
        FacemapTrainingParamSet (foreign key): TrainingParamSet key.
        training_task_id (int): Unique ID for training task.
        train_output_dir( varchar(255) ): Relative output directory for trained model 
        refined_model_name ( varchar(32) ): Name for retrained model
        model_id (smallint): Unique Model index to be inserted into FacemapModel table

    """

    definition = """      # Specification for a facemap model training instance
    -> FacemapTrainVideoSet                     # video(s) for training
    -> FacemapTrainingParamSet                  # Initially specified ROIs
    training_task_id                        : smallint
    ---
    train_output_dir                        : varchar(255)  # Trained model output directory
    refined_model_name='refined_model'      : varchar(32)
    model_id                                : smallint      # Model index for insertion into FacemapModel table
    retrain_model_id                        : smallint      # Model index for loading of 
    model_description                       : varchar(255)  # Optional, model desc for insertion into FacemapModel     
    selected_frame_ind=None                 : smallblob     # Array of frames to run training on
    """
    def infer_output_dir(self, key, relative=True, mkdir=True):
        video_file = (fbe.VideoRecording.File & key).fetch("file_path", limit=1)[0]
        video_dir = find_full_path(fbe.get_facemap_root_data_dir(), video_file).parent
        root_dir = find_root_directory(fbe.get_facemap_root_data_dir(), video_dir)

        paramset_idx = (FacemapModelTrainingTask & key).fetch1("paramset_idx")
        processed_dir = Path(fbe.get_facemap_processed_data_dir())
        output_dir = (
            processed_dir / video_dir.relative_to(root_dir) / f"facemap_train_{paramset_idx}"
        )

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir.relative_to(processed_dir) if relative else output_dir


@schema
class FacemapModelTraining(dj.Computed):
    """Automated Model training information.

    Attributes:
        FacemapModelTrainingTask (foreign key): FacemapModelTrainingTask key.
        train_model_time (datetime): Time of creation of newly trained model
        latest_snapshot (int unsigned): Latest exact snapshot index (i.e., never -1).
        config_template (longblob): Stored full config file."""

    definition = """
    -> FacemapModelTrainingTask
    ---
    train_model_time: datetime

    """

    def make(self, key):
        from facemap.pose import pose as facemap_pose
        from facemap import utils
        import cv2
        import torch
        output_dir = find_full_path(fbe.get_facemap_root_data_dir(), output_dir)

        train_fileset = [find_full_path(fbe.get_facemap_root_data_dir(), fp).as_posix() 
                         for fp in (FacemapTrainFileSet.File & 
                                    {'video_set_id': key['video_set_id']}).fetch("file_path")]
        paramset_idx = (FacemapModelTrainingTask & key).fetch('paramset_idx')

        video_suffixes = ['.mp4','.avi']
        h5_filepaths = [f for f in train_fileset if f.endswith('.h5')] 
        video_files = [f for f in train_fileset if any(f.endswith(s) for s in video_suffixes)]

        # Create a pose model object, specifying the video files
        train_model = facemap_pose.Pose(filenames=[video_files])

        # Run pose prediction setup to set facemap default model to train_model.net
        train_model.pose_prediction_setup()


        # Convert videos to images for train input
        pre_selected_frame_ind = (FacemapModelTrainingTask & key).fetch1('selected_frame_ind')

        
        # Only support single video training 
        assert len(video_files) == 1

        video_file = video_files[0]
        if len(pre_selected_frame_ind) == 0: # set selected frames to all frames

            cap = cv2.VideoCapture(video_file)
            selected_frame_ind = np.arange(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))   
        else:
            selected_frame_ind = pre_selected_frame_ind
        image_data = utils.load_images_from_video(video_file, selected_frame_ind)

        # MULTIVIDEO TODO
        # image_data = []
        # for video_file in video_files:
        #     if len(pre_selected_frame_ind) == 0: # set selected frames to all frames

        #         cap = cv2.VideoCapture(video_file)
        #         selected_frame_ind = np.arange(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))   
        #     else:
        #         selected_frame_ind = pre_selected_frame_ind

        #     image_data.append(utils.load_images_from_video(video_file, selected_frame_ind))

        # -- For multivideo image data reshaping
        # cumframes, Ly, Lx, containers = utils.get_frame_details(video_files)
        # LY, LX, sy, sx = utils.video_placement(Ly, Lx)
        # reshaped_videos = utils.multivideo_reshape(image_data, LY, LX, Ly, Lx, sy, sx)  
        
        
        # Can use existing keypoints data stored facemap_pose schema
        # keypoints_data = (facemap_pose.FacemapPoseEstimation.BodyPartPosition).fetch(as_dict=True)
        
        keypoints_file = (FacemapModelTrainingTask & key).fetch('keypoints_filename')

        # This works, but we would need to store Files in the facial pose model as well, 
        keypoints_data = utils.load_keypoints(facemap_pose.BodyPart.contents, keypoints_file)   

        # Model Parameters (fetch from TrainingParamSet as dict)
        training_params = (FacemapTrainParamSet & f'paramset_idx={paramset_idx}').fetch1('params')
        refined_model_name = (FacemapModelTrainingTask & key).fetch1('refined_model_name') # default = "refined_model"

        # Train model using train function defined in Pose class
        train_model.net = train_model.train(image_data[:,:,:,0], 
                                            keypoints_data.T, # needs to be transposed 
                                            int(training_params['epochs']), 
                                            int(training_params['batch_size']), 
                                            float(training_params['learning_rate']), 
                                            int(training_params['weight_decay']),
                                            bbox=training_params['bbox'])
        

        # Alternate (requires more imports, but allows for access to training object that can be used for cross validation)
        from facemap.pose import model_training, datasets

        dataset = datasets.FacemapDataset(
            image_data=image_data,
            keypoints_data=keypoints_data.T,
            bbox=training_params['bbox'],
        )
        # Create a dataloader object for training
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=int(training_params['batch_size']), shuffle=True
        )
        # Use preprocessed data to train the model
        train_model.net = model_training.train(
            dataloader,
            train_model.net,
            int(training_params['epochs']),
            int(training_params['weight_decay']),
        )
        print("Model training complete!")
        return self.net
        


        # Save Refined Model
        model_output_path = output_dir / f'{refined_model_name}.pth'
        torch.save(train_model.net.state_dict(), model_output_path)

        model_id = (FacemapModelTrainingTask & key).fetch1('model_id')
        model_description = (FacemapModelTrainingTask & key).fetch1('model_description')

        # Insert newly trained model results into FacemapModel table
        try:
            model_ids = facemap_pose.FacemapModel.fetch("model_id")
            if len(model_id) == 0 or model_id in model_ids:
                model_id = max(model_ids) + 1
        except ValueError:  # case that nothing has been inserted
            model_id = 0

        model_insert = dict(model_id=model_id, 
                            model_name=refined_model_name, 
                            model_description=model_description)
        model_file_insert = dict(model_id=model_id, model_file=model_output_path)

        facemap_pose.FacemapModel.insert_new_model(model_insert)
        facemap_pose.FacemapModel.File

        train_model_time = datetime.fromtimestamp(model_output_path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        self.insert1(
            {**key, 'train_model_time': train_model_time}
        )
