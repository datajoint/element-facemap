import datajoint as dj
import cv2
import inspect
import importlib
import numpy as np
from pathlib import Path
from datetime import datetime
from element_interface.utils import find_full_path, find_root_directory

schema = dj.schema()

_linking_module = None

def activate(facemap_schema_name, *, create_schema=True, create_tables=True,
             linking_module=None):
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
    assert inspect.ismodule(linking_module),\
        "The argument 'dependency' must be a module's name or a module"
    assert hasattr(linking_module, 'get_facemap_root_data_dir'),\
        "The linking module must specify a lookup function for a root data directory"

    global _linking_module
    _linking_module = linking_module

    # activate
    schema.activate(facemap_schema_name, create_schema=create_schema,
                    create_tables=create_tables,
                    add_objects=_linking_module.__dict__)


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

    if hasattr(_linking_module, 'get_facemap_processed_data_dir'):
        root_directories.append(_linking_module.get_facemap_processed_data_dir(None))

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
    if hasattr(_linking_module, 'get_facemap_processed_data_dir'):
        return _linking_module.get_facemap_processed_data_dir()
    else:
        return get_facemap_root_data_dir()[0]


def get_facemap_video_files(video_key: dict) -> str:
    """
    Retrieve the list of video files (e.g. *.avi) associated with a given video recording
    :param video_key: key of a video recording
    :return: list of Video files' full file-paths
    """
    return _linking_module.get_facemap_video_files(video_key)


# ----------------------------- Table declarations ----------------------


@schema
class VideoRecording(dj.Manual):
    definition = """
    -> Session
    -> Device
    recording_id                : int
    """

    # One VideoRecording can be saved in multiple files
    class File(dj.Part):
        definition = """
        -> master
        file_id         : int           # file id
        ---
        file_path       : varchar(255)  # filepath of video, relative to root data directory
        """


@schema
class RecordingInfo(dj.Imported):
    definition = """
    -> VideoRecording
    ---
    px_height                 : smallint  # height in pixels
    px_width                  : smallint  # width in pixels
    nframes                   : smallint  # number of frames 
    fps                       : float     # (Hz) frames per second
    recording_duration = NULL : float     # video duration in seconds
    recording_time = NULL     : datetime  # Time at the beginning of the recording with respect to the session
    """

    @property
    def key_source(self):
        return VideoRecording & VideoRecording.File

    def make(self, key):
        file_paths = (VideoRecording & key).fetch('file_path')

        nframes = 0
        px_height, px_width, fps = None, None, None

        for file_path in file_paths:
            file_path = (find_full_path(get_facemap_root_data_dir(), file_path)).as_posix()
            # Open the video
            with cv2.VideoCapture(file_path) as cap:
                info = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FPS))
                if px_height is not None:
                    assert (px_height, px_width, fps) == info
                px_height, px_width, fps = info
                nframes += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.insert1({
            **key,
            'px_height': px_height,
            'px_width': px_width,
            'nframes': nframes,
            'fps': fps,  # Assuming that this keyword is set correctly but it's usually not.
            'duration': nframes / fps,                     # TODO Something to think about.
        })
    

@schema
class FacemapParamSet(dj.Manual):
    definition = """
    # Parameters used to run the Facemap on each Recording
    -> VideoRecording
    paramset_idx        :  smallint
    ---
    params              : longblob
    param_set_hash      : uuid          # unique index
    do_mot_svd=1        : bool
    do_mov_svd=0        : bool
    paramset_desc=''    : varchar(128)
    """


@schema
class FacemapTask(dj.Manual):
    definition = """
    # Manual table for defining a processing task ready to be run
    -> FacemapParamSet
    ---
    processing_output_dir   : varchar(255)              # output directory that will keep the facemap results
    task_mode='load'        : enum('load', 'trigger')   # 'load': load computed analysis results, 'trigger': trigger computation
    """

    @classmethod
    def infer_output_dir(cls, video_key,  relative=False, mkdir=False):
        video_file = (FacemapParamSet * VideoRecording.File & video_key).fetch('file_path')[0]  # Take 1 video file
        video_dir = find_full_path(get_facemap_root_data_dir(), video_file)  # find video file's full path
        root_dir = find_root_directory(get_facemap_root_data_dir(), video_dir)  # find the video file's root directory

        paramset_key = FacemapParamSet.fetch1()
        processed_dir = Path(get_facemap_processed_data_dir())
        output_dir = processed_dir / video_dir.relative_to(root_dir) / f'facemap_{paramset_key["paramset_idx"]}'

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def auto_generate_entries(cls, video_key, task_mode):
        ### TODO: fill this part.
        ### Should we do this?

        output_dir = cls.infer_output_dir(video_key, relative=False, mkdir=True)
        pass 
    


@schema
class FacemapProcessing(dj.Computed):
    definition = """
    # Processing Procedure
    -> FacemapTask
    ---
    processing_time     : datetime  # time of generation of the facemap results
    package_version=''  : varchar(16)
    """
    
    # Processing only the VideoRecordings that have their Info inserted.
    @property
    def key_source(self):
        return FacemapTask & RecordingInfo.File

    def make(self, key):
        task_mode = (FacemapTask & key).fetch1('task_mode')

        output_dir = (FacemapTask & key).fetch1('processing_output_dir')

        if not output_dir:
            output_dir = FacemapTask.infer_output_dir(key, relative=True, mkdir=True)
            # update processing_output_dir
            FacemapTask.update1({**key, 'processing_output_dir': output_dir.as_posix()})

        if task_mode == 'trigger':
            from facemap.process import run as facemap_run
            facemap_params, motSVD, movSVD = (FacemapTask * FacemapParamSet & key).fetch1('params', 'do_mot_svd', 'do_mov_svd')

            video_files = (FacemapTask * VideoRecording.File & key).fetch('file_path')
            video_files = [find_full_path(get_facemap_root_data_dir(), video_file) for video_file in video_files]

            facemap_run(video_files, motSVD=motSVD, movSVD=movSVD, proc=facemap_params, savepath=output_dir, parent=None)

        _, facemap_dataset = get_loader_result(key, FacemapTask)
        key = {**key, 'processing_time': facemap_dataset.creation_time}

        self.insert1(key)


@schema
class FacialSignal(dj.Imported):
    definition = """
    # Facial behavioral variables estimated with Facemap
    -> FacemapProcessing
    """

    class Regions(dj.Part):
        definition = """
        -> master
        roi_id              : int               # Region no
        ---
        roi_name=''         : varchar(16)       # user-friendly name of the roi
        xrange              : longblob          # 1d np.array - x pixel indices of the region
        yrange              : longblob          # 1d np.array - y pixel indices of the region
        xrange_bin          : longblob          # 1d np.array - binned x pixel indices of the region
        yrange_bin          : longblob          # 1d np.array - binned y pixel indices of the region
        """

    class Vectors(dj.Part):
        definition = """
        -> master.Regions
        ---
        motsvd              : longblob          # 2d np.array - motion SVD for each region (nframes, components)
        movsvd              : longblob          # 2d np.array - movie SVD for each region (nframes, components)
        motmask_reshape     : longblob          # 3d np array - motion mask (y, x, components) - principle components
        movmask_reshape     : longblob          # 3d np array - movie mask (y, x, components) - principle components
        motion              : longblob          # 1d np.array - absolute motion energies across time (nframes)
        """

    class SingularValues(dj.Part):
        definition = """ # Diagonal elements of the square matrix in 1d
        -> master
        ---
        mot_sv: longblob                       # 1d np.array - singular values for the motion SVD - S_mot
        mov_sv: longblob                       # 1d np.array - singular values of the movie SVD - S_mov
        """

    class Summary(dj.Part):
        definition = """
        -> master
        ---
        sbin                : int               # spatial bin size
        avgframe            : longblob          # 2d np.array - average binned frame
        avgmotion           : longblob          # 2d nd.array - average binned motion frame
        """

    def make(self, key):
        dataset = get_loader_result(key, FacemapTask)

        self.Regions.insert([
            dict(
                key,
                roi_id=i,
                xrange=dataset['rois'][i]['xrange'],
                yrange=dataset['rois'][i]['yrange'],
                xrange_bin=dataset['rois'][i]['xrange_bin'],
                yrange_bin=dataset['rois'][i]['yrange_bin']
            ) for i in range(1, len(dataset['rois']))
        ])

        self.SingularValues.insert1({**key, 'mot_sv': dataset['motSv'], 'mov_sv': dataset['movSv']})

        self.Vectors.insert([
            dict(
                key,
                roi_id=i,
                motsvd=dataset['motSVD'][i],
                movsvd=dataset['motSVD'][i],
                motmask_reshape=dataset['motMask_reshape'][i],
                movmask_reshape=dataset['movMask_reshape'][i],
                motion=dataset['motion'][i]
            ) for i in range(1, len(dataset['rois']))
        ])

        self.Summary.insert1(
            dict(
                key,
                sbin=dataset['sbin'],
                avgframe=dataset['avgframe'],
                avgmotion=dataset['avgmotion'],                
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
    output_dir = (FacemapParamSet * table & key).fetch1('processed_output_dir')

    output_path = find_full_path(get_facemap_root_data_dir(), output_dir)
    
    loaded_dataset = np.load(output_path, allow_pickle=True).item()
    creation_time = (datetime.fromtimestamp(Path(output_path).stat().st_ctime)).isoformat()

    return loaded_dataset, creation_time
