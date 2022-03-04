import pathlib
import inspect
import importlib
import datajoint as dj
from element_interface.utils import find_full_path, dict_to_uuid, find_root_directory

schema = dj.schema()
_linking_module = None


def activate(fm_schema_name, *, create_schema=True, create_tables=True,
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
                + get_fm_root_data_dir() -> list
                    Retrieve the root data director(y/ies) with face
                    recordings for all subject/sessions.
                    :return: a string for full path to the root data directory
                + get_fm_processed_data_dir(session_key: dict) -> str
                    Optional function to retrive the desired output directory
                    for Facemap files for a given session. If unspecified,
                    output stored in the session video folder, per Facemap default
                    :return: a string for the absolute path of output directory
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(linking_module),\
        "The argument 'dependency' must be a module's name or a module"
    assert hasattr(linking_module, 'get_fm_root_data_dir'),\
        "The linking module must specify a lookup function for a root data directory"

    global _linking_module
    _linking_module = linking_module

    # activate
    schema.activate(fm_schema_name, create_schema=create_schema,
                    create_tables=create_tables,
                    add_objects=_linking_module.__dict__)


# -------------- Functions required by element-facemap ---------------

def get_fm_root_data_dir() -> list:
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
    root_directories = _linking_module.get_fm_root_data_dir()
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [root_directories]

    if hasattr(_linking_module, 'get_fm_processed_data_dir'):
        root_directories.append(_linking_module.get_fm_processed_data_dir(None))

    return root_directories


def get_fm_processed_data_dir() -> str:
    """
    If specified by the user, this function provides DeepLabCut with an output
    directory for processed files. If unspecified, output files will be stored
    in the session directory 'videos' folder, per DeepLabCut default
    get_dlc_processed_data_dir -> str
        This user-provided function specifies where DeepLabCut output files
        will be stored.
    """
    if hasattr(_linking_module, 'get_fm_processed_data_dir'):
        return _linking_module.get_fm_processed_data_dir()
    else:
        return get_fm_root_data_dir()[0]


# ----------------------------- Table declarations ----------------------


@schema
class VideoRecording(dj.Manual):
    definition = """
    -> Session
    -> Device
    recording_id: int
    ---
    recording_start_time: datetime
    """

    class File(dj.Part):
        definition = """
        -> master
        file_path: varchar(255)  # filepath of video, relative to root data directory
        """
    
    class Info(dj.Part):
        definition = """
        -> master
        ---
        px_height         : smallint  # height in pixels
        px_width          : smallint  # width in pixels
        fps               : float     # (Hz) frames per second
        """


@schema
class FacemapParamSet(dj.Lookup):
    definition = """
    # Parameters used to run the Facemap algorithm
    paramset_idx    : smallint
    ---
    paramset_desc: varchar(128)
    param_set_hash  : uuid        # hash identifying this parameterset
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(cls, processing_method: str, paramset_idx: int,
                          paramset_desc: str, params: dict):
        param_dict = {'processing_method': processing_method,
                      'paramset_idx': paramset_idx,
                      'paramset_desc': paramset_desc,
                      'params': params,
                      'param_set_hash': dict_to_uuid(params)}
        q_param = cls & {'param_set_hash': param_dict['param_set_hash']}

        if q_param:  # If the specified param-set already exists
            pname = q_param.fetch1('paramset_idx')
            if pname == paramset_idx:  # If the existed set has the same name: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    'The specified param-set already exists - name: {}'.format(pname))
        else:
            cls.insert1(param_dict)


@schema
class ProcessingTask(dj.Manual):
    definition = """  # Manual table for defining a processing task ready to be run
    -> VideoRecording
    -> FacemapParamSet
    ---
    processing_output_dir: varchar(255)         #  output directory that will keep the facemap results
    task_mode='load': enum('load', 'trigger')   # 'load': load computed analysis results, 'trigger': trigger computation
    """


@schema
class Processing(dj.Computed):
    definition = """  # Processing Procedure
    -> ProcessingTask
    ---
    processing_time     : datetime  # time of generation of the facemap results
    package_version=''  : varchar(16)
    """
    
    
    # Run processing only on Scan with ScanInfo inserted
    @property
    def key_source(self):
        return ProcessingTask & VideoRecording.File

    def make(self, key):
        task_mode = (ProcessingTask & key).fetch1('task_mode')

        output_dir = (ProcessingTask & key).fetch1('processing_output_dir')
        if not output_dir:
            output_dir = ProcessingTask.infer_output_dir(key, relative=True, mkdir=True)
            # update processing_output_dir
            ProcessingTask.update1({**key, 'processing_output_dir': output_dir.as_posix()})

        if task_mode == 'load':
            method, imaging_dataset = get_loader_result(key, ProcessingTask)
            if method == 'suite2p':
                if (scan.ScanInfo & key).fetch1('nrois') > 0:
                    raise NotImplementedError(f'Suite2p ingestion error - Unable to handle'
                                              f' ScanImage multi-ROI scanning mode yet')
                suite2p_dataset = imaging_dataset
                key = {**key, 'processing_time': suite2p_dataset.creation_time}
            elif method == 'caiman':
                caiman_dataset = imaging_dataset
                key = {**key, 'processing_time': caiman_dataset.creation_time}
            else:
                raise NotImplementedError('Unknown method: {}'.format(method))



# ---------------- HELPER FUNCTIONS ----------------


_table_attribute_mapper = {'ProcessingTask': 'processing_output_dir'}


def get_loader_result(key, table):
    """
    Retrieve the loaded processed imaging results from the fissa loader
        :param key: the `key` to one entry of ProcessingTask
        :param table: the class defining the table to retrieve
         the loaded results from (e.g. ProcessingTask)
        :return: a loader object of the loaded results
         (e.g. suite2p.Suite2p, caiman.CaImAn, etc.)
    """
    output_dir = (FacemapParamSet * table & key).fetch1(
        'processing_method', _table_attribute_mapper[table.__name__])

    output_path = find_full_path(get_fm_root_data_dir(), output_dir)

    from element_interface import fissa_loader
    
    loaded_dataset = fissa_loader(output_path)

    return loaded_dataset