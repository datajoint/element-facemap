import datajoint as dj
from element_animal import subject
from element_animal.subject import Subject
from element_facemap import fbe, facemap_inference, db_prefix
from element_lab import lab
from element_lab.lab import Lab, Location, Project, Protocol, Source, User
from element_lab.lab import Device
from element_lab.lab import User as Experimenter
from element_session import session_with_datetime as session
from element_session.session_with_datetime import Session
import pathlib


# Declare functions for retrieving data
def get_facemap_root_data_dir():
    data_dir = dj.config.get("custom", {}).get("facemap_root_data_dir")
    return pathlib.Path(data_dir) if data_dir else None


def get_processed_root_data_dir():
    data_dir = dj.config.get("custom", {}).get("facemap_processed_data_dir", None)
    return pathlib.Path(data_dir) if data_dir else None


def get_facemap_processed_data_dir():
    return get_processed_root_data_dir()


# Activate schemas
lab.activate(db_prefix + "lab")
subject.activate(db_prefix + "subject", linking_module=__name__)
session.activate(db_prefix + "session", linking_module=__name__)
fbe.activate(db_prefix + "facemap", linking_module=__name__)
facemap_inference.activate(db_prefix + "facemap_inference", linking_module=__name__)
