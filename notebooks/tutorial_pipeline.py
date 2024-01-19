import os
import pathlib
import datajoint as dj
from element_animal import subject
from element_animal.subject import Subject
from element_lab import lab
from element_lab.lab import Lab, Location, Project, Protocol, Source, User
from element_lab.lab import Device as Equipment
from element_lab.lab import User as Experimenter
from element_session import session_with_datetime as session
from element_session.session_with_datetime import Session
from element_facemap import (
    facial_behavior_estimation as fbe,
    facemap_inference,
)
import element_interface


if "custom" not in dj.config:
    dj.config["custom"] = {}

# overwrite dj.config['custom'] values with environment variables if available

dj.config["custom"]["database.prefix"] = os.getenv(
    "DATABASE_PREFIX", dj.config["custom"].get("database.prefix", "")
)

dj.config["custom"]["facemap_root_data_dir"] = os.getenv(
    "FACEMAP_ROOT_DATA_DIR", dj.config["custom"].get("facemap_root_data_dir", "")
)

db_prefix = dj.config["custom"].get("database.prefix", "")


# Declare functions for retrieving data
def get_facemap_root_data_dir():
    """Retrieve facemap root data directory."""
    facemap_root_dirs = dj.config.get("custom", {}).get("facemap_root_data_dir", None)
    if not facemap_root_dirs:
        return None
    elif isinstance(facemap_root_dirs, (str, pathlib.Path)):
        return [facemap_root_dirs]
    elif isinstance(facemap_root_dirs, list):
        return facemap_root_dirs
    else:
        raise TypeError("`facemap_root_data_dir` must be a string, pathlib, or list")


# Activate schemas
lab.activate(db_prefix + "lab")
Device = lab.Device
subject.activate(db_prefix + "subject", linking_module=__name__)
session.activate(db_prefix + "session", linking_module=__name__)
Session = session.Session

fbe.activate(db_prefix + "facemap", linking_module=__name__)
facemap_inference.activate(db_prefix + "facemap_inference", linking_module=__name__)


def get_session_directory(session_key):
    session_directory = (session.SessionDirectory & session_key).fetch1("session_dir")

    return session_directory
