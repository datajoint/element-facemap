"""Minimal database-backed fixtures for testing FacialSignal.make() directly.

This is the first test infrastructure in this repository. No dj_local_conf.json
convention is assumed; point DJ_HOST/DJ_USER/DJ_PASS at any scratch MySQL server
you don't mind creating throwaway schemas on -- everything is dropped at
teardown and lives under a `test_facemap_` schema prefix.
"""
import os
import types
import datetime
import pytest
import datajoint as dj


@pytest.fixture(scope="session", autouse=True)
def dj_conn():
    dj.config["database.host"] = os.environ.get("DJ_HOST", "127.0.0.1:3399")
    dj.config["database.user"] = os.environ.get("DJ_USER", "djtest")
    dj.config["database.password"] = os.environ.get("DJ_PASS", "djtest")
    dj.conn(reset=True)


@pytest.fixture(scope="session")
def pipeline(dj_conn):
    from element_lab import lab
    from element_animal import subject
    from element_session import session_with_datetime as session
    from element_facemap import facial_behavior_estimation as fbe

    prefix = "test_facemap_"
    lm = types.ModuleType("linking_module")
    lm.get_facemap_root_data_dir = lambda: ["/tmp"]
    lm.get_facemap_processed_data_dir = lambda session_key: "/tmp"
    lm.get_experiment_root_data_dir = lambda: ["/tmp"]
    lm.get_session_directory = lambda key: "/tmp"

    lab.activate(f"{prefix}lab")
    lm.Lab = lab.Lab
    lm.Protocol = lab.Protocol
    lm.User = lab.User
    lm.Project = lab.Project
    lm.Source = lab.Source
    lm.Device = lab.Device
    lm.Equipment = lab.Device

    subject.activate(f"{prefix}subject", linking_module=lm)
    lm.Subject = subject.Subject

    session.activate(f"{prefix}session", linking_module=lm)
    lm.Session = session.Session

    fbe.activate(f"{prefix}fbe", linking_module=lm)

    yield types.SimpleNamespace(lab=lab, subject=subject, session=session, fbe=fbe)

    fbe.schema.drop()
    session.schema.drop()
    subject.schema.drop()
    lab.schema.drop()


@pytest.fixture
def session_key(pipeline):
    """One Subject + one Session, reused across tests in this module."""
    subject_id = "test_subject"
    session_datetime = datetime.datetime(2026, 1, 1, 0, 0, 0)
    pipeline.subject.Subject.insert1(
        dict(subject=subject_id, sex="U", subject_birth_date=datetime.date(2020, 1, 1)),
        skip_duplicates=True,
    )
    pipeline.session.Session.insert1(
        dict(subject=subject_id, session_datetime=session_datetime), skip_duplicates=True
    )
    return dict(subject=subject_id, session_datetime=session_datetime)
