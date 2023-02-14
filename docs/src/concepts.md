# Concepts

## Facial Motion Tracking

Studying the inner workings of the brain requires understanding the relationship between
neural activity and the facial activity. Motions in the facial features (e.g. whisker,
eye/pupil, etc) is one type of behavioral activity that can be measured by Facemap.

Motions in the face at the whisker or nose Facial motion Pose estimation is a computer
vision method to track the position, and thereby behavior, of the subject over the
course of an experiment, which can then be paired with neuronal recordings to answer
scientific questions about the brain.

Facemap allows running SVD on a region-of-interest either in the movie itself or in the
motion movie (difference of frames). The region of interest Movie SVD: Frames itself.
Motion SVD: Difference of frames.

ROIs: User can select a rectangular, or ellipsoid shape.

Returns PCA components that give distinct facial features.


## Key Partnerships

Element Facemap was developed in collaboration with Hui Chen Lu's Lab at Indiana University Bloomington.  Our team also works with the Facemap developers to promote integration and interoperability between Facemap and the DataJoint Element Facemap (see [Sustainability Roadmap](https://datajoint.com/docs/community/partnerships/facemap/)).

## Element Features

Through our interviews and direct collaborations, we identified the common motifs to create Element Facemap.

Major features include:
- Storage of input video metadata

## Element Architecture

Each node in the following diagram represents the analysis code in the workflow and the
corresponding tables in the database.  Within the workflow, Element Facemap connects to
upstream Elements including Lab, Animal, and Session. For more detailed documentation on
each table, see the API docs for the respective schemas.

![element-facemap diagram](https://raw.githubusercontent.com/datajoint/element-facemap/main/images/attached_facemap_element.svg)

### `lab` schema ([API docs](./api/workflow_facemap/pipeline/#workflow_facemap.pipeline.Device))

| Table | Description |
| --- | --- |
| Device | Camera metadata |

### `subject` schema ([API docs](../element-animal/api/element_animal/subject))

Although not required, most choose to connect the `Session` table to a `Subject`
  table.

| Table | Description |
| --- | --- |
| Subject | Basic information of the research subject |

### `session` schema ([API docs](../element-session/api/element_session/session_with_datetime))

| Table | Description |
| --- | --- |
| Session | Unique experimental session identifier |

### `facial_behavior_estimation` schema ([API docs](./api/element_facemap/facial_behavior_estimation))

| Table | Description |
| --- | --- |
| VideoRecording | Video(s) from one recording session, for Facial Motion Tracking |
| RecordingInfo | |
| FacemapTask | A set of tasks specifying ... |
| FacemapProcessing | |
| FacialSignal | Parent table for the results of Facemap analysis |
| FacialSignal.Region | Child table for the results of each region |
| FacialSignal.Region.MovieSVD | Child table for the SVD components  |
| FacialSignal.Region.MotionSVD |  |
| FacialSignal.Summary |  |
