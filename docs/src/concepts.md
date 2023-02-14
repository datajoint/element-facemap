# Concepts

## Facial Motion Tracking

Neuroscience often involves studying relationship between neural activity and the some
other phenomena. Many mammals, including mice[^1], exhibit facial expressions that
convey information about emotional and neuronal states. Facemap[^2] is software designed
to automate the process of noting facial movements, including whisker, eye, and pupil
movements, using computer vision.

Facemap allows users to designate regions of interest (ROIs) as either rectangles or
ellipses drawn on top of example frames. The software then runs singular value
decomposition on these regions on both the raw movie frames and frame-wise difference
values, which indicate motion. The result of this principle component analysis is a set
of components, each representing distinct facial features. For best results, researchers
should use fixed camera recordings, ensuring that all motion within the ROIs reflects
the subject's facial movement.

[^1]: Dolensek, N., Gehrlach, D. A., Klein, A. S., & Gogolla, N. (2020). Facial
    expressions of emotion states and their neuronal correlates in mice. Science,
    368(6486), 89-94.

[^2]: Syeda, A., Zhong, L., Tung, R., Long, W., Pachitariu, M., & Stringer, C. (2022).
    Facemap: a framework for modeling neural activity based on orofacial tracking.
    bioRxiv, 2022-11

## Key Partnerships

Element Facemap was developed in collaboration with Hui Chen Lu's Lab at Indiana
University Bloomington.  Our team also works with the Facemap developers to promote
integration and interoperability between Facemap and the DataJoint Element Facemap (see
[Sustainability Roadmap](https://datajoint.com/docs/community/partnerships/facemap/)).

## Element Features

Through our interviews and direct collaborations, we identified the common motifs to
create Element Facemap.

Major features include:

- Ingestion and storage of input video metadata.
- Queueing and triggering of Facemap analysis.
- Ingestion of analysis outcomes as motion and video principle components.

## Element Architecture

Each node in the following diagram represents the analysis code in the workflow and the
corresponding tables in the database.  Within the workflow, Element Facemap connects to
upstream Elements including Lab, Animal, and Session. For more detailed documentation on
each table, see the API docs for the respective schemas.

![element-facemap diagram](https://raw.githubusercontent.com/datajoint/element-facemap/main/images/attached_facemap_element.svg)

### `subject` schema ([API docs](../element-animal/api/element_animal/subject))

Although not required, most choose to connect the `Session` table to a `Subject`
  table.

| Table | Description |
| --- | --- |
| Subject | Basic information of the research subject. |

### `session` schema ([API docs](../element-session/api/element_session/session_with_datetime))

| Table | Description |
| --- | --- |
| Session | Unique experimental session identifier. |

### `facial_behavior_estimation` schema ([API docs](./api/element_facemap/facial_behavior_estimation))

| Table | Description |
| --- | --- |
| VideoRecording | Video(s) from one recording session, for Facial Motion Tracking. |
| RecordingInfo | Information extracted from video file. |
| FacemapTask | Staging table for pairing of recording and Facemap parameters before processing.|
| FacemapProcessing | Automated table to execute the Facemap with inputs from FacemapTask. |
| FacialSignal | Results of the Facemap analysis. |
| FacialSignal.Region | Region properties. |
| FacialSignal.Region.MotionSVD | Components of the SVD from motion video. |
| FacialSignal.Region.MovieSVD | Components of the SVD from movie video. |
| FacialSignal.Summary | Average frames for movie and motion videos. |
