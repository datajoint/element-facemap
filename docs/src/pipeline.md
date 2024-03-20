# Data Pipeline

Each node in the following diagram represents the analysis code in the pipeline and the
corresponding table in the database.  Within the pipeline, Element Facemap
connects to upstream Elements including Lab, Animal, and Session. For more 
detailed documentation on each table, see the API docs for the respective schemas.

The element is composed of two main schemas, `facial_behavior_estimation` and `facemap_inference`. The `facial_behavior_estimation` schema is designed to handle the analysis and ingestion Facemap's SVD analysis for pupil and ROI tracking. The `facemap_inference` schema is designed to handle the analysis and ingestion of Facemap's pose estimation and tracking key points on the mouse face.

## Diagrams

### `facial_behavior_estimation` module

- The `facial_behavior_estimation` schema is designed to handle the analysis and ingestion Facemap's SVD analysis for pupil and ROI tracking.

     ![pipeline](https://raw.githubusercontent.com/datajoint/element-facemap/main/images/pipeline_facial_behavior_estimation.svg)

### `facemap_inference` module

- The `facemap_inference` schema is designed to handle the analysis and ingestion of Facemap's pose estimation and tracking key points on the mouse face.

     ![pipeline](https://raw.githubusercontent.com/datajoint/element-facemap/main/images/pipeline_facemap_inference.svg)

## Table Descriptions

### `lab` schema

- For further details see the [lab schema API docs](https://datajoint.com/docs/elements/element-lab/latest/api/element_lab/lab/)

| Table | Description |
| --- | --- |
| Device | Scanner metadata |

### `subject` schema

- Although not required, most choose to connect the `Session` table to a `Subject`
  table.

- For further details see the [subject schema API docs](https://datajoint.com/docs/elements/element-animal/latest/api/element_animal/subject/)

| Table | Description |
| --- | --- |
| Subject | Basic information of the research subject. |

### `session` schema

- For further details see the [session schema API docs](https://datajoint.com/docs/elements/element-session/latest/api/element_session/session_with_datetime/)

| Table | Description |
| --- | --- |
| Session | Unique experimental session identifier. |

### `facial_behavior_estimation` schema

- For further details see the [facial_behavior_estimation schema API docs](https://datajoint.com/docs/elements/element-facemap/latest/api/element_facemap/facial_behavior_estimation/)

| Table | Description |
| --- | --- |
| VideoRecording | Video(s) from one recording session, for Facial Motion Tracking. |
| RecordingInfo | Information extracted from video file. |
| FacemapTask | Staging table for pairing of recording and Facemap parameters before processing.|
| FacemapProcessing | Automated table to execute Facemap with inputs from FacemapTask. |
| FacialSignal | Results of the Facemap analysis. |
| FacialSignal.Region | Region properties. |
| FacialSignal.MotionSVD | Components of the SVD from motion video. |
| FacialSignal.MovieSVD | Components of the SVD from movie video. |
| FacialSignal.Summary | Average frames for movie and motion videos. |

### `facemap_inference` schema

- For further details see the [facemap_inference schema API docs](https://datajoint.com/docs/elements/element-facemap/latest/api/element_facemap/facemap_inference/)

| Table | Description |
| --- | --- |
| BodyPart | Body parts tracked by Facemap models. |
| FacemapModel | Trained models stored for facial pose inference. |
| FacemapModel.BodyPart | Body parts associated with a given model. |
| FacemapModel.File | File paths to facemap models. |
| FacemapInferenceTask | Staging table for pairing of video recordings and Facemap model before running inference. |
| FacemapInference | Automated table to execute Facemap with inputs from FacemapInferenceTask. |
| FacemapInference.BodyPartPosition | Position of individual body parts. |