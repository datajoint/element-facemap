# DataJoint Element - Facemap
This repository features DataJoint pipeline design for facial behavior tracking of head-fixed rodent with ***Facemap***.

The pipeline presented here is not a complete pipeline by itself,
 but rather a modular design of tables and dependencies specific to the Facemap workflow. 

This modular pipeline element can be flexibly attached downstream to 
any particular design of experiment session, thus assembling 
a fully functional facemap pipeline.

See [Background](Background.md) for the background information and development timeline.

## Element architecture

![element-facemap diagram](images/attached_facemap_element.svg)

As the diagram depicts, the facemap element starts immediately downstream from ***Session*** and ***Device***.
We provide an example workflow with a [pipeline script](https://github.com/datajoint/workflow-facemap/blob/main/workflow_facemap/pipeline.py) 
that models combining this Element with the corresponding [Element-Session](https://github.com/datajoint/element-session).

+ ***VideoRecording***: All recordings from a given session.
+ ***RecordingInfo***: Meta information of each video recording (number of frames, pixel lengths, fps, etc.)
+ ***FacialSignal***: Set of results from SVD of user defined regions.
+ ***FacialSignal.Region***: Information about each region (region name, pixel indices, etc)
+ ***FacialSignal.MovieSVD***: Principle components, projections, singular values for each movie region
+ ***FacialSignal.MotionSVD***: Principle components, projections, singular values for each motion region
+ ***FacialSignal.Summary***: Average frame, average motion, spatial binning factor


## Installation

+ Install `element-facemap`
     ```
     pip install element-facemap
     ```

+ Upgrade `element-facemap` previously installed with `pip`
     ```
     pip install --upgrade element-facemap
     ```

## Usage

### Element activation

To activate the `element-facemap`, ones need to provide:

1. Schema names
    + schema name for the facial behavior estimation module

2. Upstream tables
    + Session table: A set of keys identifying a recording session (see [Element-Session](https://github.com/datajoint/element-session)).
    + Device table: A Device table to specify a video recording.

3. Utility functions
    + get_facemap_root_data_dir(): Returns your root data directory.
    + get_facemap_processed_data_dir(): Returns your output root data directory
    + get_facemap_video_files(): Returns your video files

### Example usage

See [this project](https://github.com/datajoint/workflow-facemap) for an example usage of this Facemap Element.
