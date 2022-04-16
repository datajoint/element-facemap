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

+ Install `element-interface`

    + `element-interface` is a dependency of `element-facemap`, however it is not contained within `requirements.txt`.
     
    ```
    pip install "element-interface @ git+https://github.com/datajoint/element-interface"
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

## Citation

+ If your work uses DataJoint and DataJoint Elements, please cite the respective Research Resource Identifiers (RRIDs) and manuscripts.

+ DataJoint for Python or MATLAB
    + Yatsenko D, Reimer J, Ecker AS, Walker EY, Sinz F, Berens P, Hoenselaar A, Cotton RJ, Siapas AS, Tolias AS. DataJoint: managing big scientific data using MATLAB or Python. bioRxiv. 2015 Jan 1:031658. doi: https://doi.org/10.1101/031658

    + DataJoint ([RRID:SCR_014543](https://scicrunch.org/resolver/SCR_014543)) - DataJoint for < Python or MATLAB > (version < enter version number >)

+ DataJoint Elements
    + Yatsenko D, Nguyen T, Shen S, Gunalan K, Turner CA, Guzman R, Sasaki M, Sitonic D, Reimer J, Walker EY, Tolias AS. DataJoint Elements: Data Workflows for Neurophysiology. bioRxiv. 2021 Jan 1. doi: https://doi.org/10.1101/2021.03.30.437358

    + DataJoint Elements ([RRID:SCR_021894](https://scicrunch.org/resolver/SCR_021894)) - Element Facemap (version < enter version number >)