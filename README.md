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

As the diagram depicts, the facemap element starts immediately downstream from ***Session***, ...

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
    + schema name for the probe module
    + schema name for the ephys module

2. Upstream tables
    + Session table: A set of keys identifying a recording session (see [Element-Session](https://github.com/datajoint/element-session)).
    + Device table: A reference table for device, specifying the videorecording reference.

3. Utility functions
    + get_facemap_root_data_dir(): Returns your root data directory.

