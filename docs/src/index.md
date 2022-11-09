# Element Facemap

DataJoint element for tracking facial movements of a mouse with
[Facemap](https://github.com/MouseLand/facemap) analysis package. DataJoint Elements
collectively standardize and automate data collection and analysis for neuroscience
experiments. Each Element is a modular pipeline for data storage and processing with
corresponding database tables that can be combined with other Elements to assemble a
fully functional pipeline.

![diagram](https://raw.githubusercontent.com/datajoint/element-facemap/main/images/diagram_flowchart.svg)

Facemap applies singular value decomposition (SVD) to face movie to extract the
principal components (PCs) that explain distinct movements apparent on mouse’s
face. The Element is composed of a single schema for storing data and running analysis:
`facial_behavior_estimation`.

Visit the [Concepts page](./concepts.md) for more information on facial motions and
Element Facemap. To get started with building your data pipeline visit the
[Tutorials page](./tutorials.md).

![element-facemap diagram](https://raw.githubusercontent.com/datajoint/element-facemap/main/images/attached_facemap_element.svg)
