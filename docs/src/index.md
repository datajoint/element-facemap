# Element Facemap

DataJoint Element for modeling neural activity based on orofacial tracking using [Facemap](https://github.com/MouseLand/facemap). This Element supports facemap analysis using distinct keypoints on the mouse face, and computing the singular value decomposition and pupil tracking. DataJoint Elements collectively standardize and automate data collection and analysis for neuroscience experiments. Each Element is a modular pipeline for data storage and processing with corresponding database tables that can be combined with other Elements to assemble a fully functional pipeline.

## Experiment Flowchart

![flowchart](https://raw.githubusercontent.com/datajoint/element-facemap/main/images/flowchart.svg)

## Data Pipeline Diagram

![pipeline](https://raw.githubusercontent.com/datajoint/element-facemap/main/images/pipeline.svg)

+ We have designed two variations of the pipeline to handle different use cases. Displayed above is the latest `facemap_inference` schema. Details on all of the `facemap` schemas can be found in the [Data Pipeline](./pipeline.md) documentation page.

## Getting Started

+ Please fork the [repository](https://github.com/datajoint/element-facemap){:target="_blank"}

+ Clone the repository to your computer

  ```bash
  git clone https://github.com/<enter_github_username>/element-facemap
  ```

+ Install with `pip`

  ```bash
  pip install -e .
  ```

+ [Data Pipeline](./pipeline.md) - Pipeline and table descriptions

+ [Tutorials](./tutorials/index.md) - Start building your data pipeline

+ [Code Repository](https://github.com/datajoint/element-facemap/){:target="_blank"}

## Support

+ If you need help getting started or run into any errors, please contact our team by email at support@datajoint.com.