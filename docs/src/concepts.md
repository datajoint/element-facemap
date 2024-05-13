# Concepts

## Facial Motion Tracking

Neuroscience research often involves understanding the connections between neural activity and subject behavior. Many mammals, including mice, exhibit facial expressions that provide insights into their emotional and neuronal states[^1]. Facemap[^2]<sup>,</sup>[^3] is an open-source software developed to streamline the quantification of facial movements such as whisker, eye, and pupil motions through computer vision techniques.

In its initial versions, Facemap empowered researchers to identify regions of interest (ROIs) on the animal's face as either rectangles or ellipses, using example frames from video recordings. The software utilized singular value decomposition on these ROIs across both raw movie frames and frame-wise difference values to detect motion. This process, grounded in principle component analysis, yielded a set of components that represent distinct facial features. To achieve optimal results, it was recommended to use video recordings from fixed cameras to ensure that all captured motions were attributable to the subject's facial movements.[^2]

The latest iteration of Facemap introduces the ability to track keypoints across the animal's face. This feature marks a departure from solely relying on predefined ROIs, allowing for more dynamic and precise analysis of facial expressions and movements.[^3]

+ **KeyPoints Detection**: Facemap now employs cutting-edge machine learning algorithms to automatically detect and track specific facial landmarks, such as the tips of whiskers, the corners of the eyes, and the edges of the mouth. This approach enables a finer-grained analysis of facial expressions, enhancing the software's utility in behavioral neuroscience research.

+ **Dynamic Tracking**: Unlike the static ROIs, keypoints move with the subject across frames. This dynamic tracking ensures that more subtle facial movements are captured, providing richer datasets for analysis.

[^1]: Dolensek, N., Gehrlach, D. A., Klein, A. S., & Gogolla, N. (2020). Facial
    expressions of emotion states and their neuronal correlates in mice. Science,
    368(6486), 89-94.

[^2]: Syeda, A., Zhong, L., Tung, R., Long, W., Pachitariu, M.*, & Stringer, C.* (2024). Facemap: a framework for modeling neural activity based on orofacial tracking. Nature Neuroscience, 27(1), 187-195.

[^3]: Stringer, C.*, Pachitariu, M.*, Steinmetz, N., Reddy, C. B., Carandini, M., & Harris, K. D. (2019). Spontaneous behaviors drive multidimensional, brainwide activity. Science, 364(6437), eaav7893.

## Element Features

Through our interviews and direct collaborations, we identified the common motifs to
create Element Facemap.

Major features include:

- Ingestion and storage of input video metadata.
- Queueing and triggering of Facemap analysis on multiple sessions.
- Ingestion of analysis outcomes as motion and video principle components.
- Ingestion of analysis outcomes from inference of facial keypoints.