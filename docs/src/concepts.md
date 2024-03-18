# Concepts

## Facial Motion Tracking

Neuroscience often involves studying relationships between neural activity and subject behavior. Many mammals, including mice[^1], exhibit facial expressions that
convey information about emotional and neuronal states. Facemap[^2,3] is software designed
to automate the process of quantifying facial movements, including whisker, eye, and pupil
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

[^2]: Syeda, A., Zhong, L., Tung, R., Long, W., Pachitariu, M.*, & Stringer, C.* (2024). Facemap: a framework for modeling neural activity based on orofacial tracking. Nature Neuroscience, 27(1), 187-195.

[^3]: Stringer, C.*, Pachitariu, M.*, Steinmetz, N., Reddy, C. B., Carandini, M., & Harris, K. D. (2019). Spontaneous behaviors drive multidimensional, brainwide activity. Science, 364(6437), eaav7893.

## Element Features

Through our interviews and direct collaborations, we identified the common motifs to
create Element Facemap.

Major features include:

- Ingestion and storage of input video metadata.
- Queueing and triggering of Facemap analysis.
- Ingestion of analysis outcomes as motion and video principle components.


