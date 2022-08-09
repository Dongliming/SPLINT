# SPLINT
Materials provided include public OSS datasets, and source code for the learning-based traceability framework SPLINT for the FSE 2022 paper: “Semi-Supervised Pre-processing for Learning-based Traceability Framework on Real-World Software Projects”

DATA and CODE of SPLINT publicly available on https://github.com/Dongliming/SPLINT.git


Using SPLINT, we presented a framework to resolve the data imbalance and sparsity problems associated with learning-based traceability approaches in our FSE2022 paper.


Artifact types

We recovered the trace links for software process artifacts including issues (bugs) and code commits. Meta data, the calculated feature matrix, and the SPLINT source code are provided.


File Structure

1.SPLINT Datasets

For the evaluation of SPLINT, we used six OSS projects and ten industrial projects. For more information about these cases, please do not hesitate to contact us.

We provide the extracted features matrix for 6 OSS projects along with a detailed description of the feature matrix files along with the cleaned issues and commits data.

There is a high level of reproducibility in our datasets.

 
2.SPLINT Code

The structure of the SPLINT code files

- Similarity: different IR methods

- controller.py: the main class of SPLINT

- experiments.py: methods for running experiments

- metric.py: experiments' metric calculation

- textSimilarity.py: calculate features of text similarity using different IR methods

- util.py: methods that are commonly used
