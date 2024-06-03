# TAST-Topological-Analysis-of-Spatial-Transcriptomics

TAST combines the benefits of spatially aware deep learning for spatial transcriptomics with the predictive power of a description of the intrinsic geometrical structure inherent in the gene expression space. The manifold structure informed features are generated via Topological PCA. Any deep learning module can be utilized in TAST.

In this repository we present the files needed to reproduce the results stated in the paper: ... The code for Topological PCA is provided, while the deep learning modules can be downloaded from their original repositories. On the Visium DLPFC, ST, MERFISH, and BaristaSeq data, we utilized STAGATE available at ... On the STARmap data, we utilized SpaceFlow available at ... On the StereoSeq data, we utilized GraphST available at ... 

Topological PCA relies only on the Numpy and Scanpy packages, and is compatible with the versions used in each of the above stated models. Upon downloading these models, one needs to update the filepaths in each test file to reflect your workspace. Once your workspace is properly set, you can run the files simultaneously using the run_tests.py SLURM submission script. 

The ST, STARmap, and Visium DLPFC data are provided in .h5ad format at ... The MERFISH, StereoSeq, and BaristaSeq data can be obtained from the pipeline of the latest Spatial Transcriptomics benchmarking study available at ...

# Tutorial

In tutorial.ipynb we present a brief example of the TAST workflow for Spatial Domain Detection and some downstream analysis on the ST HER2 positive breast tumor data sample H1. 

# Citations 
...
