# TAST-Topological-Analysis-of-Spatial-Transcriptomics

TAST combines the benefits of spatially aware deep learning for spatial transcriptomics with the predictive power of a description of the intrinsic geometrical structure inherent in the gene expression space. The manifold structure informed features are generated via Topological PCA. Any deep learning module can be utilized in TAST.

In this repository we present the files needed to reproduce the results stated in the paper: ... The code for Topological PCA is provided, while the deep learning modules can be downloaded from their original repositories. On the Visium DLPFC, ST, MERFISH, and BaristaSeq data, we utilized STAGATE available at ... On the STARmap data, we utilized SpaceFlow available at ... On the StereoSeq data, we utilized GraphST available at ... 

# Benchmark Validation Tests
Topological PCA relies only on the Numpy and Scanpy packages, and is compatible with the versions used in each of the above stated models. Upon downloading these models, one needs to update the filepaths in each test file to reflect your workspace.  For separate deep learning modules it is recommended to create and activate separate virtual environments. Once your workspace is properly set, you can run the tests for multiple datasets simultaneously using the run_validation_tests.py SLURM submission script in the Benchmark_Validation_Tests folder. To see the results of an individual dataset, refer to each of the benchmarking files in the Benchmark_Validation_Tests folder.

# Optimization Demonstration
The optimal performance of TAST relies on a partial optimization of the parameter space for tPCA. Specifically, added performance can be gained by tuning the sparsity and geometrical structure capture scalars and the persistent Laplacian connectivity weightings. For simplicity, it is sufficient to perform just four filtrations and tune only three of these weightings as either a 0 or 1, with the first filtration being a 1 by default. We generally found it sufficient to test just three possible values for the other parameters, {1e1, 1e2, 1e3}, resulting in 3 * 3  * 2^3 = 72 total parameter combinations. These can be easily and automatically computed in parallel utilizing the workflow in the Optimization_Demonstration folder. A SLURM submission file, run_parameter_optimization.py, enables multiple tests of TAST to be computed simultaneously, after which the optimal parameter values are automically computed, stored, and input to TAST for future use. The run time of this procedure depends primarily upon the choice of deep learning architecture. For the DLPFC data, the Topological PCA embedding can be computed in under a minute on 1 core of MSU’s High Performance Computing Cluster (HPCC), while STAGATE in general may take up to 10 minutes. 

# Data 
The ST, STARmap, and Visium DLPFC data are provided in .h5ad format at ... The MERFISH, StereoSeq, and BaristaSeq data can be obtained from the pipeline of the latest Spatial Transcriptomics benchmarking study available at ...

# Tutorial
In tutorial.ipynb we present a brief example of the TAST workflow for Spatial Domain Detection and some downstream analysis on the ST HER2 positive breast tumor data sample H1. 

# Citations 
...
