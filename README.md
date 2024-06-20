# TAST-Topological-Analysis-of-Spatial-Transcriptomics

TAST combines the benefits of spatially aware deep learning for spatial transcriptomics with the predictive power of a description of the intrinsic geometrical structure inherent in the gene expression space. The manifold structure informed features are generated via Topological PCA. Any deep learning module can be utilized in TAST, but in this repository we present code for combining TAST with SpaceFlow, STAGATE, and GraphST.

In this repository we present the files needed to reproduce the benchmarking results stated in our paper: Benchmarking the Topological Analysis of Spatial Transcriptomics. The code for Topological PCA and TAST is provided. Each of the deep learning modules must be downloaded from their original repositories to be used in TAST. On the Visium DLPFC, ST, MERFISH, and BaristaSeq data, we utilized STAGATE available at https://github.com/zhanglabtools/STAGATE. On the STARmap data, we utilized SpaceFlow available at https://github.com/hongleir/SpaceFlow. On the StereoSeq data, we utilized GraphST available at https://github.com/JinmiaoChenLab/GraphST. 

# Benchmark Validation Tests
TAST relies on the Numpy, Scanpy, SciKit-Learn, JobLib, and Mclust packages, and is compatible with the versions used in each of the above stated models. Upon downloading the DL models, one may need to update the filepaths to reflect your workspace.  For separate deep learning modules it is recommended to create and activate separate virtual environments. Once your workspace is properly set, you can run the tests for each dataset by specifying the argument (i.e. python ST.py 'H1'). 

# Data 
All of the 37 datasets referenced in the paper are publicly available in .h5ad format at our lab website https://users.math.msu.edu/users/weig/. 

# Tutorial
In the Tutorial folder we present a brief example of the TAST workflow for Spatial Domain Detection and pathway enrichment analysis on the ST HER2 positive breast tumor data, as well as a trajectory inference analysis on the DLPFC data. 

# RS Index
The optional optimization of TAST (as opposed to ensemble learning) relies on computation of the Residue Similarity Index metric. The code for calculating this metric can be downloaded at: https://github.com/hozumiyu/CCP-for-Single-Cell-RNA-Sequencing/tree/main/codes/RSI. 
