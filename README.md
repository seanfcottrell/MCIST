# MCSER- MultiScale Cell Similarity Encoder Representation

MCSER combines the benefits of spatially aware deep learning for spatial transcriptomics with the predictive power of a description of the intrinsic geometrical structure inherent in the gene expression space. The manifold structure informed features are generated via Topological PCA. Any deep learning module can be utilized in TAST, but in this repository we present code for combining TAST with SpaceFlow, STAGATE, and GraphST.

In this repository we present the files needed to reproduce the benchmarking results stated in our paper. The code for Topological PCA and MCSER is provided. Each of the deep learning modules must be downloaded from their original repositories to be used in TAST. On the Visium DLPFC, ST, MERFISH, and BaristaSeq data, we utilized STAGATE available at https://github.com/zhanglabtools/STAGATE. On the STARmap data, we utilized SpaceFlow available at https://github.com/hongleir/SpaceFlow. On the StereoSeq data, we utilized GraphST available at https://github.com/JinmiaoChenLab/GraphST. 

# Benchmark Validation Tests
MCSER relies on the Numpy, Scanpy, SciKit-Learn, JobLib, and Mclust packages, and is compatible with the versions used in each of the above stated models. Upon downloading the DL models, one will need to update the filepaths to reflect your workspace. For example, TAST_SpaceFlow will need to import SpaceFlow from the proper directory. For separate deep learning modules it is recommended to create and activate separate virtual environments. Once your workspace is properly set, you can run the tests for each dataset by specifying the argument (i.e. python ST.py 'H1' or python Visium_DLPFC.py '151673'). 

# Data 
All of the 37 datasets referenced in the paper are publicly available in .h5ad format at our lab website https://weilab.math.msu.edu/DataLibrary/SpatialTranscriptomics/. 

# Tutorial
In the Tutorial folder we present a brief example of the TAST workflow for Spatial Domain Detection and pathway enrichment analysis on the ST HER2 positive breast tumor data, as well as a trajectory inference analysis on the DLPFC data. 

# RS Index
The self-supervised optimization of MCSER relies on computation of the Residue Similarity Index metric. The code for calculating this metric can be downloaded at: https://github.com/hozumiyu/CCP-for-Single-Cell-RNA-Sequencing/tree/main/codes/RSI. 

# Citations 
Sean Cottrell, Yuta Hozumi, and Guo-Wei Wei. K-nearest-neighbors induced topological pca for single cell rna-sequence data analysis. Computers in Biology and Medicine, page 108497, 2024

Kangning Dong and Shihua Zhang. Deciphering spatial domains from spatially resolved transcriptomics with an adaptive graph attention auto-encoder. Nature communications, 13(1):1739, 2022

Yahui Long, Kok Siong Ang, Mengwei Li, Kian Long Kelvin Chong, Raman Sethi, Chengwei Zhong, Hang Xu, Zhiwei Ong, Karishma Sachaphibulkij, Ao Chen, et al. Spatially informed clustering, integration, and deconvolution of spatial transcriptomics with graphst. Nature Communications, 14(1):1155, 2023.

Honglei Ren, Benjamin L Walker, Zixuan Cang, and Qing Nie. Identifying multicellular spatiotemporal organization of cells with spaceflow. Nature communications, 13(1):4076, 2022
