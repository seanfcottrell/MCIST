import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.metrics.cluster import normalized_mutual_info_score
from MCIST_GATE import MCIST_GATE

#Load Data
section_id = sys.argv[1]
adata = sc.read('/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/SpatialTranscriptomics/'+section_id+'.h5ad')

#Validation
n = adata.obs['layer'].nunique()
####################################################################################################
adata4 = adata.copy()
adata4 = MCIST_GATE(adata = adata4, spatial_rad_cutoff = 150, clustering_algo='Mclust')
res = mclustpy(np.real(adata4.obsm['STAGATE']), G=n, modelNames='EEE', random_seed=2020)
mclust_res = res['classification']
adata4.obs['STAGATE_mclust'] = mclust_res
#res = res_search_fixed_clus(adata4, n, 'STAGATE')
obs_df = adata4.obs.dropna()
NMI_Mclust_GATE = normalized_mutual_info_score(obs_df['MCIST_spatial_domains'],  obs_df['layer'])
#NMI = normalized_mutual_info_score(obs_df['leiden'],  obs_df['layer'])
NMI = normalized_mutual_info_score(obs_df['STAGATE_mclust'],  obs_df['layer'])
print('MCIST NMI = %.5f' %NMI_Mclust_GATE)
print('STAGATE NMI = %.5f' %NMI)
