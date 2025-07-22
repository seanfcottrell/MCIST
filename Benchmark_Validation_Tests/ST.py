import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
#from MCIST_GraphST import MCIST_GraphST
#from MCIST_GATE import MCIST_GATE 
from MCIST_SpaceFlow import MCIST_SpaceFlow
from sklearn.metrics.cluster import normalized_mutual_info_score

#For BaristaSeq, we combine topological features with the STAGATE embedding
#Load Data
section_id = sys.argv[1]
adata = sc.read('./SpatialTranscriptomics/'+section_id+'.h5ad')
n = adata.obs['OHE_labels'].nunique()

####################################################################################################
adata6 = adata.copy()
adata6 = MCIST_SpaceFlow(adata = adata6, n_clusters = n, clustering_algo='Mclust')
#res = res_search_fixed_clus(adata6, n, 'SpaceFlow')
res = mclustpy(np.real(adata6.obsm['SpaceFlow']), G=n, modelNames='EEE', random_seed=2020)
mclust_res = res['classification']
adata6.obs['SF_mclust'] = mclust_res
obs_df = adata6.obs.dropna()
NMI_MCIST = normalized_mutual_info_score(obs_df['MCIST_spatial_domains'],  obs_df['OHE_labels'])
#NMI = normalized_mutual_info_score(obs_df['leiden'],  obs_df['OHE_labels'])
NMI = normalized_mutual_info_score(obs_df['SF_mclust'],  obs_df['OHE_labels'])
print('MCIST NMI = %.5f' %NMI_MCIST)
print('SpaceFlow NMI = %.5f' %NMI)
