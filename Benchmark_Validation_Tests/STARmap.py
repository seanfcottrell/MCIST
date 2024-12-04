import warnings
warnings.filterwarnings('ignore')
import numpy as np
#import SpaceFlow
import scanpy as sc
import pandas as pd
import os
import sys
#from MCSER_GATE import MCSER_GATE
#from MCSER_GraphST import MCSER_GraphST
from MCSER_SpaceFlow import MCSER_SpaceFlow
import matplotlib.pyplot as plt
from mclustpy import mclustpy
from sklearn.metrics.cluster import normalized_mutual_info_score

#For STARmap
section_id = sys.argv[1]
adata = sc.read('/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/SpatialTranscriptomics/STARmap.h5ad')

#Validation
n = adata.obs['label'].nunique()
####################################################################################################
###### GraphST ######
# Mclust
#adata2 = adata.copy()
#adata2 = MCSER_GraphST(adata = adata2, n_clusters = n, clustering_algo='Mclust')
#res = mclustpy(adata2.obsm['MCSER_emb'], G=n, modelNames='EEE', random_seed=2020)
#mclust_res = res['classification']

#NMI_Mclust_GraphST = normalized_mutual_info_score(mclust_res,  adata2.obs['label'].values)
#print('MCIST Mclust GraphST NMI = %.5f' %NMI_Mclust_GraphST)

# Leiden
#adata3 = adata.copy()
#adata3 = MCSER_GraphST(adata = adata3, n_clusters = n, clustering_algo='Leiden')
#res = mclustpy(adata3.obsm['MCSER_emb'], G=n, modelNames='EEE', random_seed=2020)
#mclust_res = res['classification']

#NMI_Leiden_GraphST = normalized_mutual_info_score(mclust_res,  adata3.obs['label'].values)
#print('MCIST Leiden GraphST NMI = %.5f' %NMI_Leiden_GraphST)

####################################################################################################
###### GATE ######
# Mclust
#adata4 = adata.copy()
#adata4 = MCSER_GATE(adata = adata4, n_clusters = n, spatial_rad_cutoff = 400, clustering_algo='Mclust')
#res = mclustpy(adata4.obsm['MCSER_emb'], G=n, modelNames='EEE', random_seed=2020)
#mclust_res = res['classification']

#NMI_Mclust_GATE = normalized_mutual_info_score(mclust_res,  adata4.obs['label'].values)
#print('MCIST Mclust GATE NMI = %.5f' %NMI_Mclust_GATE)

# Leiden
#adata5 = adata.copy()
#adata5 = MCSER_GATE(adata = adata3, n_clusters = n, spatial_rad_cutoff = 400, clustering_algo='Leiden')
#res = mclustpy(adata5.obsm['MCSER_emb'], G=n, modelNames='EEE', random_seed=2020)
#mclust_res = res['classification']

#NMI_Leiden_GATE = normalized_mutual_info_score(mclust_res,  adata4.obs['label'].values)
#print('MCIST Leiden GATE NMI = %.5f' %NMI_Leiden_GATE)

####################################################################################################
###### SpaceFlow ######
# Mclust 
adata6 = adata.copy()
adata6 = MCSER_SpaceFlow(adata = adata6, n_clusters = n, clustering_algo='Mclust')
res = mclustpy(adata6.obsm['MCSER_emb'], G=n, modelNames='EEE', random_seed=2020)
mclust_res = res['classification']

NMI_Mclust_SpaceFlow = normalized_mutual_info_score(mclust_res,  adata6.obs['label'].values)
print('MCIST Mclust SpaceFlow NMI = %.5f' %NMI_Mclust_SpaceFlow)
# Leiden
adata7 = adata.copy()
adata7 = MCSER_SpaceFlow(adata = adata7, n_clusters = n, clustering_algo='Leiden')
res = mclustpy(adata7.obsm['MCSER_emb'], G=n, modelNames='EEE', random_seed=2020)
mclust_res = res['classification']

NMI_Leiden_SpaceFlow = normalized_mutual_info_score(mclust_res,  adata7.obs['label'].values)
print('MCIST Leiden SpaceFlow NMI = %.5f' %NMI_Leiden_SpaceFlow)

####################################################################################################

#Write results to CSV for easy check
results = {
        'Dataset': [section_id],
        'MCIST SpaceFlow Mclust NMI': [NMI_Mclust_SpaceFlow],
        'MCIST SpaceFlow Leiden NMI': [NMI_Leiden_SpaceFlow]
    }

results_df = pd.DataFrame(results)
file_path = 'STARmap_SF_results.csv'
file_exists = os.path.exists(file_path)
results_df.to_csv(file_path, mode='a', index=False, header=not file_exists)
