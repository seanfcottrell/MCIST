import squidpy as sq
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mclustpy import mclustpy
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import sys
import os
#from MCIST_GATE import MCIST_GATE
from MCIST_GraphST import MCIST_GraphST
#from MCIST_SpaceFlow import MCIST_SpaceFlow
import seaborn as sns
from sklearn.decomposition import PCA

adata = sq.datasets.seqfish()
n = adata.obs['celltype_mapped_refined'].nunique()

###### GraphST ######
# Mclust
adata2 = adata.copy()
adata2 = MCIST_GraphST(adata = adata2, n_clusters = n, clustering_algo='Mclust')

pca = PCA(n_components=20, random_state=42) 
Q2 = pca.fit_transform(adata2.obsm['emb'].copy())
res = mclustpy(np.real(Q2), G=n, modelNames='EEE', random_seed=2020)
mclust_res = res['classification']
adata2.obs['GraphST_mclust'] = mclust_res

sc.pl.spatial(
        adata2,
        color='MCIST_spatial_domains',
        cmap='tab20',
        frameon=False,     
        legend_loc=None,    
        show=False,
        spot_size=0.05,
        save='MCIST_GraphST_seqFISH'
    )

sc.pl.spatial(
        adata2,
        color='GraphST_mclust',
        cmap='tab20',
        frameon=False,     
        legend_loc=None,    
        show=False,
        spot_size=0.05,
        save='GraphST_seqFISH'
    )

obs_df = adata2.obs.dropna()
MCIST_NMI = normalized_mutual_info_score(obs_df['MCIST_spatial_domains'],  obs_df['celltype_mapped_refined'])
NMI = normalized_mutual_info_score(obs_df['GraphST_mclust'],  obs_df['celltype_mapped_refined'])

####################################################################################################
###### GATE ######
# Mclust
#adata4 = adata.copy()
#adata4 = MCIST_GATE(adata = adata4, n_clusters = n, spatial_rad_cutoff = 0.06, clustering_algo='Mclust')

#Q2 = adata4.obsm['STAGATE']
#res = mclustpy(np.real(Q2), G=n, modelNames='EEE', random_seed=2020)
#mclust_res = res['classification']
#adata4.obs['STAGATE_mclust'] = mclust_res

#sc.pl.spatial(
 #       adata4,
  #      color='MCIST_spatial_domains',
   #     cmap='tab20',
    #    frameon=False,     
     #   legend_loc=None,    
      #  show=False,
       # spot_size=0.05,
        #save='MCIST_GATE_seqFISH'
    #)

#sc.pl.spatial(
 #       adata4,
  #      color='STAGATE_mclust',
   #     cmap='tab20',
    #    frameon=False,     
     #   legend_loc=None,    
      #  show=False,
       # spot_size=0.05,
        #save='STAGATE_seqFISH'
    #)

#obs_df = adata4.obs.dropna()
#MCIST_NMI = normalized_mutual_info_score(obs_df['MCIST_spatial_domains'],  obs_df['celltype_mapped_refined'])
#NMI = normalized_mutual_info_score(obs_df['STAGATE_mclust'],  obs_df['celltype_mapped_refined'])

####################################################################################################
###### SpaceFlow ######
# Mclust 
#adata6 = adata.copy()
#adata6 = MCIST_SpaceFlow(adata = adata6, n_clusters = n, clustering_algo='Mclust')

#Q2 = adata6.obsm['SpaceFlow']
#res = mclustpy(np.real(Q2), G=n, modelNames='EEE', random_seed=2020)
#mclust_res = res['classification']
#adata6.obs['SpaceFlow_mclust'] = mclust_res

#sc.pl.spatial(
 #      color='MCIST_spatial_domains',
  #     frameon=False,     
   #     legend_loc=None,    
    #    show=False,
     #   spot_size=0.05,
      #  save='MCIST_SpaceFlow_seqFISH'
    #)

#sc.pl.spatial(
 #       adata6,
  #      color='SpaceFlow_mclust',
   #     cmap='tab20',
        #frameon=False,     
        #legend_loc=None,     
       # show=False,
      #  spot_size=0.05,
     #   save='SpaceFlow_seqFISH'
    #)

#obs_df = adata6.obs.dropna()
#MCIST_NMI = normalized_mutual_info_score(obs_df['MCIST_spatial_domains'],  obs_df['celltype_mapped_refined'])
#NMI = normalized_mutual_info_score(obs_df['SpaceFlow_mclust'],  obs_df['celltype_mapped_refined'])

####################################################################################################

row = pd.DataFrame([{
    'method'          : 'MCIST_GraphST',
    'Base NMI'        : NMI,
    'MCIST NMI'       : MCIST_NMI,
}])
row.to_csv('seqfish_benchmarks.csv', mode='a', index=False,
           header=not os.path.isfile('seqfish_benchmarks.csv'))
