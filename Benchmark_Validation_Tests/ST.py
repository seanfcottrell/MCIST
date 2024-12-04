import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
#from MCSER_GraphST import MCSER_GraphST
#from MCSER_GATE import MCSER_GATE 
from MCSER_SpaceFlow import MCSER_SpaceFlow
from sklearn.metrics.cluster import normalized_mutual_info_score

#For BaristaSeq, we combine topological features with the STAGATE embedding
#Load Data
section_id = sys.argv[1]
adata = sc.read('/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/SpatialTranscriptomics/'+section_id+'.h5ad')
n = adata.obs['OHE_labels'].nunique()

####################################################################################################
###### GraphST ######
# Mclust
#adata2 = adata.copy()
#adata2 = MCSER_GraphST(adata = adata2, n_clusters = n, clustering_algo='Mclust')

#obs_df = adata2.obs.dropna()
#NMI_Mclust_GraphST = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['OHE_labels'])
#print('MCIST Mclust GraphST NMI = %.5f' %NMI_Mclust_GraphST)

# Leiden
#adata3 = adata.copy()
#adata3 = MCSER_GraphST(adata = adata3, n_clusters = n, clustering_algo='Leiden')

#obs_df = adata3.obs.dropna()
#NMI_Leiden_GraphST = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['OHE_labels'])
#print('MCIST Leiden GraphST NMI = %.5f' %NMI_Leiden_GraphST)

####################################################################################################
###### GATE ######
# Mclust
#adata4 = adata.copy()
#adata4 = MCSER_GATE(adata = adata4, n_clusters = n, spatial_rad_cutoff = 2, clustering_algo='Mclust')

#obs_df = adata4.obs.dropna()
#NMI_Mclust_GATE = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['OHE_labels'])
#print('MCIST Mclust GATE NMI = %.5f' %NMI_Mclust_GATE)

# Leiden
#adata5 = adata.copy()
#adata5 = MCSER_GATE(adata = adata3, n_clusters = n, spatial_rad_cutoff = 2, clustering_algo='Leiden')

#obs_df = adata5.obs.dropna()
#NMI_Leiden_GATE = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['OHE_labels'])
#print('MCIST Leiden GATE NMI = %.5f' %NMI_Leiden_GATE)

####################################################################################################
###### SpaceFlow ######
# Mclust 
adata6 = adata.copy()
adata6 = MCSER_SpaceFlow(adata = adata6, n_clusters = n, clustering_algo='Mclust')

obs_df = adata6.obs.dropna()
NMI_Mclust_SpaceFlow = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['OHE_labels'])
print('MCIST Mclust SpaceFlow NMI = %.5f' %NMI_Mclust_SpaceFlow)
# Leiden
adata7 = adata.copy()
adata7 = MCSER_SpaceFlow(adata = adata7, n_clusters = n, clustering_algo='Leiden')

obs_df = adata7.obs.dropna()
NMI_Leiden_SpaceFlow = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['OHE_labels'])
print('MCIST Leiden SpaceFlow NMI = %.5f' %NMI_Leiden_SpaceFlow)
####################################################################################################

x = adata6.obs['new_x']
y = adata6.obs['new_y']
fig = plt.figure(figsize=(6, 6))
scatter = plt.scatter(x, y, c=adata6.obs['MCSER_spatial_domains'].values, cmap='tab10', s=55)  
plt.title(' ')
plt.gca().invert_yaxis()  
plt.gca().set_aspect('equal', 'box')  

plt.xticks([])  
plt.yticks([])  
plt.tight_layout()
plt.savefig('H1_MCSIT_sf.png')

#Write results to CSV for easy check
results = {
        'Dataset': [section_id],
        'MCIST SpaceFlow Mclust NMI': [NMI_Mclust_SpaceFlow],
        'MCIST SpaceFlow Leiden NMI': [NMI_Leiden_SpaceFlow]
    }
        #'MCIST GraphST Mclust NMI': [NMI_Mclust_GraphST],
        #'MCIST GraphST Leiden NMI': [NMI_Leiden_GraphST],
        #'MCIST GATE Mclust NMI': [NMI_Mclust_GATE],
        #'MCIST GATE Leiden NMI': [NMI_Leiden_GATE]

results_df = pd.DataFrame(results)
file_path = 'ST_SF_results.csv'
file_exists = os.path.exists(file_path)
results_df.to_csv(file_path, mode='a', index=False, header=not file_exists)
