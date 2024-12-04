import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.metrics.cluster import normalized_mutual_info_score
#from MCSER_GraphST import MCSER_GraphST
from MCSER_GATE import MCSER_GATE
#from MCSER_SpaceFlow import MCSER_SpaceFlow

#Load Data
section_id = sys.argv[1]
adata = sc.read('/mnt/home/cottre61/GFP-GAT/STAGATE_pyG/SpatialTranscriptomics/'+section_id+'.h5ad')

#Validation
n = adata.obs['layer'].nunique()
####################################################################################################
###### GraphST ######
# Mclust
#adata2 = adata.copy()
#adata2 = MCSER_GraphST(adata = adata2, n_clusters = n, clustering_algo='Mclust')

#obs_df = adata2.obs.dropna()
#NMI_Mclust_GraphST = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['layer'])
#print('MCIST Mclust GraphST NMI = %.5f' %NMI_Mclust_GraphST)

# Leiden
#adata3 = adata.copy()
#adata3 = MCSER_GraphST(adata = adata3, n_clusters = n, clustering_algo='Leiden')

#obs_df = adata3.obs.dropna()
#NMI_Leiden_GraphST = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['layer'])
#print('MCIST Leiden GraphST NMI = %.5f' %NMI_Leiden_GraphST)

####################################################################################################
###### GATE ######
# Mclust
adata4 = adata.copy()
adata4 = MCSER_GATE(adata = adata4, n_clusters = n, spatial_rad_cutoff = 150, clustering_algo='Mclust')

obs_df = adata4.obs.dropna()
NMI_Mclust_GATE = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['layer'])
print('MCIST Mclust GATE NMI = %.5f' %NMI_Mclust_GATE)

# Leiden
#adata5 = adata.copy()
#adata5 = MCSER_GATE(adata = adata3, n_clusters = n, spatial_rad_cutoff = 150, clustering_algo='Leiden')

#obs_df = adata5.obs.dropna()
#NMI_Leiden_GATE = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['layer'])
#print('MCIST Leiden GATE NMI = %.5f' %NMI_Leiden_GATE)

####################################################################################################
###### SpaceFlow ######
# Mclust 
#adata6 = adata.copy()
#adata6 = MCSER_SpaceFlow(adata = adata6, n_clusters = n, clustering_algo='Mclust')

#obs_df = adata6.obs.dropna()
#NMI_Mclust_SpaceFlow = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['layer'])
#print('MCIST Mclust SpaceFlow NMI = %.5f' %NMI_Mclust_SpaceFlow)
# Leiden
#adata7 = adata.copy()
#adata7 = MCSER_SpaceFlow(adata = adata7, n_clusters = n, clustering_algo='Leiden')

#obs_df = adata7.obs.dropna()
#NMI_Leiden_SpaceFlow = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['layer'])
#print('MCIST Leiden SpaceFlow NMI = %.5f' %NMI_Leiden_SpaceFlow)

####################################################################################################

#Plotting
#if section_id == '151673' and i == 0:
    #plt.rcParams["figure.figsize"] = (8, 8)
    #sc.pl.spatial(adata2, color='MCSER_spatial_domains', cmap = 'tab10', save='mcist_'+section_id+'.png')

#Write results to CSV for easy check
results = {
        #'MCIST GraphST Mclust NMI': [NMI_Mclust_GraphST],
        #'MCIST GraphST Leiden NMI': [NMI_Leiden_GraphST],
        'MCIST GATE Mclust NMI': [NMI_Mclust_GATE],
        #'MCIST GATE Leiden NMI': [NMI_Leiden_GATE]
        }

results_df = pd.DataFrame(results)
file_path = 'DLPFC_collected_results.csv'
file_exists = os.path.exists(file_path)
results_df.to_csv(file_path, mode='a', index=False, header=not file_exists)
