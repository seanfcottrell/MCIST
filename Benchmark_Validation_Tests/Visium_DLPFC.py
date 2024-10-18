import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.metrics.cluster import normalized_mutual_info_score
from MCSER_GATE import MCSER_GATE

#For DLPFC, we combine topological features with the STAGATE embedding
#Load Data
section_id = sys.argv[1]
adata = sc.read('./Data/DLPFC/'+section_id+'/'+section_id+'.h5ad')

#Validation
nmi_list=[1,2,3,4,5,6,7,8,9,10]

for i in range(10):
    adata2 = adata.copy()
    n = adata2.obs['layer'].nunique()
    adata2 = MCSER_GATE(adata = adata2, n_clusters = n, spatial_rad_cutoff=150)

    obs_df = adata2.obs.dropna()
    NMI = normalized_mutual_info_score(obs_df['MCSER_spatial_domains'],  obs_df['layer'])
    print('MCSER NMI = %.5f' %NMI)
    nmi_list[i] = NMI

    #Plotting
    if section_id == '151673' and i == 1:
        plt.rcParams["figure.figsize"] = (8, 8)
        sc.pl.spatial(adata2, color='MCSER_spatial_domains', cmap = 'tab20', save='tast_'+section_id+'.png')


#Results
mean = np.mean(nmi_list)
print('Average MCSER NMI:', mean)

#Write results to CSV for easy check
results = {
        'Dataset': [section_id],
        'MCSER NMI': [mean]
    }
results_df = pd.DataFrame(results)
file_path = 'collected_results.csv'
file_exists = os.path.exists(file_path)
results_df.to_csv(file_path, mode='a', index=False, header=not file_exists)
