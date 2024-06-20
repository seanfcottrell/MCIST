import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.metrics.cluster import normalized_mutual_info_score
from TAST_GATE import TAST_GATE

#For DLPFC, we combine topological features with the STAGATE embedding
#Load Data
section_id = sys.argv[1]
adata = sc.read('./Data/DLPFC/'+section_id+'/'+section_id+'.h5ad')

#Validation
nmi_list=[1,2,3,4,5,6,7,8,9,10]

for i in range(10):
    adata2 = adata.copy()
    n = adata2.obs['layer'].nunique()
    adata2 = TAST_GATE(adata = adata2, consensus = True, n_clusters = n, spatial_rad_cutoff=150, ground_truth = adata2.obs['layer'])

    obs_df = adata2.obs.dropna()
    NMI = normalized_mutual_info_score(obs_df['consensus_mclust'],  obs_df['layer'])
    print('TAST NMI = %.5f' %NMI)
    nmi_list[i] = NMI

    #Plotting
    if section_id == '151673' and i == 1:
        plt.rcParams["figure.figsize"] = (8, 8)
        sc.pl.spatial(adata2, color='consensus_mclust', cmap = 'tab20', save='tast_'+section_id+'.png')


#Results
mean = np.mean(nmi_list)
print('Average TAST NMI:', mean)

#Write results to CSV for easy check
results = {
        'Dataset': [section_id],
        'TAST NMI': [mean]
    }
results_df = pd.DataFrame(results)
file_path = 'collected_results.csv'
file_exists = os.path.exists(file_path)
results_df.to_csv(file_path, mode='a', index=False, header=not file_exists)
