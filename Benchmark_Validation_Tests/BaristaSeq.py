import os
import sys
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
from TAST_GATE import TAST_GATE
from sklearn.metrics.cluster import normalized_mutual_info_score

#For BaristaSeq, we combine topological features with the STAGATE embedding
#Load Data
section_id = sys.argv[1]
adata = sc.read('./Data/BaristaSeq/'+section_id+'.h5ad')

#Validation
nmi_list=[1,2,3,4,5,6,7,8,9,10]

for i in range(10):
    adata2 = adata.copy()
    n = adata2.obs['layer'].nunique()
    adata2 = TAST_GATE(adata = adata2, consensus = True, n_clusters = n, spatial_rad_cutoff=50, ground_truth = adata2.obs['layer'])


    NMI = normalized_mutual_info_score(adata2.obs['consensus_mclust'].values,  adata2.obs['layer'].values)
    print('TAST NMI = %.5f' %NMI)
    nmi_list[i] = NMI

    if section_id == 'Slice2' and i == 1:
        plt.rcParams["figure.figsize"] = (8, 8)
        sc.pl.spatial(adata2, color='consensus_mclust', cmap = 'tab20', save='baristaseq_'+section_id+'.png', spot_size=20)

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
