import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from TAST_GATE import TAST_GATE
from sklearn.metrics.cluster import normalized_mutual_info_score

#For MERFISH, we combine topological features with the STAGATE embedding
#Load Data
section_id = sys.argv[1]
adata = sc.read('./Data/MERFISH/'+section_id+'.h5ad')

#Validation
nmi_list=[1,2,3,4,5,6,7,8,9,10]

for i in range(10):
    adata2 = adata.copy()
    n = adata2.obs['ground_truth'].nunique()
    adata2 = TAST_GATE(adata = adata2, consensus = True, n_clusters = n, spatial_rad_cutoff=75, ground_truth = adata2.obs['ground_truth'])

    NMI = normalized_mutual_info_score(adata2.obs['consensus_mclust'].values,  adata2.obs['ground_truth'].values)
    print('TAST NMI = %.5f' %NMI)
    nmi_list[i] = NMI

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
