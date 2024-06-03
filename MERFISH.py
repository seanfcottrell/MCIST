import STAGATE_pyG as STAGATE
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from tPCA import tPCA_embedding
from mclustpy import mclustpy
from sklearn.metrics.cluster import normalized_mutual_info_score

#For MERFISH, we combine topological features with the STAGATE embedding
#Load Data
section_id = sys.argv[1]
if section_id =='MERFISH_Data24':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/MERFISH/Data24/MERFISH_0.24_20240319045320.h5ad')
if section_id =='MERFISH_Data19':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/MERFISH/Data19/MERFISH_0.19_20240319045434.h5ad')
if section_id =='MERFISH_Data14':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/MERFISH/Data14/MERFISH_0.14_20240319045431.h5ad')
if section_id =='MERFISH_Data9':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/MERFISH/Data9/MERFISH_0.09_20240319045428.h5ad')
if section_id =='MERFISH_Data4':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/MERFISH/Data4/MERFISH_0.04_20240319045554.h5ad')

#Normalization
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

#Extract tPCA parameters
rootPath = '/mnt/home/cottre61/TAST/' 
df = pd.read_csv(rootPath+'params.csv')
section_id = sys.argv[1]
dataset_row = df[df['Dataset'] == section_id].iloc[0]

zeta = [float(dataset_row['zeta1']), float(dataset_row['zeta2']), float(dataset_row['zeta3']),
        float(dataset_row['zeta4']), float(dataset_row['zeta5']), float(dataset_row['zeta6']),
        float(dataset_row['zeta7']), float(dataset_row['zeta8'])]
zeta = np.asarray(zeta)
print('zeta:',zeta)
k = int(dataset_row['k'])
print('k:', k)
beta = float(dataset_row['beta'])
print('beta:', beta)
gamma = float(dataset_row['gamma'])
print('gamma:', gamma)
n_clusters=int(dataset_row['clusters'])

#Validation
nmi_list=[1,2,3,4,5,6,7,8,9,10]

for i in range(10):
    adata2 = adata.copy()
    STAGATE.Cal_Spatial_Net(adata2, rad_cutoff=75)
    STAGATE.Stats_Spatial_Net(adata2)
    adata = STAGATE.train_STAGATE(adata2)

    #Perform tPCA
    X = adata2.X
    X = X.toarray()

    #Embedding
    Q = tPCA_embedding(X, beta, gamma, k, zeta)

    #Feature Concatenation
    Q2 = adata2.obsm['STAGATE']
    print(Q2.shape)
    Q3 = np.concatenate((Q,Q2), axis = 1)
    print(Q3.shape)
    adata2.obsm['TAST'] = Q3

    #McClust Clustering
    res = mclustpy(np.real(Q3), G=n_clusters, modelNames='EEE', random_seed=2020)
    mclust_res = res['classification']
    adata2.obs['mclust'] = mclust_res

    NMI = normalized_mutual_info_score(mclust_res,  adata.obs['ground_truth'].values)
    print('TAST NMI = %.5f' %NMI)
    nmi_list[i] = NMI

mean = np.mean(nmi_list)
print('Average TAST NMI:', mean)

#Write results to CSV
results = {
        'Dataset': [section_id],
        'TAST NMI': [mean]
    }
results_df = pd.DataFrame(results)
file_path = 'collected_results.csv'
file_exists = os.path.exists(file_path)
results_df.to_csv(file_path, mode='a', index=False, header=not file_exists)
