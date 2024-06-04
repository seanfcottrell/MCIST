import os
import sys
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scanpy as sc
import STAGATE_pyG as STAGATE
from tPCA import tPCA_embedding
from mclustpy import mclustpy
from sklearn.metrics.cluster import normalized_mutual_info_score

#For BaristaSeq, we combine topological features with the STAGATE embedding
#Load Data
section_id = sys.argv[1]
adata = sc.read('/mnt/home/cottre61/TAST/Data/BaristaSeq/'+section_id+'.h5ad')

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
    STAGATE.Cal_Spatial_Net(adata2, rad_cutoff=50)
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

    NMI = normalized_mutual_info_score(mclust_res,  adata.obs['layer'].values)
    print('TAST NMI = %.5f' %NMI)
    nmi_list[i] = NMI

    if section_id == 'slice_2' and i == 1:
        plt.rcParams["figure.figsize"] = (8, 8)
        sc.pl.spatial(adata, color='mclust', cmap = 'tab20', save='baristaseq'+section_id+'.png', spot_size=20)

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
