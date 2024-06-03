import torch
import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics.cluster import normalized_mutual_info_score
import multiprocessing as mp
from mclustpy import mclustpy
from GraphST import GraphST
from tPCA import tPCA_embedding
from sklearn.decomposition import PCA

#For StereoSeq, we combine topological features with the GraphST embedding
# Run device，by default, the package is implemented on 'cpu'. We recommend using GPU.
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

section_id = sys.argv[1]

if section_id=='Stereo-Seq_E95_E1S1':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/Stereo-Seq/E9.5/E9.5_E1S1.MOSTA_20240319045807.h5ad')
if section_id=='Stereo-Seq_E95_E2S1':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/Stereo-Seq/E9.5/E9.5_E2S1.MOSTA_20240319045818.h5ad')
if section_id=='Stereo-Seq_E95_E2S2':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/Stereo-Seq/E9.5/E9.5_E2S2.MOSTA_20240319045821.h5ad')
if section_id=='Stereo-Seq_E95_E2S3':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/Stereo-Seq/E9.5/E9.5_E2S3.MOSTA_20240319045823.h5ad')
if section_id=='Stereo-Seq_E95_E2S4':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/Stereo-Seq/E9.5/E9.5_E2S4.MOSTA_20240319045824.h5ad')
if section_id=='Stereo-Seq_E10_E1S1':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/Stereo-Seq/E10.5/E10.5_E1S1.MOSTA_20240319045825.h5ad')
if section_id=='Stereo-Seq_E10_E1S2':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/Stereo-Seq/E10.5/E10.5_E1S2.MOSTA_20240319102119.h5ad')
if section_id=='Stereo-Seq_E10_E1S3':
    adata = sc.read('/mnt/home/cottre61/TAST/Data/Stereo-Seq/E10.5/E10.5_E1S3.MOSTA_20240319102125.h5ad')

adata.var_names_make_unique()

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
    model = GraphST.GraphST(adata2, datatype='Stereo', device=device)
    # run model
    adata2 = model.train()

    #Normalization
    sc.pp.highly_variable_genes(adata2, flavor="seurat_v3", n_top_genes=3000)

    #Perform tPCA
    adata_highly_variable = adata2[:, adata2.var['highly_variable']]
    X = adata_highly_variable.X
    X = X.toarray()

    #Embedding
    Q = tPCA_embedding(X, beta, gamma, k, zeta)

    #Feature Concatenation
    pca = PCA(n_components=20, random_state=42) 
    embedding = pca.fit_transform(adata2.obsm['emb'].copy())
    print(embedding.shape)
    Q3 = np.concatenate((Q,embedding), axis = 1)

    #McClust Clustering
    res = mclustpy(np.real(Q3), G=n_clusters, modelNames='EEE', random_seed=2020)
    mclust_res = res['classification']
    adata2.obs['mclust'] = mclust_res

    NMI = normalized_mutual_info_score(mclust_res, adata.obs['ground_truth'].values)
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

