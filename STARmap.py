import warnings
warnings.filterwarnings('ignore')
import numpy as np
import SpaceFlow
import scanpy as sc
import pandas as pd
import os
import sys
from mclustpy import mclustpy
from tPCA import tPCA_embedding
from SpaceFlow import SpaceFlow 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import normalized_mutual_info_score
le = LabelEncoder()

#For STARmap, we combine topological features with the SpaceFlow embedding
section_id = sys.argv[1]
adata = sc.read('/mnt/home/cottre61/TAST/Data/STARmap/'+section_id+'.h5ad')

#Extract tPCA parameters
rootPath = '/mnt/home/cottre61/TAST/' 
df = pd.read_csv(rootPath+'params.csv')
section_id = sys.argv[1]
dataset_row = df[df['Dataset'] == 'STARmap'].iloc[0]

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
    sf = SpaceFlow.SpaceFlow(adata2)
    sf.preprocessing_data(n_top_genes=3000)
    sf.train(spatial_regularization_strength=0.1, 
         z_dim=50, 
         lr=1e-3, 
         epochs=1000, 
         max_patience=50, 
         min_stop=100, 
         random_seed=42, 
         gpu=1, 
         regularization_acceleration=True, 
         edge_subset_sz=1000000)

    #Perform tPCA
    adata_highly_variable = adata2[:, adata2.var['highly_variable']]
    X = adata_highly_variable.X
    X = X.toarray()

    #Embedding
    Q = tPCA_embedding(X, beta, gamma, k, zeta)

    #Feature Concatenation
    Q2 = sf.embedding
    print(Q2.shape)
    Q3 = np.concatenate((Q,Q2), axis = 1)
    print(Q3.shape)
    adata2.obsm['TAST'] = Q3

    #McClust Clustering
    res = mclustpy(np.real(Q3), G=n_clusters, modelNames='EEE', random_seed=2020)
    mclust_res = res['classification']
    adata2.obs['mclust'] = mclust_res
    res2 = mclustpy(np.real(Q2), G=n_clusters, modelNames='EEE', random_seed=2020)
    mclust_res2 = res2['classification']
    adata2.obs['mclust2'] = mclust_res2

    ground_truth = le.fit_transform(pd.Series(adata2.obs['label'].values))
    NMI = normalized_mutual_info_score(mclust_res, ground_truth)
    print('TAST NMI = %.5f' %NMI)
    nmi_list[i] = NMI
    if i == 1:
        plt.rcParams["figure.figsize"] = (8, 8)
        sc.pl.spatial(adata2, color='mclust', cmap = 'tab20', save=section_id+'tast.png', spot_size=150)
        sc.pl.spatial(adata2, color='mclust2', cmap = 'tab20', save=section_id+'spaceflow.png', spot_size=150)

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
