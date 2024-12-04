import scanpy as sc
import pandas as pd
from tPCA import tPCA_embedding
import numpy as np
from mclustpy import mclustpy
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed
import STAGATE_pyG as STAGATE
from rs import rs_full, rs_index

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    for res in sorted(list(np.arange(0.2, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res

def preprocess_data(adata):
    if adata.X.shape[1]>3000:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata

def MCIST_Clustering(adata, X, beta, gamma, m, zeta, n_clusters, clustering_algo):
    print('Running for Zetas:', zeta)

    #tPCA
    Q = tPCA_embedding(X, beta, gamma, m, zeta)
    #Feature Concatenation
    Q2 = adata.obsm['STAGATE']
    Q3 = np.concatenate((np.real(Q),Q2), axis = 1)

    if clustering_algo == 'Mclust':
        #Mclust
        res = mclustpy(np.real(Q3), G=n_clusters, modelNames='EEE', random_seed=2020)
        mclust_res = res['classification']
        return mclust_res, np.real(Q)
    if clustering_algo == 'Leiden':
        #Leiden
        adata2 = adata.copy()
        adata2.var_names_make_unique()
        adata2.obsm['MCIST_emb'] = np.real(Q3)
        sc.pp.neighbors(adata2, n_neighbors=20, use_rep='MCIST_emb')
        eval_resolution = res_search_fixed_clus(adata2, n_clusters)
        sc.tl.leiden(adata2, key_added="leiden", resolution=eval_resolution)
        return adata2.obs['leiden'].values, np.real(Q)
    else:
        print("Please specify either the 'Mclust' or 'Leiden' clustering algorithms to proceed.")
        exit(1)

def MCIST_GATE(adata, n_clusters, spatial_rad_cutoff, clustering_algo):
    # pre processing
    adata = preprocess_data(adata)
    # parameters
    beta = 1e1  
    gamma = 1e2
    if adata.X.shape[1]>3000:
        m = 20
    else:
        m = 10
    zeta_combinations = [
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 1],
    [0, 0, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 0, 1]]

    ################### Deep Learning ######################
    # displayed here is MCIST combined with STAGATE
    ## this section can be easily replaced with any arbitrary deep learning method
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=spatial_rad_cutoff) #rad_cutoff will depend on your data
    STAGATE.Stats_Spatial_Net(adata)
    adata = STAGATE.train_STAGATE(adata)

    ####################### MCIST ###########################
    if adata.X.shape[1]>3000:
        adata_highly_variable = adata[:, adata.var['highly_variable']]
        X = adata_highly_variable.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
    else:
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()

    #Topological PCA with different zeta configurations
    cluster_labels_and_embeddings = Parallel(n_jobs=len(zeta_combinations))(delayed(MCIST_Clustering)(adata, X, beta, gamma, m, zeta, n_clusters, clustering_algo) for zeta in zeta_combinations)
    cluster_labels = [result[0] for result in cluster_labels_and_embeddings]

    ######## Spatial Domain Detection via Agglomerative Clustering ########
    #co association matrix
    n_samples = adata.shape[0]
    co_association_matrix = np.zeros((n_samples, n_samples))

    for labels in cluster_labels:
        for i in range(n_samples):
            for j in range(n_samples):
                if labels[i] == labels[j]:
                    co_association_matrix[i, j] += 1

    co_association_matrix /= len(zeta_combinations)

    # Agglomerative (Consensus) Clustering 
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    consensus_labels = agg_clustering.fit_predict(1 - co_association_matrix)
    adata.obs['MCIST_spatial_domains'] = consensus_labels

    ######## Optimal latent embeddings according to RSI optimization ########
    tPCA_embeddings = [result[1] for result in cluster_labels_and_embeddings]
    scores = []
    for labels in cluster_labels:
        ground_truth = np.ones_like(labels) # arbitrary stand in for ground truth labels
        RS = rs_full(X, labels)
        RS_index = rs_index(RS, labels, ground_truth)
        RSI = RS_index[:,2].mean()
        scores.append(RSI) 
    max_rsi_score = max(scores)
    indices = [i for i, score in enumerate(scores) if score == max_rsi_score]
    idx = indices[0]
    # store highest performing TAST embedding for downstream analysis
    adata.obsm['MCIST_emb'] = np.concatenate((tPCA_embeddings[idx], adata.obsm['STAGATE']), axis = 1)

    return adata
