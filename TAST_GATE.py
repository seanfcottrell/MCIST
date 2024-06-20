import scanpy as sc
from tPCA import tPCA_embedding
import numpy as np
from mclustpy import mclustpy
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed
import STAGATE_pyG as STAGATE
from sklearn.metrics.cluster import normalized_mutual_info_score
from rs import rs_full, rs_index

def preprocess_data(adata):
    if adata.X.shape[1]>3000:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata

def TAST_Parallel_Clustering(adata, X, beta, gamma, m, zeta, n_clusters):
    print('Running for Zetas:', zeta)

    #tPCA
    Q = tPCA_embedding(X, beta, gamma, m, zeta)
    #Feature Concatenation
    Q2 = adata.obsm['STAGATE']
    Q3 = np.concatenate((Q,Q2), axis = 1)

    #Mclust
    res = mclustpy(np.real(Q3), G=n_clusters, modelNames='EEE', random_seed=2020)
    mclust_res = res['classification']
    return mclust_res, Q3

def TAST_GATE(adata, consensus, n_clusters, spatial_rad_cutoff, ground_truth):
    # pre processing
    adata = preprocess_data(adata)
    # parameters
    #betas = [1e1,1e2,1e3] #can add more parameter choices if you wish to optimize these scalars
    #gammas = [1e1,1e2,1e3]
    #ms = [10,15,20]
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
    # displayed here is TAST combined with STAGATE
    ## this section can be easily replaced with any arbitrary deep learning method
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=spatial_rad_cutoff) #rad_cutoff will depend on your data
    STAGATE.Stats_Spatial_Net(adata)
    adata = STAGATE.train_STAGATE(adata)

    ####################### TAST ###########################
    if adata.X.shape[1]>3000:
        adata_highly_variable = adata[:, adata.var['highly_variable']]
        X = adata_highly_variable.X
        X = X.toarray()
    else:
        X = adata.X
        X = X.toarray()
    if consensus == True:
        #Topological PCA with different zeta configurations
        cluster_labels_and_embeddings = Parallel(n_jobs=len(zeta_combinations))(delayed(TAST_Parallel_Clustering)(adata, X, beta, gamma, m, zeta, n_clusters) for zeta in zeta_combinations)
        cluster_labels = [result[0] for result in cluster_labels_and_embeddings]

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

        adata.obs['consensus_mclust'] = consensus_labels
    
    else:
        # multiple tPCA clusterings with 8 different zeta combinations
        cluster_labels_and_embeddings = Parallel(n_jobs=len(zeta_combinations))(delayed(TAST_Parallel_Clustering)(adata, X, beta, gamma, m, zeta, n_clusters) for zeta in zeta_combinations)
        cluster_labels = [result[0] for result in cluster_labels_and_embeddings]
        embeddings = [result[1] for result in cluster_labels_and_embeddings]

        # examine performance metrics for each combination 
        ## and extract highest performing parameter combination according to NMI (if ground truth available) or RSI (if not)
        scores = []
        if ground_truth is None:
            for labels in cluster_labels:
                ground_truth = np.ones_like(labels) # stand in for ground truth labels
                RS = rs_full(X, labels)
                RS_index = rs_index(RS, labels, ground_truth)
                RSI = RS_index[:,2].mean()
                scores.append(RSI) 
            max_rsi_score = max(scores)
            indices = [i for i, score in enumerate(scores) if score == max_rsi_score]
            idx = indices[0]
        else:
            for labels in cluster_labels:
                NMI = normalized_mutual_info_score(labels, ground_truth)
                scores.append(NMI) 
            max_nmi_score = max(scores)
            indices = [i for i, score in enumerate(scores) if score == max_nmi_score]
            idx = indices[0]
        
        # store highest performing TAST embedding for downstream analysis
        adata.obsm['TAST'] = embeddings[idx]
        # store highest performing spatial domain labels for future use
        adata.obs['mclust'] = cluster_labels[idx]
    return adata
    
