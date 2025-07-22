import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics.cluster import normalized_mutual_info_score
import multiprocessing as mp
from MCIST_GraphST import MCIST_GraphST
#from MCIST_GATE import MCIST_GATE
#from MCIST_SpaceFlow import MCIST_SpaceFlow

def res_search_fixed_clus(
    adata,
    fixed_clus_count,
    emb_key: str,         # name of your obsm, e.g. "STAGATE" or "MCIST_emb"
    n_neighbors: int = 20,
    increment: float = 0.02,
):
    """
    Search for a leiden resolution that yields exactly `fixed_clus_count`
    clusters *on the neighbors‐graph built from* adata.obsm[emb_key].
    """
    # 1) build the graph ONCE:
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=emb_key,
        key_added='neighbors_' + emb_key  # e.g. "neighbors_STAGATE"
    )

    # 2) loop through resolutions, always referring back to that same graph:
    graph_key = 'neighbors_' + emb_key
    for res in sorted(np.arange(0.2, 2.5, increment), reverse=True):
        # this will write to adata.obs['leiden']
        sc.tl.leiden(
            adata,
            resolution=res,
            neighbors_key=graph_key,
            key_added='leiden'  # or 'leiden_'+emb_key to keep them all
        )

        n_found = adata.obs['leiden'].nunique()
        if n_found == fixed_clus_count:
            return res

    raise ValueError(f"Couldn’t find a resolution yielding {fixed_clus_count} clusters")

section_id = sys.argv[1]
adata = sc.read('./SpatialTranscriptomics/'+section_id+'.h5ad')

# Validation
n = adata.obs['ground_truth'].nunique()
####################################################################################################
adata2 = adata.copy()
adata2 = MCIST_GraphST(adata = adata2, n_clusters = n, clustering_algo='Mclust')
#res = res_search_fixed_clus(adata2, n, 'GraphST')
res = mclustpy(np.real(adata2.obsm['GraphST']), G=n, modelNames='EEE', random_seed=2020)
mclust_res = res['classification']
adata2.obs['GraphST_mclust'] = mclust_res
obs_df = adata2.obs.dropna()
NMI_MCIST = normalized_mutual_info_score(obs_df['MCIST_spatial_domains'],  obs_df['ground_truth'])
NMI = normalized_mutual_info_score(obs_df['GraphST_mclust'],  obs_df['ground_truth'])
#NMI = normalized_mutual_info_score(obs_df['leiden'],  obs_df['ground_truth'])
print('MCIST NMI = %.5f' %NMI_MCIST)
print('GraphST NMI = %.5f' %NMI)
