import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.metrics.cluster import normalized_mutual_info_score
from MCIST_GATE import MCIST_GATE

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

#Load Data
section_id = sys.argv[1]
adata = sc.read('./SpatialTranscriptomics/'+section_id+'.h5ad')

#Validation
n = adata.obs['layer'].nunique()
####################################################################################################
adata4 = adata.copy()
adata4 = MCIST_GATE(adata = adata4, n_clusters = n, spatial_rad_cutoff = 150, clustering_algo='Mclust')
res = mclustpy(np.real(adata4.obsm['STAGATE']), G=n, modelNames='EEE', random_seed=2020)
mclust_res = res['classification']
adata4.obs['STAGATE_mclust'] = mclust_res
#res = res_search_fixed_clus(adata4, n, 'STAGATE')
obs_df = adata4.obs.dropna()
NMI_Mclust_GATE = normalized_mutual_info_score(obs_df['MCIST_spatial_domains'],  obs_df['layer'])
#NMI = normalized_mutual_info_score(obs_df['leiden'],  obs_df['layer'])
NMI = normalized_mutual_info_score(obs_df['STAGATE_mclust'],  obs_df['layer'])
print('MCIST NMI = %.5f' %NMI_Mclust_GATE)
print('STAGATE NMI = %.5f' %NMI)
