import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

#Topological PCA provides topological features for concatenation

def gaussian_kernel(dist, t):
    '''
    gaussian kernel function for weighted edges
    '''
    return np.exp(-(dist**2 / t))

def Eu_dis(X: np.ndarray) -> np.ndarray:
    return cdist(X, X, "euclidean")        

def cal_weighted_adj(data, n_neighbors, t):
    '''
    Calculate weighted adjacency matrix based on KNN
    For each row of X, put an edge between nodes i and j
    If nodes are among the n_neighbors nearest neighbors of each other
    according to Euclidean distance
    '''
    dist = Eu_dis(data)
    n = dist.shape[0]
    gk_dist = gaussian_kernel(dist, t)
    W_L = np.zeros((n, n))
    for i in range(n):
        index_L = np.argsort(dist[i, :])[1:1 + n_neighbors] 
        len_index_L = len(index_L)
        for j in range(len_index_L):
            W_L[i, index_L[j]] = gk_dist[i, index_L[j]] #weighted edges
    W_L = np.maximum(W_L, W_L.T)
    return W_L

def cal_unweighted_adj(data, n_neighbors):
    '''
    Calculate unweighted adjacency matrix based on KNN
    '''
    dist = Eu_dis(data)
    n = dist.shape[0]
    W_L = np.zeros((n, n))
    for i in range(n):
        index_L = np.argsort(dist[i, :])[1:1 + n_neighbors]
        len_index_L = len(index_L)
        for j in range(len_index_L):
            W_L[i, index_L[j]] = 1 #edges not weighted
    W_L = np.maximum(W_L, W_L.T)
    return W_L

def cal_laplace(adj):
    N = adj.shape[0]
    D = np.zeros_like(adj)
    for i in range(N):
        D[i, i] = np.sum(adj[i]) # Degree Matrix
    L = D - adj  # Laplacian
    return L


def tPCA_Algorithm(xMat,laplace,beta,gamma,k,n):
    '''
    Optimization Algorithm of tPCA 
    Solve approximately via ADMM
    Need to compute optimal principal directions matrix U
    Projected Data matrix Q
    Error term matrix E = X - UQ^T
    Z matrix used to solve Q (see supplementary information)

    Inputs are data matrix X, laplacian, scale parameters, 
    number of reduced dimensions, number of original dimensions
    '''
    # Initialize thresholds, matrices
    obj1 = 0
    obj2 = 0
    thresh = 1e-6
    V = np.eye(n) 
    vMat = np.asarray(V) # Auxillary matrix to optimize L2,1 norm
    E = np.ones((xMat.shape[0],xMat.shape[1]))
    E = np.asarray(E) # Error term X - UQ^T
    C = np.ones((xMat.shape[0],xMat.shape[1]))
    C = np.asarray(C) # Lagrangian Multiplier
    laplace = np.asarray(laplace) #Lplacian
    miu = 1 #Penalty Term
    for m in range(0, 35):
        Z = (-(miu/2) * ((E - xMat + C/miu).T @ (E - xMat + C/miu))) + beta * vMat + gamma * laplace
        # cal Q (Projected Data Matrix)
        Z_eigVals, Z_eigVects = np.linalg.eigh(np.asarray(Z))
        eigValIndice = np.argsort(Z_eigVals)
        n_eigValIndice = eigValIndice[0:k]
        n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
        # Optimal Q given by eigenvectors corresponding
        # to smallest k eigenvectors
        Q = np.array(n_Z_eigVect)  
        # cal V 
        q = np.linalg.norm(Q, ord=2, axis=1)
        qq = 1.0 / (q * 2)
        VV = np.diag(qq)
        vMat = np.asarray(VV)
        qMat = np.asarray(Q)
        # cal U (Principal Directions)
        U = (xMat - E - C/miu) @ qMat
        # cal P (intermediate step)
        P = xMat - U @ qMat.T - C/miu
        # cal E (Error Term)
        for i in range(E.shape[1]):
            E[:,i] = (np.max((1 - 1.0 / (miu * np.linalg.norm(P[:,i]))),0)) * P[:,i]
        # update C 
        C = C + miu * (E - xMat + U @ qMat.T)
        # update miu
        miu = 1.2 * miu

        obj1 = np.linalg.norm(qMat)
        if m > 0:
            diff = obj2 - obj1
            if diff < thresh:
                break # end iterations if error within accepted threshold
        obj2 = obj1
    return U #return principal directions matrix

def cal_persistence_KNN(data, n_filtrations, zetas):
    n = data.shape[0]
    '''
    Consider n neighbors and reduce by 2 neighbors at
    each iteration of filtration down to 6 nearest neighbors
    (4 filtrations)
    '''
    num_neighbors_list = range(6, n_filtrations + 1, 3)
    num_filtrations = len(num_neighbors_list)

    PL = np.zeros((num_filtrations, n, n))
    zetas = np.array(zetas)

    for idx, num_neighbors in enumerate(num_neighbors_list):
        A = cal_unweighted_adj(data, num_neighbors)
        #print('num neighbors:', num_neighbors)
        PL[idx, :, :] = cal_laplace(A)
        #print("i'th PL:", PL[idx, :, :])
        #print("zeta_i:", zetas[idx])

    Persistent_Laplacian = np.sum(zetas[:, np.newaxis, np.newaxis] * PL, axis=0)
    return Persistent_Laplacian

def tPCA_cal_projections_KNN(X_data, beta1, gamma1, k_d, num_filtrations, zeta):
    n = len(X_data)  
    M = cal_persistence_KNN(X_data, num_filtrations, zeta)
    Y = tPCA_Algorithm(X_data.transpose(), M, beta1, gamma1, k_d, n)
    return Y

#Embed gene expression data
def tPCA_embedding(X, beta, gamma, k, zeta):
    #Principal Components
    PDM = tPCA_cal_projections_KNN(X, beta, gamma, k, 15, zeta)
    PDM = np.asarray(PDM)
    #print(PDM.shape)
    TM = (np.linalg.solve(PDM.T @ PDM, PDM.T)).T
    #Projected Data Matrix
    Q = (X @ TM)
    #print(Q.shape)
    return Q
