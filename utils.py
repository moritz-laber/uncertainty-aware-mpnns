"""
utils.py
----------

Utility functions for data generation, training, and uncertainty propagation in MPNNs. 
"""

import equinox as eqx
import jax
import jax.experimental.sparse as sparse
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.typing import ArrayLike
import numpy as np
import optax
import ot
import scipy as sp
import tqdm
from typing import List, Tuple, Callable, Optional

from models import normalized_adjacency, softplus


### LIPSCHITZ ###
def gcn_lipschitz(gcn : eqx.Module, A:sparse.BCOO) -> float:
    """Estimate an upper bound on the Lipschitz constant (in 2-norm) of a GCN model.
    
    Input
    gcn : GCN model as equinox module
    A : adjacency matrix of the graph in sparse.BCOO format.
    
    Output
    C : estimated upper bound on the Lipschitz constant of the GCN model.
    """

    # get the number of layers
    L = len(gcn.layers)

    # compute the contribution of the nonlinearities to the Lipschitz constant
    # see e.g., Table 2 of Qi et al. (arXiv 2023) https://arxiv.org/abs/2306.09338
    if gcn.non_linearity == jax.nn.relu:
        C = 1.0
    elif gcn.non_linearity == jax.nn.gelu:
        C = 1.1
    elif gcn.non_linearity == jax.nn.tanh:
        C = 1.0
    elif gcn.non_linearity == jax.nn.sigmoid:
        C = 0.25
    else:
        raise NotImplementedError("Lipschitz constant for the chosen nonlinearity is not implemented.")
    
    # these nonlinearities are used at L-1 layers
    C = C ** (L - 1)

    # compute normalized adjacency
    S = normalized_adjacency(A)

    # compute propagation contribution to the Lipschitz constant
    sigmaS = jnp.linalg.svd(S.todense(), compute_uv=False)[0]
    C *= sigmaS ** L

    # compute the weight matrix contribution to the Lipschitz constant
    for i in range(L):
        Wi = gcn.layers[i].weights
        sigmaW = jnp.linalg.svd(Wi, compute_uv=False)[0]
        C *= sigmaW 
    
    return C


### SYNTHETIC DATA GENERATION ###
def construct_precision(A:jnp.array, Lam_edge:jnp.array, Lam_node:jnp.array)-> jnp.array:
    """Construct the precision matrix for a Gaussian Markov Random Field (GMRF).
    
    Input
    A : adjacency matrix of the graph (num_nodes x num_nodes)
    Lam_edge : edge coupling matrix (num_features x num_features)
    Lam_node : node precision matrix (num_features x num_features)

    Output
    Lam : precision matri of the GMRF (num_nodes*num_features x num_nodes*num_features)
    """

    # construct the graph laplacian
    D = jnp.diag(jnp.sum(A, axis=1).flatten())
    L = D - A

    # construct the precision matrix I_n x Lam_node + L x Lam_edge
    Lam  = jnp.kron(jnp.eye(L.shape[0]), Lam_node) + jnp.kron(L, Lam_edge)

    return Lam

def sample_gmrf(key:jax.random.key, mu:jnp.array, Lam:jnp.array, num_samples:int)-> jnp.array:
    """Sample from a Gaussian Markov Random Field (GMRF) with given precision matrix.
    
    Input
    key : jax random key
    mu : mean vector (num_nodes * num_features, )
    Lam : precision matrix (num_nodes * num_features, num_nodes * num_features)
    num_samples : number of samples to draw

    Output
    X : samples from the GMRF (num_nodes * num_features, num_samples)
    """

    # determine the product of node and feature dimension
    d = Lam.shape[0]

    # compute the Cholesky decomposition of the precision matrix
    L = jsp.linalg.cholesky(Lam, lower=True)

    Z = jax.random.normal(key, shape=(d, num_samples))
    X = jsp.linalg.solve_triangular(L.T, Z, lower=False)
    X += mu[:, None]                                        # adjust mean

    return X

def covariance_to_correlation(SigX: jnp.ndarray) -> jnp.ndarray:
    """Convert a covariance matrix to a correlation matrix.
    
    Input
    SigX : covariance matrix (n, n)

    Output
    RhoX : correlation matrix (n, n)
    """

    sig = jnp.sqrt(jnp.diag(SigX))
    RhoX = SigX / (sig[:, None] * sig[None, :])

    return RhoX

def sbm(rng:np.random.Generator, ns:List[int], k_ii:int, k_ij:int) -> Tuple[sp.sparse.lil_matrix, List]:
    """"Sample a graphs from th stochastic block model with b groups,
        and constant within group and between group average degree.

        Input
        rng : a numpy random number generator
        ns : list of length b, where ns[i] is the number of nodes in group i
        k_ii : average within group degree
        k_ij : average between group degree

        Output
        A : adjacency matrix of the sampled graph as (sparse lil_matrix)
        g : a list encoding the group assignment of each node     
     """

    # determine the total number of nodes
    ns = np.asarray(ns)
    n = np.sum(ns)

    # assign groups
    g = [i for i, ni in enumerate(ns) for _ in range(ni)]

    # construct the block probability matrix
    p_ii = np.asarray([k_ii/ni for ni in ns])
    p = k_ij / np.sqrt(np.outer(ns, ns))
    p[np.diag_indices_from(p)] = p_ii

    # sample edges
    A = sp.sparse.lil_matrix((n, n))
    r = rng.random(size=int(n * (n - 1.0) // 2))
    edge_idx = np.triu_indices(n, k=1)
    for idx, (i, j) in enumerate(zip(*edge_idx)):

        if r[idx] < p[g[i], g[j]]:
            A[i, j] = 1
            A[j, i] = 1

    return A, g

### REAL-WORLD DATA LOADING ###
def edgelist_to_adjacency(edge_list:np.ndarray, num_nodes:int)->sparse.BCOO:
    """"Convert edgelist to sparse adjacency matrix.
    
    Input
    edge_list : edge list as numpy array of shape (2, num_edges)
    num_nodes : number of nodes in the graph

    Output
    A_sparse : sparse adjacency matrix as sparse.BCOO
    """

    # create scipy.sparse coo matrix
    A_sparse = sp.sparse.coo_matrix(
        (np.ones(edge_list.shape[1]),
        (edge_list[0, :], edge_list[1, :])),
        shape=(num_nodes, num_nodes),
        )

    # convert to jax.experimental.sparse.BCOO
    A_sparse = sparse.BCOO.from_scipy_sparse(A_sparse)

    return A_sparse

def load_dataset(path_to_file:str, num_features:int=None, train_frac:float=None, which_split:int=0, key_split:jax.random.PRNGKey=None)->Tuple[sparse.BCOO, jnp.array, jnp.array, jnp.array, jnp.array]:
    """Load Dataset from npz file.
    
    Input
    path_to_file : path to the npz file containing the data.
    num_features : number of features to use. If None all features are used, if integer SVD is used for dimensionality reduction.
    train_frac : fraction of nodes used for training. If None predefined splits are used, if float random splits are generated using split key.
    which_split : if predefined splits are used, which split to use.
    key_split : jax.random.PRNGKey used for random train/test split generation.

    Output
    A_sparse : sparse adjacency matrix as sparse.BCOO
    X : node features as jnp.array of shape (num_nodes * num_features,)
    y : one-hot encoded labels as jnp.array of shape (num_nodes, num_classes)
    train_idx : indices of training nodes as jnp.array
    test_idx : indices of test nodes as jnp.array
    """

    # load data from npz file
    data = np.load(path_to_file)
    
    # extract features
    X = jnp.asarray(data['X'])
    num_nodes = X.shape[0]
    
    if num_features is None:
        # raw features
        num_features = X.shape[1]
    else:
        # SVD features
        Xc = X - jnp.mean(X, axis=0, keepdims=True)
        _, _, VT = jnp.linalg.svd(Xc, full_matrices=False)
        X = Xc @ VT.T[:, :num_features]
    
    # reshape into the high-dimensional feature vectors
    X = X.reshape((num_nodes * num_features,))
    
    # labels (one-hot encoded)
    y = jnp.asarray(data['y'])
    y = jax.nn.one_hot(y, num_classes=jnp.max(y) + 1)

    # train test split
    if train_frac is None:

        # attempt to use predefined splits
        try:
            train_idx_bool = jnp.asarray(data['train_split'])
            test_idx_bool = jnp.asarray(data['test_split'])

            if len(train_idx_bool)>1:
                train_idx_bool = train_idx_bool[:, which_split]
                test_idx_bool = test_idx_bool[:, which_split]
                
        except:
            raise RuntimeError("Dataset does not define train/test splits/")
    else:
        # sample train test split as boolean masks
        train_idx_bool = jax.random.bernoulli(key_split, train_frac, shape=(num_nodes,)).astype(bool)
        test_idx_bool = ~train_idx_bool

    # convert from boolean mask to indices
    train_idx = jnp.where(train_idx_bool)[0]
    test_idx =  jnp.where(~train_idx_bool)[0]

    # create adjacency matrix from edge list
    A_sparse = edgelist_to_adjacency(data['edge_list'], num_nodes=num_nodes)

    return A_sparse, X, y, train_idx, test_idx

### UNCERTAINTY PROPAGATION ###
def relative_error(a:jnp.array, b:jnp.array, p:float=1)-> jnp.array:
    """Compute the relative error between two errors in p-norm.
    
    Input
    a : first array
    b : second array
    p : p-norm to use

    Output
    error : ||a - b||_p / ||b||_b relative error between a and b in p-norm
    """

    if p==1:
        error = jnp.mean(jnp.abs(a - b) / (jnp.abs(b) + 1e-10))
    else:
        error = jnp.power(jnp.mean(jnp.power(a - b, p)), 1.0/p) / jnp.power((jnp.mean(jnp.power(b, p)) + 1e-10), 1.0/p)

    return error


def ensure_psd(A:np.ndarray, eps:float=1e-10)->np.ndarray:
    """Makes A positive semi-definite and symmetric.
    
    Input
    A : input matrix (n, n)
    eps : lower cutoff for eigenvalues of A
    
    Output
    A : positive semi-definite version of A (n, n)
    """

    A = 0.5 * (A + A.T)

    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)

    A = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return A

def matrix_sqrt(A:np.ndarray, eps:float=1e-10)->np.ndarray:
    """Compute the matrix square root of a positive semi-definite matrix A.
    
    Input
    A : input matrix (n, n)
    eps : lower cutoff for eigenvalues of A (for numerical stability)

    Output
    Asqrt : matrix square root of A (n, n)
    """

    A = 0.5 * (A + A.T)

    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)

    Asqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    return Asqrt

def wasserstein_distance(pars1:Tuple[ArrayLike, ArrayLike], pars2:Tuple[ArrayLike, ArrayLike], eps:float=1e-10, diag:bool=False)->float:
    """Compute the Wasserstein 2 distance between Gaussians two Gaussian distributions.
    
    Input
    pars1 : parameters of the first Gaussian (mu1, Sig1)
    pars2 : parameters of the second Gaussian (mu2, Sig2)
    eps : cutoff for eigenvalues used for numerical stability

    Output
    dist : Wasserstein 2 distance between the two Gaussians
    """

    # extract parameters
    mu1, Sig1 = pars1
    mu2, Sig2 = pars2

    # ensure arrays
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    Sig1 = np.asarray(Sig1, dtype=np.float64)
    Sig2 = np.asarray(Sig2, dtype=np.float64)


    if diag:
        
        # assume diagonal covariance
        sig1 = np.diag(Sig1)
        sig2 = np.diag(Sig2)
        nonneg = (sig1 >= 0) & (sig2 >= 0) # safety
        dist = np.sqrt(np.sum((mu1 - mu2)**2) + np.sum((np.sqrt(sig1[nonneg]) - np.sqrt(sig2[nonneg]))**2))

    else:
        
        # ensure matrices are positive definite
        Sig1 = ensure_psd(Sig1, eps=eps)
        Sig2 = ensure_psd(Sig2, eps=eps)
    
        # matrix square root
        Sig2_sqrt = matrix_sqrt(Sig2, eps=eps)
    
        # Inner square root
        SigSigSig = Sig2_sqrt @ Sig1 @ Sig2_sqrt
        SigSigSig = ensure_psd(SigSigSig, eps=eps)
        SigSigSig_sqrt = matrix_sqrt(SigSigSig, eps=eps)

        # Compute squared Wasserstein distance
        dist = np.sqrt(np.sum((mu1 - mu2) ** 2) + np.trace(Sig1 + Sig2 - 2 * SigSigSig_sqrt))
    
    return dist

def estimate_moments(X:jnp.array, second_raw:bool=False)-> Tuple[jnp.array, jnp.array]:
    """Estimate mean and covariance from samples.
    
    Input
    X : samples of shape (num_nodes * num_features, num_samples)
    second_raw : whether to compute raw second moments (True) or central moments (False)

    Output
    muX : mean of X (num_nodes * num_features,)
    SigX : covariance (or raw second moment if second_raw=True) of X (num_nodes * num_features, num_nodes * num_features)
    """

    n_samples = X.shape[1]

    muX = X.mean(axis=1)

    if second_raw:
        SigX = (X @ X.T) / n_samples
    else:
        SigX = ((X - muX[:, None]) @ (X - muX[:, None]).T) / n_samples

    return muX, SigX

def sample_propagation(f:Callable, X:jnp.array)->Tuple[jnp.array, jnp.array]:
    """"Propagate samples throug a function f and estimate the moments.
    
    Input
    f : function to propagate samples through f : R^n -> R^m
    X : samples of shape (n, n_samples)

    Output
    muY : mean of f(X) (m,)
    SigY : covariance of f(X) (m, m)
    """

    # propagate samples
    Y = jax.vmap(f, in_axes=1, out_axes=1)(X)

    # estimate empirical mean and covariance
    muY, SigY = estimate_moments(Y)

    return muY, SigY

def sample_propagation_batched(key:jax.random.PRNGKey, f:Callable, mu:jnp.array, sigma:float, num_samples:int, batch_size:int, verbose:bool=False) -> Tuple[jnp.array, jnp.array]:
    """Generates random Gaussian noise with standard deviation sigma around the mean mu,
       and propagates samples through the function f using a batched online estimator for
       mean and covariance of the output.

       Input
       key : jax random key
       f : function to propagate samples through f : R^n -> R^m
       muX : mean vector of shape (n,)
       sigma : standard deviation of the Gaussian noise (scalar)
       num_samples : number of samples to draw (integer)
       batch_size : size of batches to use for propagation (integer)
       verbose : whether to print progress information (boolean)
       
       Output
       muY : mean of f(X) (m,)
       SigY : covariance of f(X) (m, m)
    """

    assert num_samples % batch_size == 0, "num_samples must be divisible by batch_size"

    n_batches = num_samples // batch_size
    for i in range(0, n_batches):

        if verbose: print(f'Processing batch {i+1}/{n_batches}', end='\r')

        # generate noisy batch
        if type(sigma)==float:
            key, key_ = jax.random.split(key)
            z = jax.random.normal(key_, shape=(mu.shape[0], batch_size))
            X_batch = mu[:, None] + sigma * z
        elif sigma.shape==mu.shape:
            keys = jax.random.split(key, batch_size + 1)
            key = keys[0]
            z = jax.vmap(lambda k: jax.random.normal(k, shape=(mu.shape[0],)), out_axes=1)(keys[1:])
            X_batch = mu[:, None] + sigma[:, None] * z
        else:
            raise TypeError("sigma must be either a float or a jnp.ndarray of the same shape as mu")

        # propagate batch
        Y_batch = jax.vmap(f, in_axes=1, out_axes=1)(X_batch)

        # compute batch moments
        muY_batch, sY_batch = estimate_moments(Y_batch, second_raw=False)
        sY_batch *= batch_size

        # update running statistics
        if i == 0:
            muY = muY_batch
            sY = sY_batch
            delta = 0.
        else:
            delta = muY_batch - muY
            muY += delta * (batch_size / ((i + 1) * batch_size))
            sY += sY_batch + jnp.outer(delta, delta) * ((i * batch_size) * batch_size)/((i + 1) * batch_size)
    
    SigY = sY / num_samples

    return muY, SigY

def linear_propagation(f:Callable, muX:jnp.array, SigX:jnp.array, eps:float=0.0)->Tuple[jnp.array, jnp.array]:
    """Propagate the mean and covariance of a random variable through the linearization of f
    
    Input
    f : function f : R^n -> R^m
    muX : mean of the input random variable X in R^n
    SigX : covariance of the input random variable X in R^{n x n}
    eps : small regularizer added to SigX for numerical stability

    Output
    muY : propagated mean in R^m under linearized f
    SigY : propagated covariance in R^{m x m} under linearized f 
    """

    # PREPARATION #
    # Cholesky factorization of the covariance
    # optional: add a small regularizer to avoid numerically negative eigenvalues
    L = jsp.linalg.cholesky(SigX + eps * jnp.eye(SigX.shape[0]), lower=True)

    ## MEAN ##
    # the order 0 is non-zero, order 1 vanishes for 
    # Taylor around the mean.
    muY = f(muX)
 
    ## COVARIANCE ##
    # use that J Sigma J^T = (J L)(J L)^T where LL^T = Sigma
    # you in_axis=1 is because we need to multiply each column of L
    # and out_axis=1 because we need to stack the output as columns.
    JL = jax.vmap(
        lambda v : jax.jvp(f, (muX,), (v,))[1],
        in_axes=1,
        out_axes=1
        )(L)
    
    # E[(Y-muY)(Y-muY)^T] = E[YY^T] - muY muY^T
    # note that in first order approximation the propagated mean and the 
    # zeroth order term cancel out.
    SigY = jnp.einsum('aj, jb -> ab', JL, JL.T)

    return muY, SigY

def quadratic_propagation(f:Callable, muX:jnp.array, SigX:jnp.array, eps:float=0.0)->Tuple[jnp.array, jnp.array]:
    """Propagate the mean and covariance of a random variable through the second order approximation of a function f.
    
    Input
    f : function f : R^n -> R^m
    muX : mean of the input random variable X in R^n
    SigX : covariance of the input random variable X in R^{n x n}
    eps : small regularizer added to SigX for numerical stability

    Output
    muY : propagated mean in R^m under second order approximation of f
    SigY : propagated covariance in R^{m x m} under second order approximation of f 
    """

    # PREPARAPTION # 
    # Cholesky factorization of the covariance
    L = jsp.linalg.cholesky(SigX + eps * np.eye(SigX.shape[0]), lower=True)

    # f(mu)
    fmuX = f(muX)

    # J L for E[J(x-mu)(x-mu)^T J^T]= J Sig J^T
    # map the jvp over the columns of L (in_axis=1)
    # and stack the results as columns (out_axis=1)
    JL = jax.vmap(
        lambda v : jax.jvp(f, (muX,), (v,))[1],
        in_axes=1,
        out_axes=1
        )(L)
    
    #  E[(x-mu)H(x-mu)] = Tr(H Sig) = Tr(L^t H L)
    # compute the Hessian-vector quadratic form with nested jvps
    # and then map over the columns of L, stacking the results as
    # columns (in_axis=1, out_axis=1).
    tr_HSig = jnp.sum(
                jax.vmap(
                lambda v: jax.jvp(
                    lambda z : jax.jvp(f, (z,), (v,))[1],
                    (muX,),
                    (v,)
                )[1],
                in_axes=1,
                out_axes=1
                )(L),
                axis=1
            )
    
    # MEAN #

    # compute the mean: muY = f(x0) + 0 + 1/2 * Tr[H Sig]
    muY = fmuX + 0.5 * tr_HSig

    # COVARIANCE #

    # compute the covariance
    # sigY = f(x0)f(x0)^T + J Sig J^T + 1/2 * (f(x0)Tr(HSig)^T + Tr(HSig)f(x0)^T) + O(dx^4) - muY muY^T
    SigY = (  
              jnp.einsum('a, b -> ab', fmuX, fmuX)
            + jnp.einsum('ai, ib -> ab', JL, JL.T)
            + 0.5 * (jnp.einsum('a, b -> ab', tr_HSig, fmuX)  + jnp.einsum('a,b -> ab', fmuX, tr_HSig))
            - jnp.einsum('a,b -> ab', muY, muY)
            )

    return muY, SigY

def linear_transformation(W:jnp.ndarray, b:jnp.ndarray, muX:jnp.ndarray, SigX:jnp.ndarray)-> Tuple[jnp.ndarray, jnp.ndarray]:
    """Propagate mean and covariance through a linear transformation y = Wx + b.
    
    Input
    W : matrix of shape (m, n)
    b : bias vector of shape (m,)
    muX : mean vector of shape (n,)
    SigX : covariance matrix of shape (n, n)
    
    Output
    muY : mean vector of shape (m,)
    SigY : covariance matrix of shape (m, m)
    """

    muY = W @ muX + b
    SigY = jnp.einsum('ij, jl, lk -> ik', W, SigX, W.T)

    return muY, SigY

def nonlinear_transformation(nonlinearity:Callable, muX:jnp.array, SigX:jnp.array)-> Tuple[jnp.array, jnp.array]:
    """Propagate mean and covariance through an elementwise nonlinearity using PTPE [Zhang & Ching JMLR (2025)].
    
    Input
    nonlinearity : elementwise nonlinearity from jax.nn either jax.nn.relu, jax.nn.tanh, jax.nn.gelu
    mu : mean vector of shape (n,)
    Sig : covariance matrix of shape (n, n)

    Output
    muY : mean vector of shape (n,)
    SigY : covariance matrix of shape (n, n)
    """

    if nonlinearity == jax.nn.identity:

        A0 = muX
        A1 = jnp.ones_like(muX)
        A2 = jnp.zeros_like(muX)
    
    elif nonlinearity == jax.nn.relu or nonlinearity == softplus:

        sig2 = jnp.diag(SigX)

        A0 = (
              0.5 * muX * lax.erfc(-muX / jnp.sqrt(2.0 * sig2)) 
            + jnp.sqrt(sig2) * jsp.stats.norm.pdf(muX / jnp.sqrt(sig2))
            )
        A1 = 0.5 * lax.erfc(-muX / jnp.sqrt(2.0 * sig2))
        A2 = 0.5 * (1.0 / jnp.sqrt(sig2)) * jsp.stats.norm.pdf(muX / jnp.sqrt(sig2))

    elif nonlinearity == jax.nn.tanh:

        sig2 = jnp.diag(SigX)

        gamma = jnp.asarray([0.5583, 0.8596, 0.8596, 1.2612]) # magic numbers from the paper.
        sig2prime = 1.0/(2.0 * jnp.power(gamma, 2.0))
        sig2hat = sig2 + sig2prime[:, None]

        A0 = jnp.mean(lax.erf(muX / jnp.sqrt(2.0 * sig2hat)), axis=0)
        A1 = jnp.mean((2.0 / jnp.sqrt(sig2hat)) * jsp.stats.norm.pdf(muX / jnp.sqrt(sig2hat)), axis=0)
        A2 = 0.5 * jnp.mean(-2.0 * (muX / jnp.power(sig2hat, 3.0/2.0)) * jsp.stats.norm.pdf(muX / jnp.sqrt(sig2hat)), axis=0)

    elif nonlinearity == jax.nn.sigmoid:

        sig2 = jnp.diag(SigX)

        gamma = jnp.asarray([0.2791, 0.4298, 0.04298, 0.6306]) # magic numbers from the paper.
        sig2prime = 1.0/(2.0 * jnp.power(gamma, 2.0))
        sig2hat = sig2 + sig2prime[:, None]

        A0 = jnp.mean(0.5 * lax.erfc(-muX / jnp.sqrt(2.0 * sig2hat)), axis=0)
        A1 = jnp.mean((1.0 / jnp.sqrt(sig2hat)) * jsp.stats.norm.pdf(muX / jnp.sqrt(sig2hat)), axis=0)
        A2 = 0.5 * jnp.mean(-1.0 * (muX / jnp.power(sig2hat, 3.0/2.0)) * jsp.stats.norm.pdf(muX / jnp.sqrt(sig2hat)), axis=0)

    elif nonlinearity == jax.nn.gelu:

        sig2 = jnp.diag(SigX)
        sig2hat = 1.0 + sig2

        A0 = (
              0.5 * muX * lax.erfc(-1.0* muX / jnp.sqrt(2.0 * sig2hat))
            + (sig2 / jnp.sqrt(sig2hat)) * jsp.stats.norm.pdf(muX / jnp.sqrt(sig2hat))
              )
        A1 = (
             0.5 * lax.erfc(-1.0 * muX / jnp.sqrt(2.0 * sig2hat))
            + (muX / jnp.power(sig2hat, 3.0/2.0)) * jsp.stats.norm.pdf(muX / jnp.sqrt(sig2hat))
        )
        A2 = (
            0.5 * (1.0 + (1.0 / sig2hat) - (jnp.power(muX / sig2hat, 2.0)))
            * jnp.sqrt(1.0 / sig2hat) * jsp.stats.norm.pdf(muX / jnp.sqrt(sig2hat))
        )
    
    else:

        raise NotImplementedError("Nonlinearity not implemented for PTPE.")

    muY = A0

    # transposed at different places because of row vec vs. column vec convention
    SigY = A1[None].T * SigX * A1 + 2.0 * A2[None].T * jnp.power(SigX, 2) * A2

    return muY, SigY


def ptpe_gcn(gcn:eqx.Module, A:sparse.BCOO, final_embedding:bool, muX:jnp.array, SigX:jnp.array)-> Tuple[jnp.array, jnp.array]:
    """Propagate mean and covariance through a GCN using PTPE.
    
    Input
    gcn : GCN model as equinox Module
    A : sparse adjacency matrix of the graph (num_nodes x num_nodes)
    final_embedding : whether to skip the final non-linearity (True) or not (False)
    muX : mean of the input random variable X in R^{num_nodes x num_features}
    SigX : covariance of the input random variable X in R^{(num_nodes*num_features) x (num_nodes*num_features)}
    
    Output
    muY : propagated mean in R^{num_nodes x num_classes}
    SigY : propagated covariance in R^{(num_nodes*num_classes) x (num_nodes*num_classes)}
    """

    # get the normalized adjacency for propagation
    num_nodes = A.shape[0]
    S = normalized_adjacency(A)
    S = S.todense()

    # propagate through hidden layers
    mu = muX
    Sig = SigX
    for layer in gcn.layers[:-1]:

        # get parameters of the layer
        W = layer.weights
        b = layer.bias

        # form the matrices for operating in vectorized form
        S_I = jnp.kron(S, jnp.eye(W.shape[1]))   # S X I_{num_features}
        I_W = jnp.kron(jnp.eye(num_nodes), W)    # I_{num_nodes} X W
        one_b = jnp.kron(jnp.ones(num_nodes), b) # 1_{num_nodes} X b

        # structural update S @ X
        mu, Sig = linear_transformation(S_I, jnp.zeros(num_nodes * W.shape[1]), mu, Sig)

        # weight update
        mu, Sig = linear_transformation(I_W, one_b, mu, Sig)

        # non-linearity using ptpe
        mu, Sig = nonlinear_transformation(gcn.non_linearity, mu, Sig)

    # propagate through final layer

    # get parameters of the layer
    W = gcn.layers[-1].weights
    b = gcn.layers[-1].bias

    # form the matrices for operating in vectorized form
    S_I = jnp.kron(S, jnp.eye(W.shape[1]))     # S x I_{num_classes}
    I_W = jnp.kron(jnp.eye(num_nodes), W)      # I_{num_nodes} x W
    one_b = jnp.kron(jnp.ones(num_nodes), b)   # 1_{num_nodes} x b

    # structural update S @ X
    mu, Sig = linear_transformation(S_I, jnp.zeros(num_nodes * W.shape[1]), mu, Sig)

    # weight update
    mu, Sig = linear_transformation(I_W, one_b, mu, Sig)

    # non-linearity using ptpe
    if final_embedding:
        muY, SigY = mu, Sig
    else:
        muY, SigY = nonlinear_transformation(gcn.final_non_linearity, mu, Sig)

    return muY, SigY

def taylor_nonlinearity(f:Callable, muX:jnp.array, SigX:jnp.array, order:int=1, gaussian_closure:bool=True)-> Tuple[jnp.array, jnp.array]:
    """Propagate mean and covariance through an elementwise non-linearity using Taylor expansion.
    
    Input
    f : elementwise non-linearity f: R -> R
    muX : mean of the input random variable (n,)
    SigX : covariance of the input random variable (n x n)
    order : order of the Taylor expansion (1 or 2)
    gaussian_closure : whether to use the gaussian closure approximation for the fourth moment in second order

    Output
    muY : propagated mean (n,)
    SigY : propagated covariance (n x n)
    """

    # compute 0, 1, 2 derivatives at muX 
    f_mu = f(muX)
    dfdx_mu = jax.vmap(jax.jacfwd(f), in_axes=0, out_axes=0)(muX)
    df2dx2_mu = jax.vmap(jax.jacfwd(jax.jacfwd(f)), in_axes=0, out_axes=0)(muX)

    # extract the diagonal of the covariance matrix
    sig2X = jnp.diag(SigX)

    # propagate the mean
    if order==1:
        muY = f_mu
    elif order==2: 
        muY = f_mu + 0.5 * df2dx2_mu * sig2X
    else:
        raise NotImplementedError("Order needs to be 1 or 2.")

    # propagate the covariance matrix:
    if order==1:
        SigY = dfdx_mu[None].T * SigX * dfdx_mu
    elif order==2:
        SigY = (
                jnp.einsum('a,b -> ab', f_mu, f_mu)
                + jnp.einsum('a, b -> ab', f_mu, 0.5 * df2dx2_mu * sig2X)
                + jnp.einsum('a, b -> ab', 0.5 * df2dx2_mu * sig2X, f_mu)
                + dfdx_mu[None].T * SigX * dfdx_mu
                - jnp.einsum('a, b -> ab', muY, muY)
            )
        if gaussian_closure:
            SigY += (
                      0.25 * (jnp.einsum('a, b -> ab', df2dx2_mu * sig2X, sig2X * df2dx2_mu)
                    + 2.0 * dfdx_mu[None].T * SigX * SigX * df2dx2_mu)
                    )
    else:
        raise NotImplementedError("Order needs to be 1 or 2.")
    
    return muY, SigY


def taylor_gcn(gcn:eqx.Module, A:sparse.BCOO, final_embedding:bool, muX:jnp.array, SigX:jnp.array, order:int=2, gaussian_closure:bool=False, eps:float=0., alternative_nonlinearity:Optional[Callable]=None)-> Tuple[jnp.array, jnp.array]:
    """Propagate mean and covariance through a GCN using Taylor expansion.
    
    Input
    gcn : GCN model as equinox Module
    A : sparse adjacency matrix of the graph (num_nodes x num_nodes)
    final_embedding : whether to skip the final non-linearity (True) or not (False)
    order : order of the Taylor expansion (1 or 2)
    gaussian_closure : whether to use the gaussian closure approximation for the fourth moment in second order
    eps : small regularizer added to SigX for numerical stability
    alternative_nonlinearity : if not None, use this non-linearity instead of gcn
    muX : mean of the input random variable X in R^{num_nodes x num_features}
    SigX : covariance of the input random variable X in R^{(num_nodes*num_features) x (num_nodes*num_features)}
    
    Output
    muY : propagated mean in R^{num_nodes x num_classes}
    SigY : propagated covariance in R^{(num_nodes*num_classes) x (num_nodes*num_classes)}
    """

    # get the normalized adjacency for propagation
    num_nodes = A.shape[0]
    S = normalized_adjacency(A)
    S = S.todense()

    if alternative_nonlinearity is not None:
        nonlinearity = alternative_nonlinearity
    else:
        nonlinearity = gcn.non_linearity

    # propagate through hidden layers
    mu = muX
    Sig = SigX
    for layer in gcn.layers[:-1]:

        # get parameters of the layer
        W = layer.weights
        b = layer.bias

        # form the matrices for operating in vectorized form
        S_I = jnp.kron(S, jnp.eye(W.shape[1]))   # S X I_{num_features}
        I_W = jnp.kron(jnp.eye(num_nodes), W)    # I_{num_nodes} X W
        one_b = jnp.kron(jnp.ones(num_nodes), b) # 1_{num_nodes} X b

        # structural update S @ X
        mu, Sig = linear_transformation(S_I, jnp.zeros(num_nodes * W.shape[1]), mu, Sig)

        # weight update
        mu, Sig = linear_transformation(I_W, one_b, mu, Sig)

        # non-linearity using quadratic
        mu, Sig = taylor_nonlinearity(nonlinearity, mu, Sig, order=order, gaussian_closure=gaussian_closure)

    # propagate through final layer

    # get parameters of the layer
    W = gcn.layers[-1].weights
    b = gcn.layers[-1].bias

    # form the matrices for operating in vectorized form
    S_I = jnp.kron(S, jnp.eye(W.shape[1]))     # S x I_{num_classes}
    I_W = jnp.kron(jnp.eye(num_nodes), W)      # I_{num_nodes} x W
    one_b = jnp.kron(jnp.ones(num_nodes), b)   # 1_{num_nodes} x b

    # structural update S @ X
    mu, Sig = linear_transformation(S_I, jnp.zeros(num_nodes * W.shape[1]), mu, Sig)

    # weight update
    mu, Sig = linear_transformation(I_W, one_b, mu, Sig)

    # non-linearity using ptpe
    if final_embedding:
        muY, SigY = mu, Sig
    else:
        # these non-linearity is often not element-wise, so we use full Taylor
        if order==1:
            muY, SigY = linear_propagation(gcn.final_non_linearity, mu, Sig, eps=eps)
        elif order==2:
            muY, SigY = quadratic_propagation(gcn.final_non_linearity, mu, Sig, eps=eps, gaussian_closure=gaussian_closure)
        else:
            raise NotImplementedError("Order needs to be 1 or 2.")

    return muY, SigY


### GRAPH MACHINE LEARNING ###
def accuracy(model:eqx.Module, data:Tuple[sparse.BCOO, jnp.ndarray, jnp.ndarray], idx:jnp.ndarray) -> float:
    """Compute accuracy of a graph machine learning model

    Input
    model: equinox model
    data: tuple (A, X, y) where A is a sparse adjacency matrix, X are node features, and y are node labels
    idx: indices of nodes to compute accuracy on

    Output
    acc: accuracy value
    """

    A, X, y = data

    y_pred = model(X, A)
    acc = jnp.asarray(jnp.argmax(y_pred[idx], axis=1)==jnp.argmax(y[idx], axis=1), dtype=jnp.float32).mean()

    return acc


def train(model:eqx.Module, optimizer, data:Tuple[sparse.BCOO, jnp.ndarray, jnp.ndarray], train_idx:jnp.ndarray, num_reps:int)-> Tuple[eqx.Module, List[float]]:
    """Training loop for a graph machine learning model

    Input
    model: equinox model
    optimizer: optax optimizer
    data: tuple (A, X, y) where A is a sparse adjacency matrix, X are node features, and y are node labels
    train_idx: indices of training nodes
    num_reps: number of training iterations

    Output
    model: trained model
    loss_list: list of training losses over iterations
    """

    def loss_fun(model, train_idx, A, X, y):

        y_pred = model(X, A)
        loss_val = -jnp.mean(jnp.sum(y[train_idx] * jnp.log(y_pred[train_idx] + 1e-15), axis=1))

        return loss_val

    @eqx.filter_jit
    def step(model, train_idx, A, X, y, opt_state):

        loss_val, grads = eqx.filter_value_and_grad(loss_fun)(model, train_idx, A, X, y)
        

        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss_val
    
    # unpack data
    A, X, y = data

    # init optimizer
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    loss_list = []
    for _ in tqdm.tqdm(range(num_reps)):
        
        model, opt_state, loss_val = step(model, train_idx, A, X, y, opt_state)
        loss_list.append(loss_val.item())

    return model, loss_list


### DISTANCE ###
def FCD(X:jnp.ndarray, A_sparse:sparse.BCOO, max_samples:int=100, p:int=2, node_subset=None)->jnp.ndarray:
    """Compute the Feature Convolution Distance.
    
    Input
    X : node features (num_nodes * num_features, num_samples)
    A_sparse : sparse adjacency matrix of the graph (num_nodes, num_nodes)
    max_samples : maximum number of samples to use for estimating the Wasserstein distance
    p : p-norm to use for the Wasserstein distance (1 or 2)
    node_subset : subset of nodes to compute the distance for (if None, use all nodes)

    Output
    dXS : vector of pairwise distances between the nodes in node_subset
    """

    # set metric
    if p==1:
        metric = 'euclidean'
    elif p==2:
        metric = 'sqeuclidean'
    else:
        raise ValueError("p must be 1 or 2.")
    
    # subsetting
    if node_subset is None:
        node_subset = np.arange(A_sparse.shape[0])

    # determine dimensions
    num_nodes = A_sparse.shape[0]
    num_features = X.shape[0]//num_nodes
    num_samples = X.shape[1]

    # compute structural update matrix
    S = normalized_adjacency(A_sparse)
    S = S.todense()
    S_I = jnp.kron(S, jnp.eye(num_features))
    
    # compute the structural update
    XS = S_I @ X

    # estimate Wasserstein distance between samples
    XS = np.asarray(XS).reshape(num_nodes, num_features, num_samples)

    max_id = min(max_samples, num_samples)
    dXS = []
    for i in tqdm.tqdm(node_subset):
        for j in node_subset:
                if i<j:
                    M = ot.dist(XS[i, :, :max_id].T, XS[j, :, :max_id].T, metric=metric)
                    G = ot.emd(ot.unif(max_id), ot.unif(max_id), M)
                    if p==1:
                        dXS.append(np.sum(G * M).item())
                    elif p==2:
                        dXS.append(np.sqrt(np.sum(G * M)).item())

    dXS = jnp.asarray(dXS)
    return dXS

def wasserstein_sample(X1:jnp.ndarray, X2:jnp.ndarray, p:int=2)->jnp.ndarray:
    """"Compute the pairwise Wasserstein distance between nodes based on samples of their features.
    
    Input 
    X1 : feature samples of the first set of nodes (num_nodes_1, num_features, num_samples) or (num_nodes_1, num_samples)
    X2 : feature samples of the second set of nodes (num_nodes_2, num_features, num_samples) or (num_nodes_2, num_samples)
    p  : p-norm to use for the Wasserstein distance (1 or 2)

    Output
    d : vector of pairwise distances between the nodes in X1 and X2
    """

    # ensure numpy
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    # set metric
    if p==1:
        metric = 'euclidean'
    elif p==2:
        metric = 'sqeuclidean'
    else:
        raise ValueError('p must be 1 or 2.')
    
    # test whether the node sets are the same
    if X1.shape == X2.shape and jnp.all(X1 == X2):
        same_data = True
    else:
        same_data = False

    # optimal transport
    d = []
    for i in tqdm.tqdm(range(X1.shape[0])):
        for j in range(X2.shape[0]):

            # avoid overcounting if the node sets are the same
            if same_data and i >= j:
                continue
            else:
                
                # scalar node features
                if len(X1.shape)==2 and len(X2.shape)==2:
                    M = ot.dist(X1[i, None, :].T, X2[j, None, :].T, metric=metric)
                    G = ot.emd(ot.unif(X1.shape[1]), ot.unif(X2.shape[1]), M)
                # vector node features
                elif len(X1.shape)==3 and len(X2.shape)==3:
                    M = ot.dist(X1[i, :, :].T, X2[j, :, :].T, metric=metric)
                    G = ot.emd(ot.unif(X1.shape[2]), ot.unif(X2.shape[2]), M)
                else:
                    raise RuntimeError('X1, X2 must both be either 2D or 3D arrays.')
                
                # metric
                if p==1:
                    d.append(np.sum(G * M).item())
                elif p==2:
                    d.append(np.sqrt(np.sum(G * M)).item())
                else:
                    raise ValueError('p must be 1 or 2.')

    return jnp.asarray(d)