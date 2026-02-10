"""
models.py
----------

SGC and GCN implementations, and various helper functions.

M. Laber, 2026/02
"""

import jax
from jax import lax
import jax.experimental.sparse as sparse
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple, Callable, Optional

def softplus(x:jnp.ndarray, beta:float=10.0)->jnp.ndarray:
    """Differentiable approximation to the ReLU function.
    
    Input
    x : input array
    beta : approximation parameter, beta -> infinity gives exact ReLU

    Output
    softplus(x) : approximated ReLU of x
    
    """

    return jnp.where(
        x > 0,
        x + (1.0 / beta) * jnp.log1p(jnp.exp(-beta * x)),  # for positive x
        (1.0 / beta) * jnp.log1p(jnp.exp(beta * x))        # for negative x
    )


def flatten_model(model, A_sparse, alternative_nonlinearity=None, final_embedding=True):
    """Return a model that operates on (num_nodes * num_features,) shaped input vectors and returns (num_nodes*num_classes,) shaped output vectors.
    
    Input
    model: a GCN or SGC model as eqx.Module
    A_sparse: sparse adjacency matrix as sparse.BCOO

    Output
    flat_model: a function that takes (num_nodes * num_features,) shaped input vectors and returns (num_nodes*num_classes,) shaped output vectors
    """

    num_nodes = A_sparse.shape[0]
    num_features = model.layers[0].weights.shape[1]

    def flat_model(Xvec):

        return model(Xvec.reshape(num_nodes, num_features), A_sparse, alternative_nonlinearity=alternative_nonlinearity, final_embedding=final_embedding).reshape(-1)

    return flat_model

def normalized_adjacency(A: sparse.bcoo) -> sparse.bcoo:
    """"Compute the normalized adjacency with added self-loops.
    
    Input
    A: sparse adjacency matrix

    Output
    Aprime: normalized adjacency matrix with self-loops
    """

    # add self-loops
    A = sparse.eye(A.shape[0]) + A
    values = A.data
    indices = A.indices
    
    # compute degrees matrix and scale adjacency
    deg = A.sum(axis=1).todense().flatten()

    values *= 1.0 / jnp.sqrt(deg[indices[:, 0]])
    values *= 1.0 / jnp.sqrt(deg[indices[:, 1]])
    
    Aprime = sparse.BCOO((values, indices), shape=A.shape)
    
    return Aprime


class LinearLayer(eqx.Module):
    """"A linear layer with weights and bias."""

    weights : jnp.ndarray
    bias : jnp.ndarray

    def __init__(self, key:jax.random.PRNGKey, in_features:int, out_features:int):
        """Initialize the linear layer with random weights and zero bias.
        
        Input
        key: a key for random number generation
        in_features: number of input features
        out_features: number of output features
        """

        self.weights = jax.random.normal(key, shape=(out_features, in_features)) / jnp.sqrt(in_features)
        self.bias = jnp.zeros((out_features,))
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        return x @ self.weights.T + self.bias

class SGC(eqx.Module):
    """Simplified Graph Convolutional Network (SGC) model."""

    num_layers : int
    linear : LinearLayer
    non_linearity : Callable

    def __init__(self, key:jax.random.PRNGKey, n_layers:int, n_features:int, n_classes:int, non_linearity:Callable=jax.nn.identity):
        """Initialize the SGC model.

        Input
        key: a key for random number generation
        n_layers: number of propagation layers
        n_features: number of input features
        n_classes: number of output classes
        non_linearity: non-linearity function to apply after complete propagation
        """

        self.num_layers = n_layers
        self.linear = LinearLayer(key=key, in_features=n_features, out_features=n_classes)
        self.non_linearity = non_linearity

    def __call__(self, X: jnp.ndarray, A: sparse.bcoo, final_embedding:bool=False) -> jnp.ndarray:

        # normalize adjacency
        S = normalized_adjacency(A)

        # Propagation on the graph
        def step(xin, _):
            xout = S @ xin
            return xout, None

        X_prop, _ = lax.scan(step, X, xs=None, length=self.num_layers)

        # Weight update
        X_weight = self.linear(X_prop)

        # Apply non-linearity and weights
        if final_embedding:
            X_out = X_weight
        else:
            X_out = self.non_linearity(X_weight)

        return X_out

class GCN(eqx.Module):
    """Graph Convolutional Network (GCN) model."""

    dims : List[int]
    non_linearity : Callable
    final_non_linearity : Callable
    layers : List[LinearLayer]

    def __init__(self, key:jax.random.PRNGKey, dims:List[int], non_linearity:Callable=jax.nn.gelu, final_non_linearity:Callable=jax.nn.identity):
        """Initialize the GCN model.

        Input
        key: a key for random number generation
        dims: list of layer dimensions (including input and output dimensions)
        non_linearity: non-linearity function to apply after each layer except the last
        final_non_linearity: non-linearity function to apply after the last layer
        """

        keys = jax.random.split(key, len(dims) - 1)

        self.dims = dims
        self.non_linearity = non_linearity
        self.final_non_linearity = final_non_linearity

        self.layers = [LinearLayer(key=keys[i], in_features=dims[i], out_features=dims[i+1]) for i in range(len(dims) - 1)]

    def __call__(self, X: jnp.ndarray, A: sparse.bcoo, final_embedding:bool=False, alternative_nonlinearity:Optional[Callable]=None) -> jnp.ndarray:

        S = normalized_adjacency(A)
        
        # Propagation on the graph
        for layer in self.layers[:-1]:
            X = S @ X
            X = layer(X)
            
            if alternative_nonlinearity is not None:
                X = alternative_nonlinearity(X)
            else:
                X = self.non_linearity(X)
        
        # Final layer without non-linearity
        X = S @ X
        X = self.layers[-1](X)

        if final_embedding:
            X_out = X
        else:
            X_out = self.final_non_linearity(X)

        return X_out