"""
utils_cohen.py
----------

Utility functions for evaluating the Cohen et al. [1] baseline.

[1] Cohen et al. (ICML 2019) Certified Robustness via Randomized Smoothing
https://proceedings.mlr.press/v97/cohen19c.html.

M. Laber, 2026/02
"""

### IMPORTS ###
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import scipy as sp

from typing import Tuple, Callable

from utils import *
from models import  *

def sample_under_noise(key:jax.random.PRNGKey, f:Callable, X:jnp.array, num_samples:int, sigma:float, verbose:bool=True)->jnp.array:
    """Returns how often x is classified into each class under isotropic Gaussian noise with zero mean
       and standard deviation sigma.
    
    Input
    key : jax random key
    f : GCN that maps feature matrix (num_nodes, num_features) to one-hot predictions (num_nodes, num_classes)
    X : input data point (num_nodes, num_features)
    num_samples : number of samples to draw
    sigma : standard deviation of the Gaussian noise
    verbose : whether to print progress

    Output
    counts : array of shape (num_classes,) with the number of times each class was predicted
    """

    keys = jax.random.split(key, num_samples)
    for i, key in enumerate(keys):
        if verbose: print(f"Sampling {i+1}/{num_samples}", end='\r')
        # sample noise
        noise = jax.random.normal(key, shape=X.shape) * sigma
        X_noisy = X + noise
        # predict
        ypred = f(X_noisy)

        # initialize counts
        if i==0:
            counts = jnp.zeros(ypred.shape[1])
        
        # increment counts
        counts += (ypred == ypred.max(axis=1, keepdims=True)).astype(jnp.float16)

    return counts

def binom_p_value(k:int, n:int, p0:float)->float:
    """Two-sided binomial test p-value.
    
    Input
    k : observed number of successes
    n : number of trials
    p0 : null hypothesis for the probability of success

    Output
    p_value : p-value of the two-sided binomal test
    """

    # probability for k successes under null hypothesis
    p = jsp.stats.binom.pmf(k, n, p0)

    # probability for all other outcomes
    ps = jsp.stats.binom.pmf(jnp.arange(0, n+1), n, p0)

    # two-sided p-value
    p_value = jnp.sum(ps[ps <= p])

    return p_value

def lower_confidence_bound(k:int, n:int, alpha:float)->float:
    """Clopper-Pearson lower confidence bound for a binomial proportion.
    
    Input
    k : observed number of successes
    n : number of trials
    alpha : significance level

    Output
    p_lower : lower confidence bound
    """

    if k == 0:
        p_lower = 0.0
    else:
        p_lower = sp.stats.beta.ppf(2 * (alpha/2.), k, n - k + 1)

    return p_lower
    
def predict_cohen(key:jax.random.PRNGKey, f:Callable, sigma:float, X:jnp.array, num_samples:int, alpha:float, verbose:bool=True)->jnp.array:
    """Predict with the possibility to abstain using Cohen's method.

    Input
    key : key for random number generation
    f : GCN that maps feature matrix (num_nodes, num_features) to one-hot predictions (num_nodes, num_classes)
    sigma : standard deviation for isotropic Gaussian noise
    X : inputs (num_nodes, num_features)
    num_samples : number of samples to determine whether to abstain
    alpha : significance level
    verbose : whether to print progress

    Output
    ypred : vector (num_nodes,) with predicted class or -1 for abstained.
    """

    # create class counts for noise perturbed inputs
    counts = sample_under_noise(key, f, X, num_samples, sigma, verbose=verbose)

    # get the two highest counts (nA >= nB)
    nBnA = jnp.sort(counts, axis=1)[:, -2:]
    
    # iterate over nodes to make predictions
    ypred = []
    for v in range(X.shape[0]):
        
        if verbose: print(f"predicting node {v+1}/{X.shape[0]}", end='\r')

        nB, nA = nBnA[v, :]

        # conduct two-sided binomial test
        p_value = binom_p_value(nA.item(), nA.item() + nB.item(), 0.5)

        if p_value < alpha:
            # certified prediction
            ypred.append(jnp.argmax(counts[v, :]).item())
        else:
            # abstain
            ypred.append(-1)
    
    return jnp.array(ypred)

def certify_cohen(key:jax.random.PRNGKey, f:Callable, sigma:float, X:jnp.array, num_samples:int | Tuple[int,int], alpha:float, verbose:bool=True)->Tuple[jnp.array, jnp.array]:
    """Prediction with certified radii using Cohen's method.
    
    Input
    key : key for random number generation
    f : GCN that maps feature matrix (num_nodes, num_features) to one-hot predictions (num_nodes, num_classes)
    sigma : standard deviation for isotropic Gaussian noise
    X : feature matrix (num_nodes, num_features)
    num_samples : tuple of small sample size, large sample size or integer than both sample sizes are the same
    alpha : significance level
    verbose : whether to print progress

    Output
    y : predicted class or -1 for abstained (num_nodes,)
    r : robustness radius or -1 for abstained (num_nodes,)
    """

    # check that num_samples is valid
    if isinstance(num_samples, int):
        num_samples0 = num_samples
        num_samples1 = num_samples
    elif isinstance(num_samples, tuple) and len(num_samples) == 2:
        num_samples0 = num_samples[0]
        num_samples1 = num_samples[1]
    else:
        raise ValueError("num_samples must be of length 1 or 2")

    # create class counts for noise perturbed inputs (first sample, usually small)
    key, key_ = jax.random.split(key)
    counts0 = sample_under_noise(key, f, X, num_samples0, sigma, verbose=verbose)

    # estimate the most probable class for each node
    cA = jnp.argmax(counts0, axis=1)

    # create class counts for noise perturbed inputs (second sample, usually large)
    key, key_ = jax.random.split(key)
    counts1 = sample_under_noise(key_, f, X, num_samples1, sigma, verbose=verbose)

    # prediction and radius
    y, r = [], []
    for v in range(X.shape[0]):

        if verbose: print(f"predicting node {v+1}/{X.shape[0]}", end='\r')

        # lower confidence bound
        pAlow = lower_confidence_bound(counts1[v, cA[v]], num_samples1, alpha)

        if pAlow > 0.5:
            # return predicted class and radius
            y.append(cA[v])
            # radius
            r.append(sigma * sp.stats.norm.ppf(pAlow))
        else:
            # abstain
            y.append(-1)
            r.append(-1)

    y = jnp.array(y)
    r = jnp.array(r)
    
    return y, r