from functools import partial

import numpy as np
from scipy.stats import truncnorm

from conditional import ytilde, sigmaytilde
from params import X, ndraws, tol


def ptrn2(mu, quantile, ltilde, utilde, sigma, N, seed=100):
    """ Approximation of the cumulative distribution function of the truncated normal distribution
    :param mu: mean
    :param quantile: quantile
    :param ltilde: lower truncation point
    :param utilde: upper truncation point
    :param sigma: covariance matrix
    :param N: number of samples
    :param seed: random seed
    :return: tail probability
    """

    np.random.seed(seed)
    tail_prob = np.mean(
        truncnorm.ppf(
            q=np.random.uniform(N),
            a=[(ltilde - mu) / sigma] * N,
            b=np.array([(utilde - mu) / sigma] * N) + mu
        ) <= ((quantile - mu) / sigma)
    )

    return tail_prob


def etrn2(mu, ltilde, utilde, sigma, N, seed=100):
    """ Computation of the mean of the truncated normal distribution
    :param mu: mean
    :param ltilde: lower truncation point
    :param utilde: upper truncation point
    :param sigma: covariance matrix
    :param N: number of samples
    :param seed: random seed
    :return: tail probability
    """
    
    np.random.seed(seed)
    tail_prob = sigma * np.mean(
        truncnorm.ppf(
            q=np.random.uniform(N),
            a=[(ltilde - mu) / sigma] * N,
            b=np.array([(utilde - mu) / sigma] * N) + mu
        )
    )

    return tail_prob


def cutrn(mu, quantile, ltilde, utilde, sigma, seed=100):
    """ Find the threshold for confidence region evaluation
    :param mu: mean
    :param quantile: quantile
    :param ltilde: lower truncation point
    :param utilde: upper truncation point
    :param sigma: covariance matrix
    :param seed: random seed
    :return:
    """

    np.random.seed(seed)
    cut = sigma * truncnorm.ppf(
        q=quantile,
        a=(ltilde - mu) / sigma,
        b=(utilde - mu) / sigma) + mu

    return cut


def get_median(ltilde, utilde):
    """ Get median estimate via bisection search algorithm
    :param ltilde: the lower truncation value
    :param utilde: the upper truncation value
    :return:
    """

    yhat = ytilde
    sigmayhat = sigmaytilde
    k = len(X)
    size = 0.5

    # initialize loop
    check_uniroot = False
    while check_uniroot is False:
        scale = k
        mugridsl = yhat - scale * np.sqrt(sigmayhat)
        mugridsu = yhat + scale * np.sqrt(sigmayhat)
        mugrids = [np.float(mugridsl), np.float(mugridsu)]
        ptrn2_ = partial(ptrn2, Q=yhat, A=ltilde, B=utilde, SIGMA=np.sqrt(sigmayhat), N=ndraws)
        intermediate = np.array(list(map(ptrn2_, mugrids))) - (1 - size)

        halt_condition = abs(max(np.sign(intermediate)) - min(np.sign(intermediate))) > tol
        if halt_condition is True:
            check_uniroot = True
        if halt_condition is False:
            k = 2 * k

    # initialize loop
    mugrids = np.array([0] * 3)
    halt_condition = False
    while halt_condition is False:
        mugridsm = (mugridsl + mugridsu) / 2
        previous_line = mugrids
        mugrids = np.array([np.float(mugridsl), np.float(mugridsm), np.float(mugridsu)])
        ptrn2_ = partial(ptrn2, Q=yhat, A=ltilde, B=utilde, SIGMA=np.sqrt(sigmayhat), N=ndraws)
        intermediate = np.array(list(map(ptrn2_, mugrids))) - (1 - size)

        if max(abs(mugrids - previous_line)) == 0:
            halt_condition = True

        if (abs(intermediate[1]) < tol) or (abs(mugridsu - mugridsl) < tol):
            halt_condition = True

        if np.sign(intermediate[0]) == np.sign(intermediate[1]):
            mugridsl = mugridsm

        if np.sign(intermediate[2]) == np.sign(intermediate[1]):
            mugridsu = mugridsm

        med_u_estimate = mugridsm

    return med_u_estimate
