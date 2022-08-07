from functools import partial
from scipy.stats import truncnorm
from params.params import tol
import numpy as np


def search_mu_(ytilde, sigmaytilde, k1, ltilde, utilde, alpha):
    """ Get median estimate via bisection search algorithm
    :param ltilde: the lower truncation value
    :param utilde: the upper truncation value
    :param alpha: quantile to find inverse
    :return:
    """

    yhat = ytilde
    sigmayhat = sigmaytilde
    k = k1

    # initialize loop
    check_uniroot = False
    while check_uniroot is False:
        scale = k
        mugridsl = yhat - scale * np.sqrt(sigmayhat)
        mugridsu = yhat + scale * np.sqrt(sigmayhat)
        mugrids = np.array([np.float(mugridsl), np.float(mugridsu)])
        ptrn2_ = partial(ptrn2, y=yhat, ltilde=ltilde, utilde=utilde, std=np.sqrt(sigmayhat), N=1)
        intermediate = np.array(list(map(ptrn2_, mugrids))) - (1 - alpha)
        halt_condition = abs(max(np.sign(intermediate)) - min(np.sign(intermediate))) > tol
        if halt_condition:
            check_uniroot = True
        else:
            k = 2 * k

    # initialize loop
    mugrids = np.array([0] * 3)
    halt_condition = False
    while halt_condition is False:
        mugridsm = (mugridsl + mugridsu) / 2
        previous_line = mugrids
        mugrids = np.array([np.float(mugridsl), np.float(mugridsm), np.float(mugridsu)])
        ptrn2_ = partial(ptrn2, y=yhat, ltilde=ltilde, utilde=utilde, std=np.sqrt(sigmayhat), N=1)
        intermediate = np.array(list(map(ptrn2_, mugrids))) - (1 - alpha)

        if max(abs(mugrids - previous_line)) == 0:
            halt_condition = True

        if (abs(intermediate[1]) < tol) or (abs(mugridsu - mugridsl) < tol):
            halt_condition = True

        if np.sign(intermediate[0]) == np.sign(intermediate[1]):
            mugridsl = mugridsm

        if np.sign(intermediate[2]) == np.sign(intermediate[1]):
            mugridsu = mugridsm

        mu_estimate = mugridsm

    return mu_estimate


def ptrn2(mu, y, ltilde, utilde, std, N, seed=100):
    """ Approximation of the cumulative distribution function of the truncated normal distribution
    :param mu: mean
    :param y: observation y
    :param ltilde: lower truncation point
    :param utilde: upper truncation point
    :param std: standard deviation of winning arm
    :param N: number of samples
    :param seed: random seed
    :return: tail probability
    """

    np.random.seed(seed)
    tail_prob = np.mean(
        truncnorm.ppf(
            q=np.random.uniform(N),
            a=[(ltilde - mu) / std] * N,
            b=np.array([(utilde - mu) / std] * N) + mu
        ) <= ((y - mu) / std)
    )

    return tail_prob


def etrn2(mu, ltilde, utilde, sigma, N, seed=100):
    """ Computation of the mean of the truncated normal distribution
    :param mu: mean
    :param ltilde: lower truncation point
    :param utilde: upper truncation point
    :param sigma: variance of winning arm
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
