import numpy as np
from scipy.stats import truncnorm


def ptrn2(mu, quantile, ltilde, utilde, sigma, N, seed=100):
    """ Approximation of the cumulative distribution function of the truncated normal distribution
    :param mu: mean
    :param quantile: quantile
    :param ltilde: lower truncation point
    :param utilde: upper truncation point
    :param sigma:
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
    :param sigma:
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
    :param sigma:
    :param seed: random seed
    :return:
    """

    np.random.seed(seed)
    cut = sigma * truncnorm.ppf(
        q=quantile,
        a=(ltilde - mu) / sigma,
        b=(utilde - mu) / sigma) + mu

    return cut
