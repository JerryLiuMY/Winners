import numpy as np
import seaborn as sns
sns.set()


def data_prep(nsamples, narms, diff):
    """ Generate data for experiments
    :param nsamples: number of samples
    :param narms: number of treatment
    :param diff: difference between winning arm and the remaining arms
    :return:
    """

    mu = np.zeros(narms)
    mu[0] = mu[0] + diff
    sigma = np.eye(narms)
    Y_all = np.random.multivariate_normal(mu, sigma, size=nsamples)

    return Y_all, sigma
