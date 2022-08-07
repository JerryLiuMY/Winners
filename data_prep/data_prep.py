import numpy as np
import seaborn as sns
sns.set()


def data_prep(ntreat, diff):
    """ Generate data for experiments
    :param ntreat: number of treatment
    :param diff: difference between winning arm and the remaining arms
    :return:
    """

    mu = np.zeros(ntreat)
    mu[0] = mu[0] + diff
    sigma = np.eye(ntreat)
    Y_all = np.random.multivariate_normal(mu, sigma, size=100000)

    return Y_all, sigma
