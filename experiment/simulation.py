import numpy as np
from joblib import Parallel, delayed
from data_prep.dgp import DGP
from models.naive import Naive
from models.rd import RD
from models.winners import Winners
import multiprocessing


def simulation(ntrials, nsamples, narms, mu, cov, ntests, ntrans):
    """ RUn simulation for calculating power
    :param ntrials: Number of trails
    :param nsamples: number of samples
    :param narms: number of treatment
    :param mu: mean of the data generation
    :param cov: covariance of the data generation
    :param ntests: Number of tests for RD
    :param ntrans: Number of trans for RD
    :return: array of pvalues
    """

    narms = 5
    mu = np.arange(5) - 4
    cov = np.ones(5)
    num_cores = multiprocessing.cpu_count()
    print("num_cores={}".format(num_cores))

    parallel = Parallel(n_jobs=num_cores)
    pvals = parallel(delayed(process)(trial, nsamples, narms, mu, cov, ntests, ntrans) for trial in range(ntrials))
    pvals = np.array(pvals)
    rprob_naive = np.mean(pvals[:, 0] <= 0.05)
    rprob_winners = np.mean(pvals[:, 1] <= 0.05)
    rprob_rd = np.mean(pvals[:, 2] <= 0.05)
    rprobs = [rprob_naive, rprob_winners, rprob_rd]

    return rprobs


def process(nsamples, narms, mu, cov, ntests, ntrans):
    """ Single process for experiment
    :param nsamples: number of samples
    :param narms: number of treatment
    :param mu: mean of the data generation
    :param cov: covariance of the data generation
    :param ntests: Number of tests for RD
    :param ntrans: Number of trans for RD
    :return:
    """

    dgp = DGP(nsamples, narms=narms, mu=mu, cov=cov)
    Y, Z = dgp.get_data()
    Y_mu, sigma = dgp.get_input()

    # naive method
    naive = Naive(Y_mu, sigma)
    pval_naive = naive.get_test(null=0)

    # winners method
    winners = Winners(Y_mu, sigma)
    pval_winners = winners.get_test(null=0)

    # rd method
    rd = RD(Y, Z, int(len(Z) / 2))
    pval_rd = rd.multiple_test(ntests, ntrans)

    return pval_naive, pval_winners, pval_rd
