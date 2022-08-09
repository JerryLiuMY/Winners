import numpy as np
from joblib import Parallel, delayed
from data_prep.dgp import DGP
from main import num_cores
from models.naive import Naive
from models.rd import RD
from models.winners import Winners


def simulation(ntrials, nsamples, ntests, ntrans):
    """ RUn simulation for calculating power
    :param ntrials: Number of trails
    :param nsamples: number of samples
    :param ntests: Number of tests for RD
    :param ntrans: Number of trans for RD
    :return: array of pvalues
    """

    pvals = Parallel(n_jobs=num_cores)(delayed(process)(trial, nsamples, ntests, ntrans) for trial in range(ntrials))
    pvals = np.array(pvals)
    rprob_naive = np.mean(pvals[:, 0] <= 0.05)
    rprob_winners = np.mean(pvals[:, 1] <= 0.05)
    rprob_rd = np.mean(pvals[:, 2] <= 0.05)
    rprobs = [rprob_naive, rprob_winners, rprob_rd]

    return rprobs


def process(nsamples, ntests, ntrans):
    """ Single process for experiment
    :param nsamples: number of samples
    :param ntests: Number of tests for RD
    :param ntrans: Number of trans for RD
    :return:
    """

    dgp = DGP(nsamples, narms=5, mu=np.arange(5) - 4, cov=np.ones(5))
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
