from joblib import Parallel, delayed
from data_prep.dgp import DGP
from models.naive import Naive
from models.rd import RD
from models.winners import Winners
from datetime import datetime
import numpy as np
import multiprocessing


def simulation(ntrials, narms, nsamples, mu, cov, ntests, ntrans):
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

    # logging message
    num_cores = multiprocessing.cpu_count()
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on ntests={ntests} and ntrans={ntrans} "
          f"[num_cores={num_cores}]")

    # calculate pvalues
    parallel = Parallel(n_jobs=num_cores)
    pvals = parallel(delayed(simulation_process)(narms, nsamples, mu, cov, ntests, ntrans) for _ in range(ntrials))
    pvals = np.array(pvals)
    power_naive = np.mean(pvals[:, 0] <= 0.05)
    power_winners = np.mean(pvals[:, 1] <= 0.05)
    power_rd = np.mean(pvals[:, 2] <= 0.05)
    powers = [power_naive, power_winners, power_rd]

    return powers


def simulation_process(narms, nsamples, mu, cov, ntests, ntrans):
    """ Single process for experiment
    :param nsamples: number of samples
    :param narms: number of treatment
    :param mu: mean of the data generation
    :param cov: covariance of the data generation
    :param ntests: Number of tests for RD
    :param ntrans: Number of trans for RD
    :return:
    """

    # generate data
    dgp = DGP(narms=narms, nsamples=nsamples, mu=mu, cov=cov)
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
