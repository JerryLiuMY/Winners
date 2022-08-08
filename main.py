from joblib import Parallel, delayed
from data_prep.dgp import DGP
from models.winners import Winners
from models.naive import Naive
from models.rd import RD
import multiprocessing
import numpy as np
num_cores = multiprocessing.cpu_count()
print("num_cores={}".format(num_cores))


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


if __name__ == "__main__":
    # define parameters
    ntrials, nsamples = 500, 5000
    ntrans = 500
    ntests_li = [1, 2, 3, 4, 5, 10, 20]

    # perform test
    for ntests in ntests_li:
        probs = simulation(ntrials, nsamples, ntests, ntrans)
        rprob_naive, rprob_winners, rprob_rd = probs
        with open("output_small.txt", "a") as f:
            print(f"estimated mu, ntrials={ntrials}, sample size={nsamples}, ntests={ntests}, ntrans={ntrans}: ",
                  probs, file=f)
