import numpy as np
from models.winners import Winners
from data_prep.data_prep import data_prep


def experiment(ntreat, diff):
    """ RUn experiment
    :param ntreat: number of treatment
    :param diff: difference between winning arm and the remaining arms
    :return: coverage rate
    """

    # generate data
    tol = 1e-5
    Y_all, sigma = data_prep(ntreat, diff)

    # find coverage rate
    coverage = []
    sample = 0
    for Y in Y_all:
        # logging massage
        print(f"Working on sample {sample}")
        sample = sample + 1

        # logging massage
        winners = Winners(Y, sigma)
        ltilde, utilde = winners.get_truncation()
        mu_lower = winners.search_mu(ltilde, utilde, alpha=1-0.025, tol=tol)
        mu_upper = winners.search_mu(ltilde, utilde, alpha=0.025, tol=tol)

        # append coverage rate
        coverage.append((diff > mu_lower) & (diff < mu_upper))

    return np.mean(coverage)
