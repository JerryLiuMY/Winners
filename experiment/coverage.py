from matplotlib import pyplot as plt
from data_prep.dgp import DGP
from models.winners import Winners
from models.naive import Naive
from datetime import datetime
import seaborn as sns
import numpy as np
sns.set()


def find_coverage(model_name, ntrials, nsamples, narms, mu, cov):
    """ RUn experiment
    :param model_name: Model name
    :param ntrials: Number of trials
    :param nsamples: Number of samples
    :param narms: number of treatment
    :param mu: mean of the data generation
    :param cov: covariance of the data generation
    :return: coverage rate
    """

    # define parameters
    tol = 1e-5
    model_dict = {"Naive": Naive, "Winners": Winners}
    Model = model_dict[model_name]

    # find coverage rate
    coverage_li = []
    for idx in range(ntrials):
        if idx % 100 == 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on trial {idx}")

        # search for mu
        Y, sigma = DGP(nsamples, narms, mu, cov).get_input()
        model = Model(Y, sigma)
        mu_lower = model.search_mu(alpha=0.025, tol=tol)
        mu_upper = model.search_mu(alpha=1-0.025, tol=tol)

        # append coverage rate
        coverage_li.append((np.max(mu) > mu_upper) & (np.max(mu) < mu_lower))

    return np.mean(coverage_li)


def plot_coverage(model_name, ntrials):
    """ Plot the calculated coverage rate
    :param model_name: Model name
    :param ntrials: Number of trials
    :return:
    """

    # define parameters
    narms_li = [2, 10, 50]
    nsamples_li = [_ * 50 for _ in narms_li]
    mu_max_li = list(np.arange(0, 8 + 0.5, 0.5))

    # get coverage rate
    coverage_arr = np.empty(shape=(len(narms_li), len(mu_max_li)))
    for i, (nsamples, narms) in enumerate(zip(nsamples_li, narms_li)):
        for j, mu_max in enumerate(mu_max_li):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                  f"Working on number of arms = {narms} and mu_max = {mu_max} ")
            mu, cov = np.array([mu_max] + [0] * (narms-1)), np.ones(narms)
            coverage_arr[i, j] = find_coverage(model_name, ntrials, nsamples, narms, mu, cov)
    coverage_arr = np.round(coverage_arr, 2)

    # plot coverage rate
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(coverage_arr[0, :], "o-", label="narms=2")
    ax.plot(coverage_arr[1, :], "v-", label="narms=10")
    ax.plot(coverage_arr[2, :], "*-", label="narms=50")
    ax.set_xticks(np.arange(len(mu_max_li)))
    ax.set_xticklabels([val if idx % 2 == 0 else "" for idx, val in enumerate(mu_max_li)])
    ax.set_xlabel("Mean of best arm")
    ax.set_ylabel("Coverage probability")
    ax.set_title(f"Coverage probability of Conventional 95% CIs ({model_name})")
    ax.legend(loc="lower right")

    return fig
