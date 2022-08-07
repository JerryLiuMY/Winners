from matplotlib import pyplot as plt
from data_prep.data_prep import data_prep
from models.winners import Winners
from models.naive import Naive
from datetime import datetime
import seaborn as sns
import numpy as np
sns.set()


def find_coverage(model_name, ntreat, diff, nsample):
    """ RUn experiment
    :param model_name: Model name
    :param ntreat: number of treatment
    :param diff: difference between winning arm and the remaining arms
    :param nsample: Number of samples
    :return: coverage rate
    """

    # generate data
    tol = 1e-5
    Y_all, sigma = data_prep(ntreat, diff, nsample)
    model_dict = {"Naive": Naive, "Winners": Winners}
    Model = model_dict[model_name]

    # find coverage rate
    coverage = []
    for idx, Y in enumerate(Y_all):
        # logging massage
        if idx % 100 == 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on sample {idx}")

        # logging massage
        model = Model(Y, sigma)
        mu_lower = model.search_mu(alpha=0.025, tol=tol)
        mu_upper = model.search_mu(alpha=1-0.025, tol=tol)

        # append coverage rate
        coverage.append((diff > mu_upper) & (diff < mu_lower))

    return np.mean(coverage)


def plot_coverage(model_name, nsample):
    """ Plot the naive method for calculating coverage rate
    :param model_name: Model name
    :param nsample: Number of samples
    :return:
    """

    # define parameters
    ntreats = np.array([2, 10, 50])
    diffs = np.arange(0, 8 + 0.5, 0.5)

    # get coverage rate
    coverage_arr = np.empty(shape=(len(ntreats), len(diffs)))
    for i, ntreat in enumerate(ntreats):
        for j, diff in enumerate(diffs):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                  f"Working on number of treatment {ntreat} and difference {diff}")
            coverage_arr[i, j] = find_coverage(model_name, ntreat, diff, nsample)
    coverage_arr = np.round(coverage_arr, 2)

    # plot coverage rate
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(coverage_arr[0, :], "o-", label="ntreat=2")
    ax.plot(coverage_arr[1, :], "v-", label="ntreat=10")
    ax.plot(coverage_arr[2, :], "*-", label="ntreat=50")
    ax.set_xticks(np.arange(len(diffs)))
    ax.set_xticklabels([val if idx % 2 == 0 else "" for idx, val in enumerate(diffs)])
    ax.set_xlabel("Difference")
    ax.set_ylabel("coverage probability")
    ax.set_title(f"Coverage probability of Conventional 95% CIs ({model_name})")
    ax.legend(loc="lower right")

    return fig
