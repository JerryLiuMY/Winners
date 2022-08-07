from matplotlib import pyplot as plt
from data_prep.data_prep import data_prep
from models.winners import Winners
from models.naive import Naive
from datetime import datetime
import seaborn as sns
import numpy as np
sns.set()


def find_coverage(model_name, nsamples, narms, diff):
    """ RUn experiment
    :param model_name: Model name
    :param nsamples: Number of samples
    :param narms: number of treatment
    :param diff: difference between winning arm and the remaining arms
    :return: coverage rate
    """

    # generate data
    tol = 1e-5
    Y_all, sigma = data_prep(nsamples, narms, diff)
    model_dict = {"Naive": Naive, "Winners": Winners}
    Model = model_dict[model_name]

    # find coverage rate
    coverage_li = []
    for idx, Y in enumerate(Y_all):
        # logging massage
        if idx % 100 == 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on sample {idx}")

        # logging massage
        model = Model(Y, sigma)
        mu_lower = model.search_mu(alpha=0.025, tol=tol)
        mu_upper = model.search_mu(alpha=1-0.025, tol=tol)

        # append coverage rate
        coverage_li.append((diff > mu_upper) & (diff < mu_lower))

    return np.mean(coverage_li)


def plot_coverage(model_name, nsamples):
    """ Plot the calculated coverage rate
    :param model_name: Model name
    :param nsamples: Number of samples
    :return:
    """

    # define parameters
    ntreats_li = [2, 10, 50]
    diff_li = list(np.arange(0, 8 + 0.5, 0.5))

    # get coverage rate
    coverage_arr = np.empty(shape=(len(ntreats_li), len(diff_li)))
    for i, ntreats in enumerate(ntreats_li):
        for j, diff in enumerate(diff_li):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                  f"Working on number of treatment = {ntreats} and difference = {diff}")
            coverage_arr[i, j] = find_coverage(model_name, nsamples, ntreats, diff)
    coverage_arr = np.round(coverage_arr, 2)

    # plot coverage rate
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(coverage_arr[0, :], "o-", label="ntreats=2")
    ax.plot(coverage_arr[1, :], "v-", label="ntreats=10")
    ax.plot(coverage_arr[2, :], "*-", label="ntreats=50")
    ax.set_xticks(np.arange(len(diff_li)))
    ax.set_xticklabels([val if idx % 2 == 0 else "" for idx, val in enumerate(diff_li)])
    ax.set_xlabel("Difference")
    ax.set_ylabel("coverage probability")
    ax.set_title(f"Coverage probability of Conventional 95% CIs ({model_name})")
    ax.legend(loc="lower right")

    return fig
