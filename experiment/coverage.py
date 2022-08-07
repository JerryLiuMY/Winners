from matplotlib import pyplot as plt
from data_prep.data_prep import data_prep
from models.winners import Winners
from models.naive import Naive
from datetime import datetime
import seaborn as sns
import numpy as np
sns.set()


def find_coverage(model_name, ntrials, narms, diff):
    """ RUn experiment
    :param model_name: Model name
    :param ntrials: Number of trials
    :param narms: number of treatment
    :param diff: difference between winning arm and the remaining arms
    :return: coverage rate
    """

    # generate data
    tol = 1e-5
    Y_all, sigma = data_prep(ntrials, narms, diff)
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


def plot_coverage(model_name, ntrials):
    """ Plot the calculated coverage rate
    :param model_name: Model name
    :param ntrials: Number of trials
    :return:
    """

    # define parameters
    diff_li = list(np.arange(0, 8 + 0.5, 0.5))
    narms_li = [2, 10, 50]

    # get coverage rate
    coverage_arr = np.empty(shape=(len(diff_li), len(narms_li)))
    for i, diff in enumerate(diff_li):
        for j, narms in enumerate(narms_li):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                  f"Working on difference = {diff} and number of arms = {narms}")
            coverage_arr[i, j] = find_coverage(model_name, ntrials, narms, diff)
    coverage_arr = np.round(coverage_arr, 2)

    # plot coverage rate
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(coverage_arr[:, 0], "o-", label="narms=2")
    ax.plot(coverage_arr[:, 1], "v-", label="narms=10")
    ax.plot(coverage_arr[:, 2], "*-", label="narms=50")
    ax.set_xticks(np.arange(len(diff_li)))
    ax.set_xticklabels([val if idx % 2 == 0 else "" for idx, val in enumerate(diff_li)])
    ax.set_xlabel("Difference")
    ax.set_ylabel("coverage probability")
    ax.set_title(f"Coverage probability of Conventional 95% CIs ({model_name})")
    ax.legend(loc="lower right")

    return fig
