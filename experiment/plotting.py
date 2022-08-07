from experiment.experiment import experiment
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


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
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on number of treatment {ntreat}")
        for j, diff in enumerate(diffs):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on difference {diff}")
            coverage_arr[i, j] = experiment(model_name, ntreat, diff, nsample)
    coverage_arr = np.round(coverage_arr, 2)

    # plot coverage rate
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(coverage_arr[0, :], "o-", label="ntreat=2")
    ax.plot(coverage_arr[1, :], "v-", label="ntreat=10")
    ax.plot(coverage_arr[2, :], "*-", label="ntreat=50")
    ax.set_xticks(np.arange(len(diffs)))
    ax.set_xticklabels([val if idx % 2 == 0 else "" for idx, val in enumerate(diffs)])
    ax.set_xlabel("Difference")
    ax.set_ylabel("coverage probability")
    ax.set_title(f"Coverage probability of Conventional 95% CIs ({model_name})")
    ax.legend()

    return fig
