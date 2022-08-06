import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


def naive(ntreat, diff):
    """ Naive method
    :param ntreat: number of treatment
    :param diff: difference between winning arm and the remaining arms
    :return:
    """

    mu = np.zeros(ntreat)
    mu[0] = mu[0] + diff
    cov = np.eye(ntreat)
    Y = np.random.multivariate_normal(mu, cov, size=100000)
    coverage = np.mean([(diff > np.max(Y, axis=1) - 1.96) & (diff < np.max(Y, axis=1) + 1.96)])

    return coverage


def plot_naive():
    """ Plot the naive method for calculating coverage rate
    :return:
    """

    # define parameters
    ntreats = np.array([2, 10, 50])
    diffs = np.arange(0, 8 + 0.5, 0.5)

    # get coverage rate
    coverage_arr = np.empty(shape=(len(ntreats), len(diffs)))
    for i, ntreat in enumerate(ntreats):
        for j, diff in enumerate(diffs):
            coverage_arr[i, j] = naive(ntreat, diff)
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
    ax.set_title("Unconditional coverage probability of Conventional 95% CIs")
    ax.legend()

    return fig
