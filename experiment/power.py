from matplotlib import pyplot as plt
from data_prep.dgp import DGP
from models.winners import Winners
from models.naive import Naive
from datetime import datetime
import seaborn as sns
import numpy as np
sns.set()


def find_power(model_name, ntrials, nsamples, narms, mu, cov, null_li):
    """ RUn experiment
    :param model_name: Model name
    :param ntrials: Number of trials
    :param nsamples: Number of samples
    :param narms: number of treatment
    :param mu: mean of the data generation
    :param cov: covariance of the data generation
    :param null_li: List of null hypothesis
    :return: coverage rate
    """

    # define parameters
    model_dict = {"Naive": Naive, "Winners": Winners}
    Model = model_dict[model_name]

    # find coverage rate
    power_li = []
    for idx, null in enumerate(null_li):
        if idx % 100 == 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on null hypothesis {idx}")

        # find power
        power_li_sub = []
        for _ in np.range(ntrials):
            Y, sigma = DGP(nsamples, narms, mu, cov).get_input()
            model = Model(Y, sigma)
            power_li_sub.append(model.get_power(null=null))
        power_li.append(np.mean(power_li_sub))

    return power_li


def plot_power(model_name, ntrials):
    """ Plot the calculated coverage rate
    :param model_name: Model name
    :param ntrials: Number of samples
    :return:
    """

    # define parameters
    diff_li = [1, 2, 3, 4]
    narms_li = [2, 10, 50]
    nsamples_li = [_ * 50 for _ in narms_li]

    # plot power
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i, diff in enumerate(diff_li):
        ax = axes[i // 2, i % 2]
        null_li = np.linspace(diff - 5, diff + 5, 101)
        for j, (nsamples, narms) in enumerate(zip(nsamples_li, narms_li)):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                  f"Working on difference = {diff} and number of arms = {narms}")
            mu, cov = np.arange(narms) - (narms - 1), np.ones(narms)
            power_li = find_power(model_name, ntrials, nsamples, narms, mu, cov, null_li)
            ax.plot(power_li, "-", label=f"narms={narms}")

        ax.set_xticks([val for idx, val in enumerate(np.arange(len(null_li))) if idx % 10 == 0])
        ax.set_xticklabels([round((val - diff)) for idx, val in enumerate(null_li) if idx % 10 == 0])
        ax.set_xlabel("null - true")
        ax.set_ylabel("Power")
        ax.set_title(f"Power of the method for difference = {diff}")
        ax.legend(loc="lower right")

    return fig
