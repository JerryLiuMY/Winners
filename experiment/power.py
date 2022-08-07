from matplotlib import pyplot as plt
from data_prep.data_prep import data_prep
from models.winners import Winners
from models.naive import Naive
from datetime import datetime
import seaborn as sns
import numpy as np
sns.set()


def find_power(model_name, nsamples, narms, diff, null_li):
    """ RUn experiment
    :param model_name: Model name
    :param nsamples: Number of samples
    :param narms: number of treatment
    :param diff: difference between winning arm and the remaining arms
    :param null_li: List of null hypothesis
    :return: coverage rate
    """

    # generate data
    Y_all, sigma = data_prep(nsamples, narms, diff)
    model_dict = {"Naive": Naive, "Winners": Winners}
    Model = model_dict[model_name]

    # find coverage rate
    power_li = []
    for idx, null in enumerate(null_li):
        # logging massage
        if idx % 100 == 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Working on null hypothesis {idx}")

        # find power corresponding to the null
        power_li_sub = []
        for Y in Y_all:
            model = Model(Y, sigma)
            power_li_sub.append(model.get_power(null=null))
        power_li.append(np.mean(power_li_sub))

    return power_li


def plot_power(model_name, nsamples):
    """ Plot the calculated coverage rate
    :param model_name: Model name
    :param nsamples: Number of samples
    :return:
    """

    # define parameters
    ntreats_li = [2, 10, 50]
    diff_li = [1, 2, 3, 4]

    # get power
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i, diff in enumerate(diff_li):
        ax = axes[i // 2, i % 2]
        null_li = np.linspace(diff - 5, diff + 5, 101)
        for j, ntreat in enumerate(ntreats_li):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                  f"Working on difference = {diff} and number of treatment = {ntreat}")
            power_li = find_power(model_name, nsamples, ntreat, diff, null_li)
            ax.plot(power_li, "-", label=f"ntreat={ntreat}")

        ax.set_xticks([val for idx, val in enumerate(np.arange(len(null_li))) if idx % 10 == 0])
        ax.set_xticklabels([round((val - diff)) for idx, val in enumerate(null_li) if idx % 10 == 0])
        ax.set_xlabel("null - true")
        ax.set_ylabel("Power")
        ax.set_title(f"Power of the method for difference = {diff}")
        ax.legend(loc="lower right")

    return fig
