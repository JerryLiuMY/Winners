from joblib import Parallel, delayed
import multiprocessing
from matplotlib import pyplot as plt
from data_prep.dgp import DGP
from models.winners import Winners
from models.naive import Naive
from models.rd import RD
from datetime import datetime
import seaborn as sns
import numpy as np
sns.set()


def find_power(model_name, ntrials, narms, nsamples, mu, cov, **kwargs):
    """ Run experiment
    :param model_name: model name
    :param ntrials: number of trials
    :param nsamples: number of samples
    :param narms: number of treatment
    :param mu: mean of the data generation
    :param cov: covariance of the data generation
    :return: power
    """

    # find power
    num_cores = multiprocessing.cpu_count()
    parallel = Parallel(n_jobs=num_cores)
    pvals = parallel(delayed(power_process)(model_name, narms, nsamples, mu, cov, **kwargs) for _ in range(ntrials))
    power = np.mean(np.array(pvals) <= 0.05)

    return power


def power_process(model_name, narms, nsamples, mu, cov, **kwargs):
    """ Single process of finding power
    :param model_name: model name
    :param nsamples: number of samples
    :param narms: number of treatment
    :param mu: mean of the data generation
    :param cov: covariance of the data generation
    """

    # define parameters
    model_dict = {"Naive": Naive, "Winners": Winners, "RD": RD}
    Model = model_dict[model_name]

    # generate data
    Y, T = DGP(narms, nsamples, mu, cov).get_data()
    Y_mu, sigma = DGP(narms, nsamples, mu, cov).get_input()

    # find pval
    if model_name == "RD":
        ntests, ntrans = kwargs["ntests"], kwargs["ntrans"]
        model = Model(Y, T, b=int(len(T) / 2), null=0)
        pval = model.multiple_test(ntests, ntrans)
    else:
        model = Model(Y_mu, sigma)
        pval = model.get_test(null=0)

    return pval


def plot_power(model_name, ntrials, narms_li, nsamples_li, mu_max_li):
    """ Plot the calculated power
    :param model_name: model name
    :param ntrials: number of trials
    :param nsamples_li: number of samples list
    :param narms_li: number of treatment list
    :param mu_max_li: mean of the best arm list
    :return:
    """

    # get power
    power_arr = np.empty(shape=(len(narms_li), len(mu_max_li)))
    for i, (nsamples, narms) in enumerate(zip(nsamples_li, narms_li)):
        for j, mu_max in enumerate(mu_max_li):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                  f"Working on number of arms = {narms} and mu_max = {round(mu_max, 2)} ")
            mu, cov = np.array([mu_max] + [0] * (narms-1)), np.ones(narms)
            if model_name == "RD":
                power_arr[i, j] = find_power(model_name, ntrials, narms, nsamples, mu, cov, ntests=5, ntrans=500)
            else:
                power_arr[i, j] = find_power(model_name, ntrials, narms, nsamples, mu, cov)
    power_arr = np.round(power_arr, 2)

    # plot power
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(power_arr[0, :], "o-", label="narms=2")
    ax.plot(power_arr[1, :], "v-", label="narms=10")
    ax.plot(power_arr[2, :], "*-", label="narms=50")
    ax.set_xticks([idx for idx, val in enumerate(mu_max_li) if idx % 5 == 0])
    ax.set_xticklabels([val for idx, val in enumerate(mu_max_li) if idx % 5 == 0])
    ax.set_xlabel("Mean of best arm")
    ax.set_ylabel("Power")
    ax.set_title(f"Power of the method with null = 0 ({model_name})")
    ax.legend(loc="lower right")

    return fig
