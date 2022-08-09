from experiment.simulation import simulation
from experiment.power import plot_power
from experiment.coverage import plot_coverage
import pickle5 as pickle
import numpy as np
import os
results_path = "./__results__"


def run_coverage(model_name, ntrials):
    """ Plot the calculated power
    :param model_name: model name
    :param ntrials: number of trials
    :return:
    """

    # define path and parameters
    narms_li = [2, 10, 50]
    nsamples_li = [_ * 50 for _ in narms_li]
    mu_max_li = list(np.arange(0, 8 + 0.5, 0.5))
    coverage_path = os.path.join(results_path, "coverage")
    if not os.path.isdir(coverage_path):
        os.mkdir(coverage_path)

    # save figure
    fig = plot_coverage(model_name, ntrials, nsamples_li, narms_li, mu_max_li)
    fig.savefig(os.path.join(coverage_path, f"{model_name.lower()}_coverage.pdf"), bbox_inches="tight")


def run_power(model_name, ntrials):
    """ Plot the calculated power
    :param model_name: model name
    :param ntrials: number of trials
    :return:
    """

    # define parameters
    narms_li = [2, 10, 50]
    nsamples_li = [_ * 50 for _ in narms_li]
    mu_max_li = list(np.arange(0, 4 + 0.1, 0.1))
    power_path = os.path.join(results_path, "power")
    if not os.path.isdir(power_path):
        os.mkdir(power_path)

    # save figure
    fig = plot_power(model_name, ntrials, nsamples_li, narms_li, mu_max_li)
    fig.savefig(os.path.join(power_path, f"{model_name.lower()}_power.pdf"), bbox_inches="tight")


def run_simulation(ntrials):
    """ Run simulation for comparing powers between methods
    :param ntrials: number of trials
    :return:
    """

    # define parameters
    nsamples, narms = 5000, 5
    mu = (np.arange(narms) - 3) / 10
    cov = np.ones(narms)
    ntests_li = [1, 2, 3, 4, 5, 10, 20]
    ntrans = 500

    # perform test
    for ntests in ntests_li:
        # define paths
        ntests_path = os.path.join(results_path, f"ntests_{ntests}_ntrans_{ntrans}")
        if not os.path.isdir(ntests_path):
            os.mkdir(ntests_path)

        # save params
        params_dict1 = {"ntrials": ntrials, "nsamples": nsamples, "narms": narms, "mu": mu, "cov": cov}
        params_dict2 = {"ntests": ntests, "ntrans": ntrans}
        params_dict = {**params_dict1, **params_dict2}
        with open(os.path.join(ntests_path, "params.pkl"), "wb") as handle:
            pickle.dump(params_dict, handle, protocol=4)

        # save results
        rprobs = simulation(ntrials, nsamples, narms, mu, cov, ntests, ntrans)
        rprob_naive, rprob_winners, rprob_rd = rprobs
        rprobs_dict = {"rprob_naive": rprob_naive, "rprob_winners": rprob_winners, "rprob_rd": rprob_rd}
        with open(os.path.join(ntests_path, "rprobs.pkl"), "wb") as handle:
            pickle.dump(rprobs_dict, handle, protocol=4)
