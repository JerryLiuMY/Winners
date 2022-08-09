from experiment.simulation import simulation
import numpy as np
from experiment.power import plot_power
from experiment.coverage import plot_coverage
import os
import json
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


def run_simulation():

    # define parameters
    ntrials, nsamples = 500, 5000
    ntrans = 500
    ntests_li = [1, 2, 3, 4, 5, 10, 20]

    # perform test
    for ntests in ntests_li:
        # save params
        ntests_path = os.path.join(results_path, str(ntests))
        os.mkdir(ntests_path)
        params_dict = {"ntrials": ntrials, "nsamples": nsamples, "ntests": ntests, "ntrans": ntrans}
        with open(os.path.join(ntests_path, "params.json"), "w") as f:
            json.dump(params_dict, f)

        # save results
        rprobs = simulation(ntrials, nsamples, ntests, ntrans)
        rprob_naive, rprob_winners, rprob_rd = rprobs
        rprobs_dict = {"rprob_naive": rprob_naive, "rprob_winners": rprob_winners, "rprob_rd": rprob_rd}
        with open(os.path.join(ntests_path, "rprobs.json"), "w") as f:
            json.dump(rprobs_dict, f)
