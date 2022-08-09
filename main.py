from experiment.simulation import simulation
import multiprocessing
import os
import json
num_cores = multiprocessing.cpu_count()
print("num_cores={}".format(num_cores))


def main():
    # define parameters
    ntrials, nsamples = 500, 5000
    ntrans = 500
    ntests_li = [1, 2, 3, 4, 5, 10, 20]
    path = "./results"

    # perform test
    for ntests in ntests_li:

        # define path
        ntests_path = os.path.join(path, str(ntests))
        os.mkdir(ntests_path)

        # save params
        params_dict = {"ntrials": ntrials, "nsamples": nsamples, "ntests": ntests, "ntrans": ntrans}
        with open(os.path.join(ntests_path, "params.json"), "w") as f:
            json.dump(params_dict, f)

        # save results
        rprobs = simulation(ntrials, nsamples, ntests, ntrans)
        rprob_naive, rprob_winners, rprob_rd = rprobs
        rprobs_dict = {"rprob_naive": rprob_naive, "rprob_winners": rprob_winners, "rprob_rd": rprob_rd}
        with open(os.path.join(ntests_path, "rprobs.json"), "w") as f:
            json.dump(rprobs_dict, f)
