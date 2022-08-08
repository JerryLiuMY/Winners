from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from data_prep.dgp import DGP
from models.rd import RD
num_cores = multiprocessing.cpu_count()
print("num_cores={}".format(num_cores))


def simulation(ntrials, sample_size, ntests, ntrans):
    def process():
        dgp = DGP(sample_size, 5, np.arange(5) - 4, np.ones(5))
        Y, Z = dgp.get_data()
        rd = RD(Y, Z, int(len(Z) / 2))
        pvalue = rd.multiple_test(ntests, ntrans)

        return pvalue

    pvalues = Parallel(n_jobs=num_cores)(delayed(process)() for _ in range(ntrials))
    pvalues = np.array(pvalues)

    return pvalues


ntrials = 500
sample_size = 5000
ntrans = 500
num_tests = [1, 2, 3, 4, 5, 10, 20]
for ntests in num_tests:
    pvalues = simulation(ntrials, sample_size, ntests, ntrans)
    reject_prob = np.mean(pvalues <= 0.05)
    with open("output_small.txt", "a") as f:
        print(f"estimated mu, ntrials={ntrials}, sample size={sample_size}, ntests={ntests}, ntrans={ntrans}: ",
              reject_prob, file=f)
