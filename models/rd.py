import numpy as np
from scipy.stats import norm


class RD(object):

    def __init__(self, Y, T, b, null=0):
        self.n = len(Y)
        self.Y = Y
        self.T = T
        self.b = b
        self.k = len(set(T))
        self.null = null
        if set(T) != set(np.arange(self.k)):
            raise ValueError("Wrong T.")

    def sample_splitting(self, Y, T, row_idx1):
        idx = np.arange(self.n)
        # row_idx1 = np.random.choice(idx, self.b, replace=False)
        row_idx2 = np.array(list(set(idx) - set(row_idx1)))
        Y1, T1, Y2, T2 = Y[row_idx1], T[row_idx1], Y[row_idx2], T[row_idx2]
        best_arm = self.get_best_arm(Y1, T1)
        mean_best = np.mean(Y2[T2 == best_arm])
        var_best = np.var(Y2[T2 == best_arm])
        se = np.sqrt(var_best / len(Y2[T2 == best_arm]))
        t = mean_best / se
        cf = (mean_best - norm.ppf(0.975) * se, mean_best + norm.ppf(0.975) * se)
        pval = (1 - norm.cdf(np.abs(t))) * 2
        return mean_best, se, t, cf, pval

    def get_best_arm(self, Y, T):
        arms = list(set(T))
        best_arm = arms[0]
        best_mean = -np.inf
        for i in arms:
            mean = np.mean(Y[T == i])
            if mean > best_mean:
                best_arm = i
                best_mean = mean
        return best_arm

    def get_residual(self):
        mu_params = np.zeros(self.k)
        for i in range(self.k):
            mu_params[i] = np.mean(self.Y[self.T == i])
            # mu_params[i] = (np.arange(5)-4)[i]
        mu_params[np.argmax(mu_params)] = self.null
        mu = mu_params[self.T]
        return self.Y - mu, mu_params

    def single_test(self):
        idx = np.arange(self.n)
        row_idx1 = np.random.choice(idx, self.b, replace=False)
        return self.sample_splitting(self.Y, self.T, row_idx1)[-1]

    def multiple_test(self, ntests, ntrans):
        idx = np.arange(self.n)
        row_idx1s = [np.random.choice(idx, self.b, replace=False) for i in range(ntests)]
        pvals = np.zeros(ntrans)
        eps, mu = self.get_residual()
        # get p_obs
        p_obs = np.mean([self.sample_splitting(self.Y, self.T, ridx)[-1] for ridx in row_idx1s])
        for i in range(ntrans):
            # permute T
            T = self.T.copy()
            np.random.shuffle(T)
            # residual randomization
            g = np.random.choice([0, 1], size=len(T))
            Y = mu[T] + g * eps
            ps = [self.sample_splitting(Y, T, ridx)[-1] for ridx in row_idx1s]
            p = np.mean(ps)
            pvals[i] = p
        pvalue = np.mean(pvals < p_obs)
        return pvalue
