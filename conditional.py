import numpy as np
from winners import DGP


class Conditional:
    def __init__(self, means, covs, X, T):
        self.means = means
        self.covs = covs
        self.X = X
        self.T = T
        self.narms = len(self.T)
        self.estimate = self.get_estimate()
        self.best_arm = self.get_estimate()
        self.Y, self.Z = self.get_YZ()

    def get_estimate(self):
        """ get a list of estimated means """

        estimate = np.array([X[T == arm].mean() for arm in range(self.narms)])

        return estimate

    def get_best_arm(self):
        """ get the best arm """

        best_arm = np.argmax(self.estimate)  # find the best arm \tilde{\theta}

        return best_arm

    def get_YZ(self):
        """ get Y and Z """

        # find Y and Z
        Y = np.array([self.X[self.T == self.best_arm].mean()])
        sigma_XY = covs[:, [self.best_arm]]
        sigma_YY = covs[self.best_arm, self.best_arm]
        Z = self.estimate - sigma_XY / sigma_YY @ Y

        return Y, Z

    def get_interval(self):
        """ Get interval for truncated normal distribution """
        
        # L_list: get list for lower bound
        L_list = []
        for arm in range(narms):
            if covs[self.best_arm, self.best_arm] > covs[self.best_arm, arm]:
                num = covs[self.best_arm, self.best_arm] * (self.Z[arm] - self.Z[self.best_arm])
                den = covs[self.best_arm, self.best_arm] - covs[self.best_arm, arm]
                L_list.append(num / den)

        if len(L_list) == 0:
            L = -np.inf
        else:
            L = max(L_list)

        # U_list: get list for upper bound
        U_list = []
        for arm in range(narms):
            if covs[self.best_arm, self.best_arm] < covs[self.best_arm, arm]:
                num = covs[self.best_arm, self.best_arm] * (self.Z[arm] - self.Z[self.best_arm])
                den = covs[self.best_arm, self.best_arm] - covs[self.best_arm, arm]
                U_list.append(num / den)

        if len(U_list) == 0:
            U = np.inf
        else:
            U = min(U_list)

        # V_list: get list for condition
        V_list = []
        for arm in range(narms):
            if covs[self.best_arm, self.best_arm] == covs[self.best_arm, arm]:
                V_list.append(-(self.Z[arm] - self.Z[self.best_arm]))

        # Y: get the interval for truncated mean
        if min(V_list) >= 0:
            Y = [L, U]
        else:
            Y = None

        return Y


if __name__ == "__main__":
    np.random.seed(0)
    nsamples = 1000
    narms = 20
    means = np.random.normal(10, 1, size=narms)
    vars = np.ones(narms)
    covs = np.diag(vars)
    dpg = DGP(nsamples, narms, means, vars)
    X, T = dpg.get_data()
    conditional = Conditional(means, covs, X, T)
