import numpy as np


class DGP(object):

    def __init__(self, nsamples, narms, mu, cov):
        self.n = nsamples
        self.k = narms
        self.mu = mu
        self.cov = cov
        if self.verify_inputs() is False:
            raise ValueError("Incorrect inputs.")

    def verify_inputs(self):
        if len(self.mu) != self.k or len(self.cov) != self.k:
            return False
        if self.n % self.k != 0:
            return False
        return True

    def get_potentials(self):
        Y = np.zeros((self.n, self.k))
        for i in range(self.k):
            Y[:, i] = np.random.normal(self.mu[i], self.cov[i], size=self.n)

        return Y

    def get_treatment(self):
        T = []
        for i in range(self.k):
            T += [i] * int(self.n / self.k)
        T = np.array(T)
        np.random.shuffle(T)

        return T

    def get_data(self):
        Yp = self.get_potentials()
        T = self.get_treatment()
        Y = np.zeros(self.n)
        for i in range(self.k):
            Y[T == i] = Yp[T == i, i]

        return Y, T

    def get_input(self):
        Y, T = self.get_data()
        Y = np.array([np.mean(Y[T == t]) for t in sorted(set(T))])
        sigma = np.diag(self.cov / np.sqrt(self.n // self.k))

        return Y, sigma
