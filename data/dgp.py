import numpy as np


class DGP(object):

    def __init__(self, nsamples, narms, means, vars):
        self.n = nsamples
        self.k = narms
        self.means = means
        self.vars = vars
        if self.verify_inputs() is False:
            raise ValueError("Incorrect inputs.")

    def verify_inputs(self):
        if len(self.means) != self.k or len(self.vars) != self.k:
            return False
        if self.n % self.k != 0:
            return False
        return True

    def get_potentials(self):
        Y = np.zeros((self.n, self.k))
        for i in range(self.k):
            Y[:,i] = np.random.normal(self.means[i], self.vars[i], size=self.n)
        return Y

    def get_treatment(self):
        Z = []
        for i in range(self.k):
            Z += [i]*int(self.n/self.k)
        Z = np.array(Z)
        np.random.shuffle(Z)
        return Z

    def get_data(self):
        Yp = self.get_potentials()
        Z = self.get_treatment()
        Y = np.zeros(self.n)
        for i in range(self.k):
            Y[Z==i] = Yp[Z==i,i]
        return Y, Z
