import numpy as np


class Base(object):

    def __init__(self, Y, sigma):
        self.Y = Y
        self.sigma = sigma
        self.k = len(Y)

        # index of the winning arm
        self.theta_tilde = np.argmax(self.Y)
        # estimate associated with the winning arm
        self.ytilde = Y[self.theta_tilde]
        # variance of all the estimates
        self.sigmaytilde = self.sigma[self.theta_tilde, self.theta_tilde]
        # covariance of the winning arm and other arms
        self.sigmaytilde_vec = np.array(self.sigma[self.theta_tilde, 0: self.k])
        # normalised difference
        self.ztilde = np.array(Y) - (self.sigma[self.theta_tilde, 0: self.k]) / self.sigmaytilde * self.ytilde
