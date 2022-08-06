from scipy.stats import norm
import numpy as np


class Naive(object):

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

    def search_mu(self, alpha):
        """ Naive method
        :param alpha: alpha value
        :return: mu value corresponding to the alpha
        """

        mu_alpha = self.ytilde + norm.ppf(q=alpha, loc=0, scale=self.sigmaytilde)

        return mu_alpha
