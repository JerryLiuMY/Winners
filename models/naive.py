from models.base import Base
from scipy.stats import norm
import numpy as np


class Naive(Base):

    def __init__(self, Y_mu, sigma):
        super().__init__(Y_mu, sigma)

    def search_mu(self, alpha, *args, **kwargs):
        """ Search for mu given an alpha value
        :param alpha: alpha value
        :return: mu value corresponding to the alpha
        """

        stdytilde = np.sqrt(self.sigmaytilde)
        mu_alpha = self.ytilde + norm.ppf(q=1-alpha, loc=0, scale=stdytilde)

        return mu_alpha

    def get_test(self, null):
        """ Hypothesis test for the method
        :param null: null hypothesis
        :return:
        """

        # define parameters
        yhat = self.ytilde
        stdytilde = np.sqrt(self.sigmaytilde)

        # find pvalue
        pval = 1 - np.abs(1 - 2 * norm.cdf(x=yhat, loc=null, scale=stdytilde))

        return pval
