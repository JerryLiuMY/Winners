from models.base import Base
from scipy.stats import norm
import numpy as np


class Naive(Base):

    def __init__(self, Y, sigma):
        super().__init__(Y, sigma)

    def search_mu(self, alpha, *args, **kwargs):
        """ Search for mu given an alpha value
        :param alpha: alpha value
        :return: mu value corresponding to the alpha
        """

        mu_alpha = self.ytilde + norm.ppf(q=1-alpha, loc=0, scale=self.sigmaytilde)

        return mu_alpha

    def find_power(self, null):
        """ Find power of the method
        :param null: null hypothesis
        :return:
        """

        power = np.abs(1 - 2 * norm.cdf(self.ytilde, loc=null, scale=self.sigmaytilde))

        return power
