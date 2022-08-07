from models.base import Base
from scipy.stats import norm


class Naive(Base):

    def __init__(self, Y, sigma):
        super().__init__(Y, sigma)

    def search_mu(self, alpha):
        """ Naive method
        :param alpha: alpha value
        :return: mu value corresponding to the alpha
        """

        mu_alpha = self.ytilde + norm.ppf(q=alpha, loc=0, scale=self.sigmaytilde)

        return mu_alpha
