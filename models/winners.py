from models.base import Base
from scipy.stats import truncnorm
import numpy as np
np.finfo(np.double).precision = 100


class Winners(Base):

    def __init__(self, Y, sigma):
        super().__init__(Y, sigma)

    def get_truncation(self):
        """ Get the truncation threshold for the truncated normal distribution
        :return ltilde: lower truncation threshold
        :return utilde: upper truncation threshold
        """

        # The lower truncation value
        ind_l = np.array(self.sigmaytilde > self.sigmaytilde_vec)
        if sum(ind_l) == 0:
            ltilde = -np.inf
        elif sum(ind_l) > 0:
            ltilde = max(self.sigmaytilde * (self.ztilde[ind_l] - self.ztilde[self.theta_tilde]) /
                         (self.sigmaytilde - self.sigmaytilde_vec[ind_l]))
        else:
            raise ValueError("Invalid ind_l value")

        # The upper truncation value
        ind_u = np.array(self.sigmaytilde < self.sigmaytilde_vec)
        if sum(ind_u) == 0:
            utilde = +np.inf
        elif sum(ind_u) > 0:
            utilde = min(self.sigmaytilde * (self.ztilde[ind_u] - self.ztilde[self.theta_tilde]) /
                         (self.sigmaytilde - self.sigmaytilde_vec[ind_u]))
        else:
            raise ValueError("Invalid ind_u value")

        # The V truncation value
        ind_v = np.array(self.sigmaytilde_vec == self.sigmaytilde)
        if sum(ind_v) == 0:
            vtilde = 0
        elif sum(ind_v) > 0:
            vtilde = min(-(self.ztilde[ind_v] - self.ztilde[self.theta_tilde]))
        else:
            raise ValueError("Invalid ind_v value")

        if vtilde < 0:
            return None

        return ltilde, utilde

    def search_mu(self, alpha, tol):
        """ search for mu for the given alpha
        :param alpha: alpha value
        :param tol: tolerance
        :return: mu value corresponding to the alpha
        """

        # define parameters
        yhat = self.ytilde
        stdytilde = np.sqrt(self.sigmaytilde)
        ltilde, utilde = self.get_truncation()

        # search loop
        std_temp = stdytilde
        upper_limit = yhat - std_temp
        lower_limit = yhat + std_temp
        upper_a, upper_b = (ltilde - upper_limit) / stdytilde, (utilde - upper_limit) / stdytilde
        lower_a, lower_b = (ltilde - lower_limit) / stdytilde, (utilde - lower_limit) / stdytilde
        upper_quantile = truncnorm.cdf(x=yhat, a=upper_a, b=upper_b, loc=upper_limit, scale=stdytilde)
        lower_quantile = truncnorm.cdf(x=yhat, a=lower_a, b=lower_b, loc=lower_limit, scale=stdytilde)

        rloop = 1
        while not ((alpha > lower_quantile) and (alpha < upper_quantile)):
            std_temp = std_temp * 1.05
            upper_limit = yhat - std_temp
            lower_limit = yhat + std_temp
            upper_a, upper_b = (ltilde - upper_limit) / stdytilde, (utilde - upper_limit) / stdytilde
            lower_a, lower_b = (ltilde - lower_limit) / stdytilde, (utilde - lower_limit) / stdytilde
            upper_quantile = truncnorm.cdf(x=yhat, a=upper_a, b=upper_b, loc=upper_limit, scale=stdytilde)
            lower_quantile = truncnorm.cdf(x=yhat, a=lower_a, b=lower_b, loc=lower_limit, scale=stdytilde)
            rloop += 1

        # print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Range loop ({rloop} iterations) [alpha={alpha}] - "
        #       f"lower_limit = {round(lower_limit, 2)} (q={round(lower_quantile, 3)}) and "
        #       f"upper_limit = {round(upper_limit, 2)} (q={round(upper_quantile, 3)})")

        # bisection loop
        middle_limit = (lower_limit + upper_limit) / 2
        middle_a, middle_b = (ltilde - middle_limit) / stdytilde, (utilde - middle_limit) / stdytilde
        middle_quantile = truncnorm.cdf(x=yhat, a=middle_a, b=middle_b, loc=middle_limit, scale=stdytilde)

        bloop = 1
        while np.abs(middle_quantile - alpha) > tol and bloop < 100:
            if (alpha > lower_quantile) and (alpha < middle_quantile):
                upper_limit = middle_limit
                upper_a, upper_b = (ltilde - upper_limit) / stdytilde, (utilde - upper_limit) / stdytilde
                upper_quantile = truncnorm.cdf(x=yhat, a=upper_a, b=upper_b, loc=upper_limit, scale=stdytilde)
            elif (alpha > middle_quantile) and (alpha < upper_quantile):
                lower_limit = middle_limit
                lower_a, lower_b = (ltilde - lower_limit) / stdytilde, (utilde - lower_limit) / stdytilde
                lower_quantile = truncnorm.cdf(x=yhat, a=lower_a, b=lower_b, loc=lower_limit, scale=stdytilde)
            else:
                raise ValueError("Floating error")

            middle_limit = (lower_limit + upper_limit) / 2
            middle_a, middle_b = (ltilde - middle_limit) / stdytilde, (utilde - middle_limit) / stdytilde
            middle_quantile = truncnorm.cdf(x=yhat, a=middle_a, b=middle_b, loc=middle_limit, scale=stdytilde)
            bloop += 1

        # print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Bisection loop ({bloop} iterations) [alpha={alpha}] - "
        #       f"middle_limit = {round(middle_limit, 2)} (q={round(middle_quantile, 3)})\n")

        mu_alpha = middle_limit

        return mu_alpha

    def find_power(self, null):
        # define parameters
        yhat = self.ytilde
        stdytilde = np.sqrt(self.sigmaytilde)
        ltilde, utilde = self.get_truncation()

        # find power
        a, b = (ltilde - null) / stdytilde, (utilde - null) / stdytilde
        power = np.abs(1 - 2 * truncnorm.cdf(x=yhat, a=a, b=b, loc=null, scale=stdytilde))

        return power
