from scipy.stats import truncnorm
import numpy as np


class Winners(object):

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

    def get_truncation(self):
        """ Get the truncation threshold for the truncated normal distribution
        :return ltilde: lower truncation threshold
        :return utilde: upper truncation threshold
        """

        # The lower truncation value
        ind_l = self.sigmaytilde > self.sigmaytilde_vec
        if sum(ind_l) == 0:
            ltilde = -np.inf
        elif sum(ind_l) > 0:
            ltilde = max(self.sigmaytilde * (self.ztilde[ind_l] - self.ztilde[self.theta_tilde]) /
                         (self.sigmaytilde - self.sigmaytilde_vec[ind_l]))
        else:
            raise ValueError("Invalid ind_l value")

        # The upper truncation value
        ind_u = self.sigmaytilde < self.sigmaytilde_vec
        if sum(ind_u) == 0:
            utilde = +np.inf
        elif sum(ind_u) > 0:
            utilde = min(self.sigmaytilde * (self.ztilde[ind_u] - self.ztilde[self.theta_tilde]) /
                         (self.sigmaytilde - self.sigmaytilde_vec[ind_u]))
        else:
            raise ValueError("Invalid ind_u value")

        # The V truncation value
        ind_v = (self.sigmaytilde_vec == self.sigmaytilde)
        if sum(ind_v) == 0:
            vtilde = 0
        elif sum(ind_v) > 0:
            vtilde = min(-(self.ztilde[ind_v] - self.ztilde[self.theta_tilde]))
        else:
            raise ValueError("Invalid ind_v value")

        if vtilde < 0:
            return None

        return ltilde, utilde

    def search_mu(self, ltilde, utilde, alpha, tol):
        """ search for mu for the given alpha
        :param ltilde: lower truncation limit
        :param utilde: upper truncation limit
        :param alpha: alpha value
        :param tol: tolerance
        :return: mu value corresponding to the alpha
        """

        yhat = self.ytilde
        sigmayhat = self.sigmaytilde

        # define search range
        sigma_temp = sigmayhat
        lower_limit = yhat + sigma_temp  # lower limit plus
        upper_limit = yhat - sigma_temp  # upper limit minus
        lower_a, lower_b = (ltilde - lower_limit) / sigmayhat, (utilde - lower_limit) / sigmayhat
        upper_a, upper_b = (ltilde - upper_limit) / sigmayhat, (utilde - upper_limit) / sigmayhat
        lower_quantile = truncnorm.cdf(x=yhat, a=lower_a, b=lower_b, loc=lower_limit, scale=sigmayhat)
        upper_quantile = truncnorm.cdf(x=yhat, a=upper_a, b=upper_b, loc=upper_limit, scale=sigmayhat)
        # print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Initial range loop -- "
        #       f"lower_limit={round(lower_limit, 2)} (q={round(lower_quantile, 3)}) and "
        #       f"upper_limit={round(upper_limit, 2)} (q={round(upper_quantile, 3)})")

        range_loop = 1
        if not ((alpha > lower_quantile) and (alpha < upper_quantile)):
            sigma_temp = sigma_temp * 2
            lower_limit = yhat + sigma_temp  # lower limit plus
            upper_limit = yhat - sigma_temp  # upper limit minus
            lower_a, lower_b = (ltilde - lower_limit) / sigmayhat, (utilde - lower_limit) / sigmayhat
            upper_a, upper_b = (ltilde - upper_limit) / sigmayhat, (utilde - upper_limit) / sigmayhat
            lower_quantile = truncnorm.cdf(x=yhat, a=lower_a, b=lower_b, loc=lower_limit, scale=sigmayhat)
            upper_quantile = truncnorm.cdf(x=yhat, a=upper_a, b=upper_b, loc=upper_limit, scale=sigmayhat)
            # print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Range loop {range_loop} -- "
            #       f"lower_limit={round(lower_limit, 2)} (q={round(lower_quantile, 3)}) and "
            #       f"upper_limit={round(upper_limit, 2)} (q={round(upper_quantile, 3)})")
            range_loop += 1

        # bisection search between lower_limit and upper_limit
        middle_limit = (lower_limit + upper_limit) / 2
        middle_a, middle_b = (ltilde - middle_limit) / sigmayhat, (utilde - middle_limit) / sigmayhat
        middle_quantile = truncnorm.cdf(x=yhat, a=middle_a, b=middle_b, loc=middle_limit, scale=sigmayhat)
        # print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Initial bisection loop -- "
        #       f"lower_limit={round(lower_limit, 2)} (q={round(lower_quantile, 3)}) and "
        #       f"upper_limit={round(upper_limit, 2)} (q={round(upper_quantile, 3)})")

        bisection_loop = 1
        while np.abs(middle_quantile - alpha) > tol:
            if (alpha > lower_quantile) and (alpha < middle_quantile):
                upper_limit = middle_limit
                upper_a, upper_b = (ltilde - upper_limit) / sigmayhat, (utilde - upper_limit) / sigmayhat
                upper_quantile = truncnorm.cdf(x=yhat, a=upper_a, b=upper_b, loc=middle_limit, scale=sigmayhat)
            elif (alpha > middle_quantile) and (alpha < upper_quantile):
                lower_limit = middle_limit
                lower_a, lower_b = (ltilde - lower_limit) / sigmayhat, (utilde - lower_limit) / sigmayhat
                lower_quantile = truncnorm.cdf(x=yhat, a=lower_a, b=lower_b, loc=middle_limit, scale=sigmayhat)
            else:
                raise ValueError("Floating error")

            middle_limit = (lower_limit + upper_limit) / 2
            middle_a, middle_b = (ltilde - middle_limit) / sigmayhat, (utilde - middle_limit) / sigmayhat
            middle_quantile = truncnorm.cdf(x=yhat, a=middle_a, b=middle_b, loc=middle_limit, scale=sigmayhat)
            # print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Bisection loop {bisection_loop} -- "
            #       f"lower_limit={round(lower_limit, 2)} (q={round(lower_quantile, 3)}) and "
            #       f"upper_limit={round(upper_limit, 2)} (q={round(upper_quantile, 3)})")
            bisection_loop += 1

        mu_alpha = middle_limit

        return mu_alpha
