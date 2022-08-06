from functools import partial
from tools.func import ptrn2
from params.params import tol
import numpy as np


class WINNERS(object):

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

    def search_mu(self, ltilde, utilde, alpha):
        pass

    def search_mu_(self, ltilde, utilde, alpha):
        """ Get median estimate via bisection search algorithm
        :param ltilde: the lower truncation value
        :param utilde: the upper truncation value
        :param alpha: quantile to find inverse
        :return:
        """

        yhat = self.ytilde
        sigmayhat = self.sigmaytilde
        k = self.k

        # initialize loop
        check_uniroot = False
        while check_uniroot is False:
            scale = k
            mugridsl = yhat - scale * np.sqrt(sigmayhat)
            mugridsu = yhat + scale * np.sqrt(sigmayhat)
            mugrids = np.array([np.float(mugridsl), np.float(mugridsu)])
            ptrn2_ = partial(ptrn2, y=yhat, ltilde=ltilde, utilde=utilde, std=np.sqrt(sigmayhat), N=1)
            intermediate = np.array(list(map(ptrn2_, mugrids))) - (1 - alpha)
            halt_condition = abs(max(np.sign(intermediate)) - min(np.sign(intermediate))) > tol
            if halt_condition:
                check_uniroot = True
            else:
                k = 2 * k

        # initialize loop
        mugrids = np.array([0] * 3)
        halt_condition = False
        while halt_condition is False:
            mugridsm = (mugridsl + mugridsu) / 2
            previous_line = mugrids
            mugrids = np.array([np.float(mugridsl), np.float(mugridsm), np.float(mugridsu)])
            ptrn2_ = partial(ptrn2, y=yhat, ltilde=ltilde, utilde=utilde, std=np.sqrt(sigmayhat), N=1)
            intermediate = np.array(list(map(ptrn2_, mugrids))) - (1 - alpha)

            if max(abs(mugrids - previous_line)) == 0:
                halt_condition = True

            if (abs(intermediate[1]) < tol) or (abs(mugridsu - mugridsl) < tol):
                halt_condition = True

            if np.sign(intermediate[0]) == np.sign(intermediate[1]):
                mugridsl = mugridsm

            if np.sign(intermediate[2]) == np.sign(intermediate[1]):
                mugridsu = mugridsm

            mu_estimate = mugridsm

        return mu_estimate
