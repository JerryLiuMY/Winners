from functools import partial
from tools.helper import ptrn2
from params.params import X, sigma, ndraws, tol
import numpy as np

# generate data
Y = X  # replicate the vector of estimates
k = len(X)  # number of treatment arms
sigma = np.kron(np.array([[1, 1], [1, 1]]), sigma)  # replicate the covariance matrix

# compute variables
theta_tilde = np.argmax(X)  # index of the winning arm
ytilde = Y[theta_tilde]  # estimate associated with the winning arm
sigmaytilde = sigma[k + theta_tilde, k + theta_tilde]  # variance of all the estimates
sigmaxytilde = sigma[theta_tilde, (k + theta_tilde)]  # variance of the winning arm
sigmaxytilde_vec = np.array(sigma[(k + theta_tilde), 0:k])  # covariance of the winning arm and other arms
ztilde = np.array(X) - (sigma[(k + theta_tilde), 0:k]) / sigmaytilde * ytilde  # normalised difference


def get_truncation():
    """ Get the truncation threshold for the truncated normal distribution
    :return:
    """

    # The lower truncation value
    ind_l = sigmaxytilde > sigmaxytilde_vec
    if sum(ind_l) == 0:
        ltilde = -np.inf
    elif sum(ind_l) > 0:
        ltilde = max(sigmaytilde * (ztilde[ind_l] - ztilde[theta_tilde]) / (sigmaxytilde - sigmaxytilde_vec[ind_l]))
    else:
        raise ValueError("Invalid ind_l value")

    # The upper truncation value
    ind_u = sigmaxytilde < sigmaxytilde_vec
    if sum(ind_u) == 0:
        utilde = +np.inf
    elif sum(ind_u) > 0:
        utilde = min(sigmaytilde * (ztilde[ind_u] - ztilde[theta_tilde]) / (sigmaxytilde - sigmaxytilde_vec[ind_u]))
    else:
        raise ValueError("Invalid ind_u value")

    # The V truncation value
    ind_v = (sigmaxytilde_vec == sigmaxytilde)
    if sum(ind_v) == 0:
        vtilde = 0
    elif sum(ind_v) > 0:
        vtilde = min(-(ztilde[ind_v] - ztilde[theta_tilde]))
    else:
        raise ValueError("Invalid ind_v value")

    if vtilde < 0:
        return None

    return ltilde, utilde


def search_mu(ltilde, utilde, size):
    """ Get median estimate via bisection search algorithm
    :param ltilde: the lower truncation value
    :param utilde: the upper truncation value
    :param size: quantile to find inverse
    :return:
    """

    yhat = ytilde
    sigmayhat = sigmaytilde
    k = len(X)

    # initialize loop
    check_uniroot = False
    while check_uniroot is False:
        scale = k
        mugridsl = yhat - scale * np.sqrt(sigmayhat)
        mugridsu = yhat + scale * np.sqrt(sigmayhat)
        mugrids = np.array([np.float(mugridsl), np.float(mugridsu)])
        ptrn2_ = partial(ptrn2, Q=yhat, A=ltilde, B=utilde, SIGMA=np.sqrt(sigmayhat), N=ndraws)
        intermediate = np.array(list(map(ptrn2_, mugrids))) - (1 - size)

        halt_condition = abs(max(np.sign(intermediate)) - min(np.sign(intermediate))) > tol
        if halt_condition is True:
            check_uniroot = True
        if halt_condition is False:
            k = 2 * k

    # initialize loop
    mugrids = np.array([0] * 3)
    halt_condition = False
    while halt_condition is False:
        mugridsm = (mugridsl + mugridsu) / 2
        previous_line = mugrids
        mugrids = np.array([np.float(mugridsl), np.float(mugridsm), np.float(mugridsu)])
        ptrn2_ = partial(ptrn2, Q=yhat, A=ltilde, B=utilde, SIGMA=np.sqrt(sigmayhat), N=ndraws)
        intermediate = np.array(list(map(ptrn2_, mugrids))) - (1 - size)

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
