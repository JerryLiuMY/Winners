from params import X, sigma, tol
import numpy as np

# Generate data
Y = X  # replicate the vector of estimates
K = len(X)  # number of treatment arms
sigma = np.kron(np.array([[1, 1], [1, 1]]), sigma)  # replicate the covariance matrix

# Compute variables
theta_tilde = np.argmax(X)  # index of the winning arm
ytilde = Y[theta_tilde]  # estimate associated with the winning arm
sigmaytilde = sigma[K + theta_tilde, K + theta_tilde]  # variance of all the estimates
sigmaxytilde = sigma[theta_tilde, (K + theta_tilde)]  # variance of the winning arm
sigmaxytilde_vec = np.array(sigma[(K + theta_tilde), 0:K])  # covariance of the winning arm and other arms
ztilde = np.array(X) - (sigma[(K + theta_tilde), 0:K]) / sigmaytilde * ytilde  # normalised difference


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

    return [ltilde, utilde]


""" MED_U_ESTIMATE (MEDIAN UNBIASED ESTIMATE) """
YHAT = YTILDE
SIGMAYHAT = SIGMAYTILDE
L = LTILDE
U = UTILDE
SIZE = 0.5
NMC = NDRAWS

CHECK_UNIROOT = False
k = K

while CHECK_UNIROOT is False:
    SCALE = k
    MUGRIDSL = YHAT - SCALE * np.sqrt(SIGMAYHAT)
    MUGRIDSU = YHAT + SCALE * np.sqrt(SIGMAYHAT)
    MUGRIDS = [np.float(MUGRIDSL), np.float(MUGRIDSU)]
    PTRN2_ = partial(PTRN2, Q=YHAT, A=L, B=U, SIGMA=np.sqrt(SIGMAYHAT), N=NMC)
    INTERMEDIATE = np.array(list(map(PTRN2_, MUGRIDS))) - (1 - SIZE)
    HALT_CONDITION = abs(max(np.sign(INTERMEDIATE)) - min(np.sign(INTERMEDIATE))) > TOL
    if HALT_CONDITION is True:
        CHECK_UNIROOT = True
    if HALT_CONDITION is False:
        k = 2 * k

# INITIALISE LOOP.
HALT_CONDITION = False
MUGRIDS = [0] * 3

# SIMPLE BISECTION SEARCH ALGORITHM.
while HALT_CONDITION is False:
    MUGRIDSM = (MUGRIDSL + MUGRIDSU) / 2
    PREVIOUS_LINE = MUGRIDS
    MUGRIDS = [np.float(MUGRIDSL), np.float(MUGRIDSM), np.float(MUGRIDSU)]
    PTRN2_ = partial(PTRN2, Q=YHAT, A=L, B=U, SIGMA=np.sqrt(SIGMAYHAT), N=NMC)
    INTERMEDIATE = np.array(list(map(PTRN2_, MUGRIDS))) - (1 - SIZE)

    if max(abs(MUGRIDS - PREVIOUS_LINE)) == 0:
        HALT_CONDITION = True

    if (abs(INTERMEDIATE[1]) < TOL) or (abs(MUGRIDSU - MUGRIDSL) < TOL):
        HALT_CONDITION = True

    if np.sign(INTERMEDIATE[0]) == np.sign(INTERMEDIATE[1]):
        MUGRIDSL = MUGRIDSM

    if np.sign(INTERMEDIATE[2]) == np.sign(INTERMEDIATE[1]):
        MUGRIDSU = MUGRIDSM

    PYHAT = MUGRIDSM

MED_U_ESTIMATE = PYHAT
