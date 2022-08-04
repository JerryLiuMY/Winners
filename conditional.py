from params import X, sigma
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
