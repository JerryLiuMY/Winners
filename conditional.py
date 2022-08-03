from params import X, SIGMA
import numpy as np

# Generate data
Y = X  # replicate the vector of estimates
K = len(X)  # number of treatment arms and the index of the winning arm
SIGMA = np.kron(np.array([[1, 1], [1, 1]]), SIGMA)  # replicate the covariance matrix

# Compute variables
theta_tilde = np.argmax(X)  # index of the winning arm
ytilde = Y[theta_tilde]  # estimate associated with the winning arm
sigmaytilde = SIGMA[K + theta_tilde, K + theta_tilde]  # variance of all the estimates
sigmaxytilde_vec = np.array(SIGMA[(K + theta_tilde), 0:K])  # covariance of the winning arm and other arms
sigmaxytilde = SIGMA[theta_tilde, (K + theta_tilde)]  # variance of the winning arm

# Normalised difference between each treatment arm and winning arm
ztilde = np.array(X) - (SIGMA[(K + theta_tilde), 0:K]) / sigmaytilde * ytilde

# The lower truncation value
ind_l = sigmaxytilde > sigmaxytilde_vec
if sum(ind_l) == 0:
    ltilde = -np.inf
elif sum(ind_l) > 0:
    ltilde = max(sigmaytilde * (ztilde[ind_l] - ztilde[theta_tilde]) / (sigmaxytilde - sigmaxytilde_vec[ind_l]))
else:
    raise ValueError("Invalid IND_L value")

# The upper truncation value
ind_u = sigmaxytilde < sigmaxytilde_vec
if sum(ind_u) == 0:
    utilde = +np.inf
elif sum(ind_u) > 0:
    utilde = min(sigmaytilde * (ztilde[ind_u] - ztilde[theta_tilde]) / (sigmaxytilde - sigmaxytilde_vec[ind_u]))
else:
    raise ValueError("Invalid IND_L value")

# The V truncation value
ind_v = (sigmaxytilde_vec == sigmaxytilde)
if sum(ind_v) == 0:
    vtilde = 0
elif sum(ind_v) > 0:
    vtilde = min(-(ztilde[ind_v] - ztilde[theta_tilde]))
else:
    raise ValueError("Invalid IND_L value")
