from params import X, SIGMA, ndraws
import numpy as np

# Generate data
Y = X  # replicate the vector of estimates
K = len(X)  # number of treatment arms and the index of the winning arm
SIGMA = np.kron(np.array([[1, 1], [1, 1]]), SIGMA)  # replicate the covariance matrix

# Compute variables
THETA_TILDE = np.argmax(X)  # index of the winning arm
YTILDE = Y[THETA_TILDE]  # estimate associated with the winning arm
SIGMAYTILDE = SIGMA[K + THETA_TILDE, K + THETA_TILDE]  # variance of all the estimates
SIGMAXYTILDE_VEC = np.array(SIGMA[(K + THETA_TILDE), 0:K])  # variance fo the winning arm
SIGMAXYTILDE = SIGMA[THETA_TILDE, (K + THETA_TILDE)]  # covariance of the winning arm and other arms

# Normalised difference between each treatment arm and winning arm
ZTILDE = np.array(X) - (SIGMA[(K + THETA_TILDE), 0:K]) / SIGMAYTILDE * YTILDE

# The lower truncation value
IND_L = SIGMAXYTILDE_VEC < SIGMAXYTILDE
if sum(IND_L) == 0:
    LTILDE = -np.inf
elif sum(IND_L) > 0:
    LTILDE = max(SIGMAYTILDE * (ZTILDE[IND_L] - ZTILDE[THETA_TILDE]) / (SIGMAXYTILDE - SIGMAXYTILDE_VEC[IND_L]))
else:
    raise ValueError("Invalid IND_L value")

# The upper truncation value
IND_U = SIGMAXYTILDE_VEC > SIGMAXYTILDE
if sum(IND_U) == 0:
    UTILDE = +np.inf
elif sum(IND_U) > 0:
    UTILDE = min(SIGMAYTILDE * (ZTILDE[IND_U] - ZTILDE[THETA_TILDE]) / (SIGMAXYTILDE - SIGMAXYTILDE_VEC[IND_U]))
else:
    raise ValueError("Invalid IND_L value")

# the V truncation value
IND_V = (SIGMAXYTILDE_VEC == SIGMAXYTILDE)
if sum(IND_V) == 0:
    VTILDE = 0
elif sum(IND_V) > 0:
    VTILDE = min(-(ZTILDE[IND_V] - ZTILDE[THETA_TILDE]))
else:
    raise ValueError("Invalid IND_L value")
