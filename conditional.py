from params import X, SIGMA, ndraws
from scipy.stats import truncnorm
from functools import partial
import numpy as np

# Generate data
INPUT = np.random.normal(size=len(X) * ndraws).reshape(ndraws, -1)  # draw from standard normal distribution
Y = X  # replicate the vector of estimates
K = len(X)  # number of treatment arms and the index of the winning arm
SIGMA = np.kron(np.array([[1, 1], [1, 1]]), SIGMA)  # replicate the covariance matrix

# Compute variables
theta_tilde = np.argmax(X)  # index of the winning arm
YTILDE = Y[theta_tilde]  # estimate associated with the winning arm
SIGMAYTILDE = SIGMA[K + theta_tilde, K + theta_tilde]  # variance of all the estimates
SIGMAXYTILDE_VEC = np.array(SIGMA[(K + theta_tilde), 0:K])  # variance fo the winning arm
SIGMAXYTILDE = SIGMA[theta_tilde, (K + theta_tilde)]  # covariance of the winning arm and other arms

# Normalised difference between each treatment arm and winning arm
ZTILDE = np.array(X) - (SIGMA[(K + theta_tilde), 0:K]) / SIGMAYTILDE * YTILDE

# The lower truncation value
IND_L = SIGMAXYTILDE_VEC < SIGMAXYTILDE
if sum(IND_L) == 0:
    LTILDE = -np.inf
elif sum(IND_L) > 0:
    LTILDE = max(SIGMAYTILDE * (ZTILDE[IND_L] - ZTILDE[theta_tilde]) / (SIGMAXYTILDE - SIGMAXYTILDE_VEC[IND_L]))
else:
    raise ValueError("Invalid IND_L value")

# The upper truncation value
IND_U = SIGMAXYTILDE_VEC > SIGMAXYTILDE
if sum(IND_U) == 0:
    UTILDE = +np.inf
elif sum(IND_U) > 0:
    UTILDE = min(SIGMAYTILDE * (ZTILDE[IND_U] - ZTILDE[theta_tilde]) / (SIGMAXYTILDE - SIGMAXYTILDE_VEC[IND_U]))
else:
    raise ValueError("Invalid IND_L value")

# the V truncation value
IND_V = (SIGMAXYTILDE_VEC == SIGMAXYTILDE)
if sum(IND_V) == 0:
    VTILDE = 0
elif sum(IND_V) > 0:
    VTILDE = min(-(ZTILDE[IND_V] - ZTILDE[theta_tilde]))
else:
    raise ValueError("Invalid IND_L value")


# APPROXIMATION OF THE CUMULATIVE DISTRIBUTION FUNCTION (I.E., X[I] <= Q) OF THE
# TRUNCATED NORMAL DISTRIBUTION, WHERE WHERE (A,B) ARE THE TRUNCATION POINTS AND
# THE MEAN IS MU.
def PTRN2(MU, Q, A, B, SIGMA, N, SEED=100):
    np.random.seed(SEED)
    TAIL_PROB = np.mean(
        truncnorm.ppf(q=np.random.uniform(N), a=[(A - MU) / SIGMA] * N, b=np.array([(B - MU) / SIGMA] * N) + MU) <=
        ((Q - MU) / SIGMA)
    )

    return TAIL_PROB

# Median unbiased estimate
YHAT = YTILDE
SIGMAYHAT = SIGMAYTILDE
L = LTILDE
U = UTILDE
SIZE = 0.5
NMC = ndraws

CHECK_UNIROOT = False
k = K

while CHECK_UNIROOT is False:
    SCALE = k
    MUGRIDSL = YHAT - SCALE * np.sqrt(SIGMAYHAT)
    MUGRIDSU = YHAT + SCALE * np.sqrt(SIGMAYHAT)
    MUGRIDS = [np.float(MUGRIDSL), np.float(MUGRIDSU)]
    PTRN2_ = partial(PTRN2, Q=YHAT, A=L, B=U, SIGMA=np.sqrt(SIGMAYHAT), N=NMC)
    INTERMEDIATE = np.array(list(map(PTRN2_, MUGRIDS))) - (1 - SIZE)
    HALT_CONDITION = abs(max(np.sign(INTERMEDIATE)) - min(np.sign(INTERMEDIATE))) > tol
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

    if (abs(INTERMEDIATE[1]) < tol) or (abs(MUGRIDSU - MUGRIDSL) < tol):
        HALT_CONDITION = True

    if np.sign(INTERMEDIATE[0]) == np.sign(INTERMEDIATE[1]):
        MUGRIDSL = MUGRIDSM

    if np.sign(INTERMEDIATE[2]) == np.sign(INTERMEDIATE[1]):
        MUGRIDSU = MUGRIDSM

    PYHAT = MUGRIDSM

MED_U_ESTIMATE = PYHAT


# COMPUTATION OF THE MEAN OF THE TRUNCATED NORMAL DISTRIBUTION, WHERE (A,B) ARE
# THE TRUNCATION POINTS AND THE MEAN IS MU.
def ETRN2(MU, A, B, SIGMA, N, SEED=100):
    np.random.seed(SEED)
    TAIL_PROB = SIGMA * np.mean(
        truncnorm.ppf(q=np.random.uniform(N), a=[(A - MU) / SIGMA] * N, b=np.array([(B - MU) / SIGMA] * N) + MU)
    )

    return TAIL_PROB


# FINDS THE THRESHOLD FOR CONFIDENCE REGION EVALUATION.
def CUTRN(MU, Q, A, B, SIGMA, SEED=100):
    np.random.seed(SEED)
    CUT = SIGMA * truncnorm.ppf(q=Q, a=(A - MU) / SIGMA, b=(B - MU) / SIGMA) + MU

    return CUT


# FINDS THE THRESHOLD FOR CONFIDENCE REGION EVALUATION IN THE HYBRID SETTING.
def CHYRN(MU, Q, A, B, SIGMA, CV_BETA, SEED=100):
    np.random.seed(SEED)
    CUT = SIGMA * truncnorm.ppf(p=Q, a=max((A - MU) / SIGMA, -CV_BETA), b=min((B - MU) / SIGMA, +CV_BETA)) + MU

    return CUT
