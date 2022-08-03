from scipy.stats import truncnorm
from functools import partial
import numpy as np
from conditional import ytilde, sigmaytilde, ltilde, utilde, K
from params import ndraws


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


YHAT = ytilde
SIGMAYHAT = sigmaytilde
L = ltilde
U = utilde
SIZE = 0.5
tol = 1e-6
NMC = ndraws
CHECK_UNIROOT = False
k = K
SCALE = k
MUGRIDSL = YHAT - SCALE * np.sqrt(SIGMAYHAT)
MUGRIDSU = YHAT + SCALE * np.sqrt(SIGMAYHAT)
MUGRIDS = [np.float(MUGRIDSL), np.float(MUGRIDSU)]
PTRN2_ = partial(PTRN2, Q=YHAT, A=L, B=U, SIGMA=np.sqrt(SIGMAYHAT), N=NMC)
INTERMEDIATE = np.array(list(map(PTRN2_, MUGRIDS))) - (1 - SIZE)
HALT_CONDITION = abs(max(np.sign(INTERMEDIATE)) - min(np.sign(INTERMEDIATE))) > tol


# Median unbiased estimate
while CHECK_UNIROOT is False:
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
