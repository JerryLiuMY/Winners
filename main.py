import numpy as np
from models.winners import WINNERS
from tqdm import tqdm

# define mean and covariance
mu = 4 * np.ones(10)
mu[0] = mu[0] + 1
sigma = np.eye(10)

# generate samples
Y_all = np.random.multivariate_normal(mu, sigma, size=100)

# find coverage rate
coverage = []
for Y in tqdm(Y_all):
    winners = WINNERS(Y, sigma)
    ltilde, utilde = winners.get_truncation()
    mu_lower = winners.search_mu(ltilde, utilde, 0.025)
    mu_upper = winners.search_mu(ltilde, utilde, 1 - 0.025)
    coverage.append((mu[0] > mu_lower) & (mu[0] < mu_upper))

coverage_rate = np.mean(coverage)
