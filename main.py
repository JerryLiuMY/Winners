from models.winners import get_truncation
from models.winners import search_mu
ltilde, utilde = get_truncation()
mu_estimate = search_mu(ltilde, utilde, size=0.5)
