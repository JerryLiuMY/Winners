from models.conditional import get_truncation
from models.conditional import search_mu
ltilde, utilde = get_truncation()
mu_estimate = search_mu(ltilde, utilde, size=0.5)
