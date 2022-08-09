# Randomization
<p>
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-v3-brightgreen.svg" alt="python"></a> &nbsp;
</p>

## Coverage Probability
The coverage probability of the methods with `ntrials=1000`, `narms=2, 10, 50` and `50` samples per arm. The difference between the winning arm and the remaining arms ranging from `0` to `8`. The mean and covariance being `mu, cov = np.array([mu_max] + [0] * (narms-1)), np.ones(narms)`.

### Naive method
![alt text](./__resources__/naive_coverage.jpg?raw=true "Title")

### Winners method
![alt text](./__resources__/winners_coverage.jpg?raw=true "Title")


## Power
The power of the methods with `ntrials=1000`, `narms=2, 10, 50` and `50` samples per arm. The difference between the winning arm and the remaining arms ranging from `0` to `4`. The mean and covariance being `mu, cov = np.array([mu_max] + [0] * (narms-1)), np.ones(narms)` with `null=0`. 

### Naive method
![alt text](./__resources__/naive_power.jpg?raw=true "Title")

### Winners method
![alt text](./__resources__/winners_power.jpg?raw=true "Title")

### RD method
For the RD method `ntests_li = 5` and `ntrans = 500`.

![alt text](./__resources__/rd_power.jpg?raw=true "Title")

### Comparison
<a href="./__results__/simulation" target="_blank">Comparision</a> of power between different methods with `ntrials=1000`, `narms=5`, `nsamples=5000`, `mu = (np.arange(narms) - 3) / 10`, `cov = np.ones(narms)`. For the RD method `ntests_li = [1, 2, 3, 4, 5, 10, 20]` and `ntrans = 500`.

## Caveats
- **Winners:** Fast computation when `mu` is known and the quantity to compute is `alpha`. Slow computation when `alpha` is known and the quantity to compute is `mu`.

