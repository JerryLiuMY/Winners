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

|                | ntests=1, ntrans=500 | ntests=2, ntrans=500 | ntests=3, ntrans=500 | ntests=4, ntrans=500 | ntests=5, ntrans=500 | ntests=10, ntrans=500 | ntests=20, ntrans=500 |
|----------------|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:---------------------:|:---------------------:|
| Naive Method   |        0.264%        |        0.159%        |        0.423%        |        0.309%        |        0.106%        |        0.416%         |        -0.001%        |
| Winners Method |         1.72         |         1.26         |        11.01         |         2.12         |         1.06         |         8.68          |         0.07          |
| RD Method      |        66.6%         |        70.5%         |        68.6%         |        65.9%         |        68.7%         |         67.3%         |           /           |

## Caveats
- **Winners:** Fast computation when `mu` is known and the quantity to compute is `alpha`. Slow computation when `alpha` is known and the quantity to compute is `mu`.

