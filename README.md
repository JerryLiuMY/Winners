# Randomization
<p>
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-v3-brightgreen.svg" alt="python"></a> &nbsp;
</p>

## Coverage Probability
The coverage probability with `ntreat=2, 10, 50` and difference between the winning arm and the remaining arms ranging from `0` to `8`.

### Naive method
![alt text](./__resources__/naive_coverage.jpg?raw=true "Title")

### Winners method
![alt text](./__resources__/winners_coverage.jpg?raw=true "Title")


## Power
<a href="./__results__/simulation" target="_blank">Comparision</a> of power between different methods with `nsamples=5000`, `narms=5`, `mu = (np.arange(narms) - 3) / 10`, `cov = np.ones(narms)` for `ntests_li = [1, 2, 3, 4, 5, 10, 20]` and `ntrans = 500`.

### Naive method
![alt text](./__resources__/naive_power.jpg?raw=true "Title")

### Winners method
![alt text](./__resources__/winners_power.jpg?raw=true "Title")

## Caveats
- **Winners:** Fast computation when `mu` is known and the quantity to compute is `alpha`. Slow computation when `alpha` is known and the quantity to compute is `mu`.
