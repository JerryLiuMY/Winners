# Randomization
<p>
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-v3-brightgreen.svg" alt="python"></a> &nbsp;
</p>

## Coverage Probability
The coverage probability with `ntreat=2, 10, 50` and difference between the winning arm and the remaining arms ranging from `0` to `8`.

### Naive method
![alt text](./__resources__/naive.jpg?raw=true "Title")

### Winners method
![alt text](./__resources__/winners.jpg?raw=true "Title")


## Caveats
- **Winners:** Fast computation when `mu` is known and the quantity to compute is `alpha`. Slow computation when `alpha` is known and the quantity to compute is `mu`.
