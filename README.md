# Switching Linear Dynamical Systems
Prior for the Markov circle:

```r
sampling(
  sm, data = list(
    K = 2, N = 2, T = nrow(data),
    y = as.matrix(data),
    Mu = list(cbind(A(1), c(0, 0)), cbind(A(2), c(0, 0))),
    Omega = rep(list(diag(1, 3)), 2), 
    Psi = rep(list(diag(0.5, 2)), 2),
    nu = rep(2, 2)
  ),
  chains = 1, iter = 4000, warmup = 1000,
  control = list(adapt_delta = 0.9)
)
```
