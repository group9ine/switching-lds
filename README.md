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

Prior for r-slds:

```r
data_list <- list(
  K = 4,  # number of hidden states
  N = 2,  # dimension of observed data
  T = nrow(nascar_scl),
  y = as.matrix(nascar_scl),
  # MNW parameters for A, b, Q prior
  Mu = list(
    cbind(diag(1, 2), c(1, 0)),  # first straight
    cbind(rot(theta), c(0, 0)),  # first curve
    cbind(diag(1, 2), c(-1, 0)), # second straight
    cbind(rot(theta), c(0, 0))   # second curve
  ),
  Omega = rep(list(diag(1, 3)), 4),  # diagonal precision matrix
  
  # The exp. value of the Wishart distribution is nu * Psi,
  # so we're setting E[Q] = diag(1, 2) with these values.
  # A lower nu gives a more homogenous distribution
  Psi = rep(list(diag(0.5, 2)), 4),
  nu = rep(2, 4),
  # MN parameters for R, r prior
  # without better intuition, pick something uniform
  Mu_r = rep(list(matrix(1, nrow = 3, ncol = 3)), 4),
  Sigma_r = rep(list(diag(1, 3)), 4),
  Omega_r = rep(list(diag(1, 3)), 4)
)
```
