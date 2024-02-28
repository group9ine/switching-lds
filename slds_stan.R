library(data.table)
library(rstan)
options(mc.cores = parallel::detectCores() - 2)
rstan_options(auto_write = TRUE)

nascar <- fread("data/nascar/dataset.csv", sep = ",")
names(nascar) <- c("x", "y")
n_pts <- nrow(nascar)

plot(nascar$x, nascar$y, type = "l")

sm <- stan_model(
  file = "stan/slds.stan",
  model_name = "SLDS",
  allow_optimizations = TRUE
)

fit <- sampling(
  sm, data = list(
    K = 4, M = 2, N = ncol(nascar), T = nrow(nascar),
    y = as.list(transpose(nascar)),
    init_z = 1, init_x = c(-1, 1),
    alpha = rep(1, 4),
    Mu_x = matrix(0, nrow = 2, ncol = 3),
    Omega_x = solve(matrix(c(1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1), nrow = 3)),
    Psi_x = solve(matrix(c(1, 0, 0, 1), nrow = 2)) / 2,
    nu_x = 2,
    Mu_y = matrix(0, nrow = 2, ncol = 3),
    Omega_y = solve(matrix(c(1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1), nrow = 3)),
    Psi_y = solve(matrix(c(1, 0, 0, 1), nrow = 2)) / 2,
    nu_y = 2
  ),
  chains = 2, iter = 2000
)
