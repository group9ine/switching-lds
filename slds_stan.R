library(data.table)
library(rstan)
options(mc.cores = parallel::detectCores() - 2)
rstan_options(auto_write = TRUE)

nascar_full <- fread("data/nascar/dataset.csv", sep = ",")
names(nascar_full) <- c("x", "y")

# small subset
nascar <- nascar_full[seq(1, 1e4, 100)]

plot(nascar$x, nascar$y, type = "b")

sm <- stan_model(
  file = "stan/slds.stan",
  model_name = "SLDS",
  allow_optimizations = TRUE
)

sm_x <- stan_model(
  file = "stan/slds-x.stan",
  model_name = "SLDS",
  allow_optimizations = TRUE
)

fit_x <- sampling(
  sm_x, data = list(
    K = 4, M = 2, N = ncol(nascar), T = nrow(nascar),
    y = as.list(transpose(nascar)),
    alpha = matrix(1, nrow = 4, ncol = 4),
    mu = c(1, 1), Sigma = diag(1, 2),
    Mu_x = matrix(0, nrow = 2, ncol = 3),
    Omega_x = solve(matrix(c(1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1), nrow = 3)),
    Psi_x = solve(matrix(c(1, 0, 0, 1), nrow = 2)) / 2,
    nu_x = 2,
    Mu_y = matrix(0, nrow = 2, ncol = 3),
    Omega_y = solve(matrix(c(1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1), nrow = 3)),
    Psi_y = solve(matrix(c(1, 0, 0, 1), nrow = 2)) / 2,
    nu_y = 2
  ),
  chains = 2, iter = 300
)

fit <- sampling(
  sm, data = list(
    K = 4, M = 2, N = ncol(nascar), T = nrow(nascar),
    y = as.list(transpose(nascar)),
    alpha = matrix(1, nrow = 4, ncol = 4),
    Mu = matrix(0, nrow = 2, ncol = 3),
    Omega = solve(matrix(c(1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1), nrow = 3)),
    Psi = solve(matrix(c(1, 0, 0, 1), nrow = 2)) / 2,
    nu = 2
  ),
  chains = 2, iter = 300
)

x_smp <- extract(fit, "x")$x
dim(x_smp) <- c(dim(x_smp)[1] * dim(x_smp)[2], dim(x_smp)[3])
x_smp <- as.data.table(x_smp)
setnames(x_smp, c("x1", "x2"))

library(ggplot2)

ggplot(x_smp, aes(x1, x2)) + geom_bin2d(bins = 70)
