library(data.table)
library(ggplot2)
library(rstan)
options(mc.cores = parallel::detectCores() - 2)
rstan_options(auto_write = TRUE)

nascar_full <- fread("data/nascar/dataset.csv", sep = ",")
names(nascar_full) <- c("x", "y")

# take a subset
nascar <- nascar_full[seq(1, 1.5e4, 50)]
nrow(nascar)

plot(nascar$x, nascar$y, type = "b")
points(nascar$x[1], nascar$y[1], col = "firebrick", pch = 19, size = 3)
plot(nascar$x, type = "l")

dt <- mean(sqrt(diff(nascar$x)^2 + diff(nascar$y)^2))
A <- matrix(c(cos(dt), sin(dt), -sin(dt), cos(dt)), ncol = 2)
b <- -A %*% c(1, 0)

sm <- stan_model(
  file = "stan/slds.stan",
  model_name = "SLDS",
  allow_optimizations = TRUE
)

fit <- sampling(
  sm, data = list(
    K = 4, M = 2, N = 2, T = nrow(nascar),
    y = as.matrix(nascar),
    Mu = list(
      cbind(diag(1, 2), c(dt, 0)), cbind(A, b),
      cbind(diag(1, 2), c(-dt, 0)), cbind(A, -b)
    ),
    Omega = rep(list(diag(1, 3)), 4), 
    Psi = rep(list(diag(1 / 3, 2)), 4),
    nu = rep(3, 4)
  ),
  chains = 2, iter = 2000
)
