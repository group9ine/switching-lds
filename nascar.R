library(data.table)
library(ggplot2)
library(rstan)
options(mc.cores = parallel::detectCores() - 2)
rstan_options(auto_write = TRUE)

'nascar_full <- fread("data/nascar/dataset.csv", sep = ",")
names(nascar_full) <- c("x", "y")

# take a subset
nascar <- nascar_full[seq(1, 4e4 + 5e3, 200)]
nrow(nascar)

plot(nascar$x, nascar$y, type = "b")
points(
  nascar$x[1], nascar$y[1],
  col = "firebrick", pch = 19
)
plot(nascar$x, type = "l")'
# synthetic nascar dataset
period <- 10
dt <- 2 / period
z <- rep(1:4, each = period, times = 10)
theta <- pi / period  # dtheta in the curves
# 2D rotation matrix generating function
rot <- \(t) matrix(c(cos(t), sin(t), -sin(t), cos(t)), ncol = 2)

# linear transformation matrix (roto-translation + bias)
A <- function(z) {
  if (z == 1) {
    cbind(diag(1, 2), c(dt, 0))
  } else if (z == 2) {
    cbind(rot(theta), -rot(theta) %*% c(1, 0) + c(1, 0) )
  } else if (z == 3) {
    cbind(diag(1, 2), c(-dt, 0))
  } else {
    cbind(rot(theta), -rot(theta) %*% c(-1, 0) + c(-1, 0) )
  }
}

# gaussian noise matrix
Q <- function(z) {
  MASS::mvrnorm(
    n = 1, mu = c(0, 0),
    Sigma = if (z %% 2) {  # straight sections (z odd)
      matrix(c(5e-5, 0, 0, 1e-4), ncol = 2)
    } else {  # curves (z even)
      matrix(c(2.5e-4, 1e-5, 1e-5, 2.5e-4), ncol = 2)
    }
  )
}

# generate the data
x <- matrix(0, nrow = 2, ncol = length(z))
x[, 1] <- c(-1, -1)
for (j in seq(1, length(z) - 1)) {
  x[, j + 1] <- A(z[j]) %*% c(x[, j], 1)
}
# add noise only after having done all the linear transformations
for (j in seq_along(z)) {
  x[, j] <- x[, j] + Q(z[j])
}

nascar <- data.table(t(x))
setnames(nascar, c("x", "y"))
# rescale to avoid numerical issues
nascar <- 100 * nascar / nascar[, sqrt(mean(diff(x)^2 + diff(y)^2))]

ggplot(nascar, aes(x, y)) +
  geom_path(colour = "steelblue") +
  geom_point(data = \(x) x[1], colour = "firebrick", size = 3)

sm <- stan_model(
  file = "stan/r-slds.stan",
  model_name = "SLDS",
  allow_optimizations = TRUE
)

data_list <- list(
  K = 4,  # number of hidden states
  N = 2,  # dimension of observed data
  T = nrow(nascar),
  y = as.matrix(nascar),
  # MNW parameters for A, b, Q prior
  Mu = list(
    A(1),  # first straight
    A(2),  # first curve
    A(3), # second straight
    A(4)  # second curve
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

fit <- sampling(
  sm, data = data_list,
  chains = 6, iter = 2000, warmup = 1000
)

params <- as.data.frame(extract(fit, permuted = FALSE))
setDT(params)
params[, grep("chain:[^1]|log_|lp", names(params)) := NULL]
names(params) <- gsub("chain:1.", "", names(params), fixed = TRUE)

runmean <- sapply(seq_len(nrow(params)), \(n) mean(log(params$`pi[2,1]`[1:n])))
plot(runmean, type = "l")

divergent <- get_sampler_params(fit, inc_warmup = FALSE)[[1]][, "divergent__"]
sum(divergent) / length(divergent)

plot_par <- function(pars) {
  p <- melt(params[, ..pars], measure.vars = pars) |>
    ggplot(aes(value)) +
      geom_histogram(boundary = 0, bins = 50)
  if (length(pars) > 1) {
    p <- p + facet_wrap(vars(variable), nrow = length(pars))
  }

  return(p)
}

par_name <- \(pattern) names(params)[grep(pattern, names(params))]

par_name("pi.1") |> plot_par()
par_name("A.1") |> plot_par()
par_name("b.1") |> plot_par()

z_star <- params[, lapply(.SD, mean), .SDcols = par_name("z_")] |>
  unlist(recursive = FALSE, use.names = FALSE) |>
  as.integer()

all(z_star == z)
