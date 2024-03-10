library(data.table)
library(ggplot2)
library(cmdstanr)
options(mc.cores = 2)
theme_set(theme_minimal(base_size = 18, base_family = "Source Serif 4"))

# synthetic nascar dataset
period <- 10
dt <- 2 / period
z <- rep(1:4, each = period, times = 12)
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
nascar <- nascar / nascar[, mean(sqrt(diff(x)^2 + diff(y)^2))]

ggplot(nascar, aes(x, y)) +
  geom_path(colour = "steelblue") +
  geom_point(data = \(x) x[1], colour = "firebrick", size = 3)

# compile the base rSLDS model
mod <- cmdstan_model("stan/r-slds.stan", compile = FALSE)
mod$check_syntax(pedantic = TRUE)
mod$compile(cpp_options = list(
  stan_cpp_optims = TRUE,
  stan_no_range_checks = TRUE
))

# set up data and priors
data_list <- list(
  K = 4,  # number of hidden states
  N = 2,  # dimension of observed data
  T = nrow(nascar),
  y = as.matrix(nascar),
  # MNW parameters for A, b, Q prior
  Mu = list(A(1), A(2), A(3), A(4)),
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

fit <- mod$sample(
  data = data_list,
  output_dir = "out",
  chains = 2,
  iter_warmup = 1000,
  iter_sampling = 2000,
  show_exceptions = FALSE
)

# save results to file
fit$save_object(file = "out/fit.rds")

# check divergences and other problems
fit$diagnostic_summary()
dgn <- fit$sampler_diagnostics(format = "df") |> as.data.table()
names(dgn) <- gsub("__|[.]", "", names(dgn))
for (col in c("divergent", "chain"))
  set(dgn, j = col, value = factor(dgn[[col]]))

ggplot(dgn, aes(iteration, accept_stat, colour = divergent)) +
  geom_point() +
  facet_wrap(vars(chain), nrow = 2)

# take a look at draws
draws <- fit$draws(format = "df") |> as.data.table()
names(draws) <- gsub("__|[.]", "", names(draws))

par_name <- \(rgx) names(draws)[grep(rgx, names(draws))]
par_name("A.4") |>
  fit$draws() |>
  bayesplot::mcmc_hist_by_chain()

par_name("b.1") |>
  fit$draws() |>
  bayesplot::mcmc_hist_by_chain()

z_draws <- draws[, c(par_name("z_star.[0-9]"), "chain"), with = FALSE][
  , lapply(.SD, mean), by = chain]
z_draws <- melt(
  z_draws, id.vars = "chain",
  variable.name = "iter", value.name = "z_star"
)[ , `:=`(iter = as.integer(gsub("[^0-9]", "", iter)),
          chain = factor(chain))]

ggplot(z_draws, aes(iter, z_star, colour = chain)) +
  geom_point()

par_name("A.3") |>
  fit$draws(format = "df") |>
  dplyr::filter(.chain == 2) |>
  bayesplot::mcmc_hist()
