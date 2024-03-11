library(data.table)
library(ggplot2)
library(cmdstanr)
options(mc.cores = 2)
theme_set(theme_minimal(base_size = 18, base_family = "Source Serif 4"))

# 2D rotation matrix generating function
rot <- \(t) matrix(c(cos(t), sin(t), -sin(t), cos(t)), ncol = 2)

period <- 8
dt <- 1 / period
theta <- 1.5 * pi / period

A <- function(z) {
  if (z == 1) {
    cbind(diag(1, 2), c(dt, dt))
  } else if (z == 2) {
    cbind(rot(theta), c(0, 0))
  } else {
    cbind(diag(1, 2), c(-dt, dt))
  }
}

# gaussian noise matrix
Q <- function(z) {
  MASS::mvrnorm(
    n = 1, mu = c(0, 0),
    Sigma = if (z %% 2) {  # straight sections (z odd)
      matrix(c(5e-5, 0, 0, 1e-4), ncol = 2)
    } else {  # curve (z even)
      matrix(c(2.5e-4, 1e-5, 1e-5, 2.5e-4), ncol = 2)
    }
  )
}

# hidden state sequence
z <- rep(1:3, each = period, times = 10)
# generate the data
x <- matrix(0, nrow = 2, ncol = length(z))
x[, 1] <- c(0, 0)
for (j in seq(1, length(z) - 1)) {
  x[, j + 1] <- A(z[j]) %*% c(x[, j], 1)
}
# add noise only after having done all the linear transformations
for (j in seq_along(z)) {
  x[, j] <- x[, j] + Q(z[j])
}

pacman <- data.table(t(x))
setnames(pacman, c("x", "y"))

# rescale to avoid numerical issues
sc_fct <- 10 / pacman[, mean(sqrt(diff(x)^2 + diff(y)^2))]
pacman <- pacman * sc_fct

ggplot(pacman, aes(x, y)) +
  geom_path(colour = "steelblue") +
  geom_point(data = \(x) x[1], colour = "firebrick", size = 3)

# compile the rSLDS model
mod <- cmdstan_model("stan/r-slds.stan", compile = FALSE)
mod$check_syntax(pedantic = TRUE)
mod$compile(cpp_options = list(
  stan_cpp_optims = FALSE,
  stan_no_range_checks = FALSE
))

# set up data and priors
data_list <- list(
  K = 3,  # number of hidden states
  N = 2,  # dimension of observed data
  T = nrow(pacman),
  y = as.matrix(pacman),
  # MNW parameters for A, b, Q prior
  Mu = list(
    cbind(diag(1, 2), sc_fct * c(dt, dt)),
    A(2),
    cbind(diag(1, 2), -sc_fct * c(dt, dt))
  ),
  Omega = rep(list(diag(1, 3)), 3),  # diagonal precision matrix
  # The exp. value of the Wishart distribution is nu * Psi,
  # so we're setting E[Q] = diag(1, 2) with these values.
  # A lower nu gives a more homogenous distribution
  Psi = rep(list(diag(0.5, 2)), 3),
  nu = rep(2, 3),
  # MN parameters for R, r prior
  # without better intuition, pick something uniform
  Mu_r = rep(list(matrix(1, nrow = 2, ncol = 3)), 3),
  Sigma_r = rep(list(diag(1, 2)), 3),
  Omega_r = rep(list(diag(1, 3)), 3)
)

fit <- mod$sample(
  data = data_list,
  output_dir = "out",
  chains = 2,
  iter_warmup = 1000,
  iter_sampling = 2000,
  show_exceptions = TRUE
)

# save results to file
fit$save_object(file = "out/pacman_fit.rds")

draws <- fit$draws(format = "df") |> as.data.table()
names(draws) <- gsub("[.]|__", "", names(draws))

par_name <- \(rgx) names(draws)[grep(rgx, names(draws))]

z_star <- melt(
  draws[, lapply(.SD, mean), by = chain, .SDcols = par_name("z_star")],
  id.vars = "chain", variable.name = "iter", value.name = "z"
)[, `:=`(chain = factor(chain), iter = as.integer(gsub("[^0-9]", "", iter)))]

ggplot(z_star, aes(iter, z)) +
  geom_point() +
  facet_wrap(vars(chain), nrow = 2)

par_name("A.3") |>
  fit$draws() |>
  bayesplot::mcmc_hist_by_chain()
