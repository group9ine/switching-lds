library(data.table)
library(ggplot2)
library(cmdstanr)
options(mc.cores = 2)

# read nascar dataset and take a subset
nascar_full <- fread("data/nascar/dataset.csv", sep = ",")
names(nascar_full) <- c("x", "y")
nascar <- nascar_full[seq(1, 4e4 + 5e3, 150)]
nrow(nascar)

# rescale to avoid numerical issues
nascar_scl <- nascar / mean(sqrt(diff(nascar$x)^2 + diff(nascar$y)^2))
max(nascar); min(nascar)
max(nascar_scl); min(nascar_scl)

ggplot(nascar_scl, aes(x, y)) +
  geom_path(colour = "steelblue") +
  geom_point(data = \(x) x[1, ], colour = "firebrick")

# compile the base rSLDS model
mod <- cmdstan_model("stan/r-slds.stan", compile = FALSE)
mod$check_syntax(pedantic = TRUE)
mod$compile(cpp_options = list(
  stan_cpp_optims = TRUE,
  stan_no_range_checks = TRUE
))

# 2D rotation matrix generating function
rot <- \(t) matrix(c(cos(t), sin(t), -sin(t), cos(t)), ncol = 2)
# estimate rotation angle
theta <- 1 / 50

# set up data and priors
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
