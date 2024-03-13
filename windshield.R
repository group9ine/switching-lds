library(data.table)
library(ggplot2)
cmd_inst <- require(cmdstanr)
if (!cmd_inst) {
  library(rstan)
  rstan_options(auto_write = TRUE)
}
options(mc.cores = 6)
theme_set(theme_minimal(base_size = 18, base_family = "Source Serif 4"))

# 2D rotation matrix generating function
rot <- \(t) matrix(c(cos(t), sin(t), -sin(t), cos(t)), ncol = 2)
theta <- pi/10
A <- \(z, t) if (z == 1) rot(t) else rot(-t)
# gaussian noise matrix
Q <- \(z) MASS::mvrnorm(1, mu = c(0, 0), Sigma = diag(1e-4, 2))

# hidden state sequence
z <- rep(c(1,2), times = 20, each = 10)
# generate the data
x <- matrix(0, nrow = 2, ncol = length(z))
x[, 1] <- c(1, 0)
for (j in seq(1, length(z) - 1)) {
  x[, j + 1] <- A(z[j], theta) %*% x[, j]
}
# add noise only after having done all the linear transformations
for (j in seq_along(z)) {
  x[, j] <- x[, j] + Q(z[j])
}

wind <- data.table(t(x))
setnames(wind, c("x", "y"))

# rescale to avoid numerical issues
sc_fct <- 1 / wind[, mean(sqrt(diff(x)^2 + diff(y)^2))]
#sc_fct <- 1
wind <- wind * sc_fct

ggplot(wind, aes(x, y)) +
  geom_path(colour = "steelblue") +
  geom_point(data = \(x) x[1], colour = "firebrick", size = 3)

# set up data and priors
data_list <- list(
  K = 2, N = 2, T = nrow(wind), y = as.matrix(wind),
  Mu_A = list(A(1, theta), A(2, theta)),
  lambda_A = 2, kappa_A = 0.2,
  mu_b = rep(list(c(0, 0)), 2),
  lambda_b = 0.5, kappa_b = 0.05,
  lambda_Q = 2, kappa_Q = 2, 
  Mu_R = rep(list(matrix(c(1,1), nrow=1)), 2),
  lambda_R = 2, kappa_R = 2,
  mu_r = rep(list(c(1)), 2),
  lambda_r = 2, kappa_r = 2
)

if (cmd_inst) {
  # compile the rSLDS model
  mod <- cmdstan_model("stan/r-slds.stan", compile = FALSE)
  mod$check_syntax(pedantic = TRUE)
  mod$compile(cpp_options = list(
    stan_cpp_optims = FALSE,
    stan_no_range_checks = FALSE
  ))

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
} else {
  sm <- stan_model(
    file = "stan/r-slds.stan",
    model_name = "SLDS",
    allow_optimizations = TRUE
  )
}

if (!cmd_inst){
  fit <- sampling(
    sm, data = data_list,
    chains = 6, iter = 2500, warmup = 1000
  )
}

saveRDS(fit, "fit-wind.rds")


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

par_name("^A.2") |>
  fit$draws() |>
  bayesplot::mcmc_hist()

par_name("^b.2") |> fit$draws() |> bayesplot::mcmc_hist_by_chain()
