library(data.table)
library(ggplot2)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = 6)
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
z <- rep(1:3, each = period, times = 20)
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

# set up data and priors
data_list <- list(
  K = 3, N = 2, T = nrow(pacman), y = as.matrix(pacman),
  Mu_A = list(diag(1, 2), rot(theta), diag(1, 2)),
  lambda_A = 2, kappa_A = 0.2,
  mu_b = list(sc_fct * c(dt, dt), c(0, 0), sc_fct * c(-dt, dt)),
  lambda_b = 0.5, kappa_b = 0.05,
  lambda_Q = 2, kappa_Q = 1,
  Mu_R = rep(list(diag(1, 2)), 3),
  lambda_R = 2, kappa_R = 2,
  mu_r = rep(list(c(1, 1)), 3),
  lambda_r = 2, kappa_r = 2
)

sm <- stan_model(
  file = "stan/r-slds.stan",
  model_name = "SLDS",
  allow_optimizations = TRUE
)

pacman_fit <- sampling(
  sm, data = data_list,
  chains = 4, iter = 2500, warmup = 1000
)

# ANALYSIS OF THE FIT
params <- as.data.frame(extract(fit, permuted = FALSE))
setDT(params)
names(params) <- gsub("chain:[0-9].", "", names(params))

plot_par <- function(pars) {
  p <- melt(params[, ..pars], measure.vars = pars,
            value.name = "val", variable.name = "var") |>
    ggplot(aes(val)) +
    geom_histogram(boundary = 0, bins = 50)
  if (length(pars) > 1) {
    p <- p + facet_wrap(vars(var), nrow = length(pars))
  }
  
  return(p)
}

par_name <- \(pattern) names(params)[grep(pattern, names(params))]

par_name("^z_star")

par_name("^b\\[3.") |> plot_par()

par_name("pi.1") |> plot_par()

z_star <- params[, lapply(.SD, mean), .SDcols = par_name("z_star")] |>
  unlist(recursive = FALSE, use.names = FALSE) |>
  as.integer()

all(z_star == z)

# SAVE THE RESULTS
#saveRDS(fit, "fit_pacman.rds")

fit <- readRDS("fit_pacman.rds")

# TIME SERIES RECONSTRUCTION
draws <- as.data.table(extract(fit, permuted = FALSE))
setnames(draws, c("iter", "chain", "param", "value"))

# select parameters for reconstruction and reshape data.table
rec_draws <- draws[grepl("^(?:[AbQ]|z_star)", param)][
  , chain := sub("chain:", "", chain, fixed = TRUE)] |>
    dcast(iter + chain ~ param)

z_cols <- names(rec_draws)[grep("z_star", names(rec_draws))]
z_cols <- z_cols[order(as.integer(gsub("z_star.(\\d+).", "\\1", z_cols)))]

evolve <- function(grp) {
  z_star <- as.integer(grp[, z_cols, with = FALSE])
  A <- lapply(z_star, function(z) {
    grp[, sapply(1:2, \(j) sprintf("A[%d,%d,%d]", z, 1:2, j)),
        with = FALSE] |>
      as.numeric() |>
      matrix(ncol = 2)
  })
  b <- lapply(z_star, function(z) {
    grp[, sprintf("b[%d,%d]", z, 1:2), with = FALSE] |>
      as.numeric()
  })
  Q <- lapply(z_star, function(z) {
    L_Q <- grp[, sapply(1:2, \(j) sprintf("Q[%d,%d,%d]", z, 1:2, j)),
               with = FALSE] |>
      as.numeric() |>
      matrix(ncol = 2)
    return(tcrossprod(L_Q))
  })
  res <- vector(mode = "list", length = length(z_star))
  res[[1]] <- c(0, 0)
  for (t in seq(2, length(z_star)))
    res[[t]] <- A[[t]] %*% res[[t - 1]] + b[[t]]
  for (t in seq(2, length(z_star)))
    res[[t]] <- res[[t]] + MASS::mvrnorm(1, c(0, 0), Q[[t]])
  return(transpose(res))
}

evol <- fread("evolution.csv")
#evol <- rec_draws[, evolve(.SD), by = .(iter, chain)]
setnames(evol, 3:4, c("x", "y"))

p <- evol[chain != 2 & iter %in% sample(iter, 20)] |>
  ggplot(aes(x, y, group = iter)) +
    geom_path(alpha = 0.075, linewidth = 0.5, colour = "#003462") +
    facet_wrap(
      vars(chain), nrow = 3,
      labeller = as_labeller(c(`1` = "Chain 1", `3` = "Chain 3"))
    ) +
    labs(x = expression(italic("x")), y = expression(italic("y")))

ggsave(
  "pacman_evol.svg", plot = p, width = 4 * 2.5,
  height = 3 * 2.5, units = "in"
)
