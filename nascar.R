library(data.table)
library(ggplot2)
library(rstan)
options(mc.cores = parallel::detectCores() - 2)
rstan_options(auto_write = TRUE)

nascar_full <- fread("data/nascar/dataset.csv", sep = ",")
names(nascar_full) <- c("x", "y")

# take a subset
nascar <- nascar_full[seq(1, 2e4, 50)]
nrow(nascar)

plot(nascar$x, nascar$y, type = "b"); points(
  nascar$x[1], nascar$y[1], col = "firebrick", pch = 19
)
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
  chains = 1, iter = 4000, warmup = 1000,
  control = list(adapt_delta = 0.9)
)

params <- as.data.frame(extract(fit, permuted = FALSE))
setDT(params)
params[, grep("chain:[^1]|log_|lp", names(params)) := NULL]
names(params) <- gsub("chain:1.", "", names(params), fixed = TRUE)

runmean <- sapply(seq_len(nrow(params)), \(n) mean(log(params$`pi[2,1]`[1:n])))
plot(runmean, type = "l")

divergent <- get_sampler_params(fit, inc_warmup = FALSE)[[1]][, "divergent__"]
sum(divergent) / length(divergent)

params$divergent <- divergent

plot_par <- function(pars) {
  p <- melt(params[, ..pars], measure.vars = pars) |>
    ggplot(aes(value)) +
      geom_histogram(boundary = 0, bins = 50)
  if (length(pars) > 1)
    p <- p + facet_wrap(vars(variable), nrow = length(pars))

  return(p)
}

par_name <- \(pattern) names(params)[grep(pattern, names(params))]

par_name("pi.4") |> plot_par()
par_name("A.4") |> plot_par()
par_name("b.4") |> plot_par()

params[, par_name("z_"), with = FALSE] |> colSums()
