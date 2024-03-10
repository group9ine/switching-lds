library(data.table)
library(ggplot2)
library(rstan)
options(mc.cores = parallel::detectCores() - 2)
rstan_options(auto_write = TRUE)

nascar_full <- fread("data/nascar/dataset.csv", sep = ",")
names(nascar_full) <- c("x", "y")

# take a subset
nascar <- nascar_full[seq(1, 4e4 + 5e3, 200)]
nrow(nascar)

plot(nascar$x, nascar$y, type = "b")
points(
  nascar$x[1], nascar$y[1],
  col = "firebrick", pch = 19
)
plot(nascar$x, type = "l")

# Synthetic nascar
z <- rep(1:4, each=20, times=4)
#z <- sample(1:4, size = 300, replace = TRUE, prob = c(0.7, 0.3))
theta <- pi / 20
rot <- \(t) matrix(c(cos(t), sin(t), -sin(t), cos(t)), ncol = 2)

A <- function(z) {
  if (z == 1) {
    cbind(diag(1, 2), c(0.1, 0))
  }
  else if (z == 2) {
    cbind(rot(theta), -rot(theta) %*% c(1,0) + c(1,0) )
  } 
  else if (z == 3) {
    cbind(diag(1, 2), c(-0.1, 0))
  }
  else {
    cbind(rot(theta), -rot(theta) %*% c(-1,0) + c(-1,0) )
  }
}

x <- rep(list(c(0, 0)), length(z))
x[[1]] <- c(-1, -1)
for (i in seq_along(z)) {
  x[[i + 1]] <- A(z[i]) %*% c(x[[i]],1) + rnorm(1, 0, 0.001)
}
x[[1]] <- NULL

data <- as.data.table(transpose(x))
setnames(data, c("x", "y"))

# rescaling
data <- data / mean(sqrt(diff(data$x)^2 + diff(data$y)^2))
plot(data$x[1:80], data$y[1:80], type = "p")

sm <- stan_model(
  file = "stan/r-slds.stan",
  model_name = "SLDS",
  allow_optimizations = TRUE
)

fit <- sampling(
  sm, data = data_list,
  chains = 2, iter = 3000, warmup = 1000
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
