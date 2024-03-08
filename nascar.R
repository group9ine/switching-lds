library(data.table)
library(ggplot2)
library(rstan)
options(mc.cores = parallel::detectCores() - 2)
rstan_options(auto_write = TRUE)

nascar_full <- fread("data/nascar/dataset.csv", sep = ",")
names(nascar_full) <- c("x", "y")

# take a subset
nascar <- nascar_full[seq(1, 4e4+5e3, 200)]
nrow(nascar)

plot(nascar$x, nascar$y, type = "b"); points(
  nascar$x[1], nascar$y[1], col = "firebrick", pch = 19
)
plot(nascar$x, type = "l")

dt <- mean(sqrt(diff(nascar$x)^2 + diff(nascar$y)^2))
A <- matrix(c(cos(dt), sin(dt), -sin(dt), cos(dt)), ncol = 2)
b <- -A %*% c(1, 0)

states <- c(-1,1)
z <- sample(states, size = 300, replace = T, prob = c(0.7, 0.3)) 
theta <- pi/12
A <- function(z){
  if(z==1) matrix(c(cos(theta),sin(theta),-sin(theta),cos(theta)), ncol=2)
  else matrix(c(cos(-theta),sin(-theta),-sin(-theta),cos(-theta)), ncol=2)
}

x <- rep(list(matrix(c(0,0), ncol=1)), length(z)+1)
x[[1]] <- matrix(c(1,1), ncol=1)
for (i in seq_along(z)){
  x[[i+1]] <- A(z[i]) %*% x[[i]] + rnorm(1, 0, 0.001)
}
x[[1]] <- NULL

x <- lapply(x, \(x) x*10)
data <- as.data.table(transpose(x)) 
setnames(data, c('x', 'y'))
plot(data$x, data$y, type = 'l')

sm <- stan_model(
  file = "stan/slds.stan",
  model_name = "SLDS",
  allow_optimizations = TRUE
)

fit <- sampling(
  sm, data = list(
    K = 2, N = 2, T = nrow(data),
    y = as.matrix(data),
    Mu = list(cbind(A(1), c(0,0)), cbind(A(2), c(0,0))),
    Omega = rep(list(matrix(c(1,0,0,0,1,0,0,0,1), ncol=3)), 2), 
    Psi = rep(list(diag(1/2, 2)), 2),
    nu = rep(2, 2)
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

par_name("pi.1") |> plot_par()
par_name("A.1") |> plot_par()
par_name("b.1") |> plot_par()

ss <- params[, par_name("z_"), with = FALSE] |> colSums()
