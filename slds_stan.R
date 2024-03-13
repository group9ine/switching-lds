library(data.table)
library(ggplot2)
library(rstan)
options(mc.cores = parallel::detectCores() - 2)
rstan_options(auto_write = TRUE)

# 2D rotation matrix generating function
rot <- \(t) matrix(c(cos(t), sin(t), -sin(t), cos(t)), ncol = 2)
theta <- pi/10
A <- \(z, t) if (z == 1) rot(t) else rot(-t)
# gaussian noise matrix
Q <- \(z) MASS::mvrnorm(1, mu = c(0, 0), Sigma = diag(1e-4, 2))

states <- c(1,2)
# hidden state sequence
z <- sample(states, size = 250, replace = T, prob = c(0.7, 0.3))
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


# one layer model
sm <- stan_model(
  file = "stan/slds.stan",
  model_name = "SLDS",
  allow_optimizations = TRUE
)

fit <- sampling(
  sm, data = list(
    K = 2, N = 2, T = length(wind$x),
    y = wind,
    Mu = list(matrix(c(1, 0.1, 0.1, 1, 1, 1), ncol = 3),
              matrix(c(1, 0.1, 0.1, 1, 1, 1), ncol = 3)),
    Omega = lapply(1:2, \(i) diag(1, 3)),
    Psi = lapply(1:2, \(i) matrix(c(1, 0.1, 0.1, 1), ncol=2)),
    nu = c(2, 2)
  ),
  chains = 2, iter = 2000
)

for(j in 1:2){
  for(k in 1:2){
    A <- extract(
      fit, pars = c(paste("A[1,",j,",",k,"]", sep=""), paste("A[2,",j,",",k,"]", sep="")),
      permuted = FALSE, inc_warmup = FALSE
    ) |> as.data.table()
    
    ggp <- ggplot(A, aes(value, fill = parameters)) +
      geom_histogram(bins = 50, position = "identity", alpha = 0.5) +
      facet_grid(rows = vars(chains))

    ggsave(paste("markov-wing-plots/A[*,",j,",",k,"]",".png", sep=""), ggp, width = 8, height = 5)
  }
}




# x-y layers model
sm_x <- stan_model(
  file = "stan/slds-x.stan",
  model_name = "SLDS",
  allow_optimizations = TRUE
)

fit_x <- sampling(
  sm_x, data = list(
    K = 4, M = 2, N = ncol(nascar), T = nrow(nascar),
    y = as.list(transpose(nascar)),
    alpha = matrix(1, nrow = 4, ncol = 4),
    mu = c(1, 1), Sigma = diag(1, 2),
    Mu_x = matrix(0, nrow = 2, ncol = 3),
    Omega_x = solve(matrix(c(1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1), nrow = 3)),
    Psi_x = solve(matrix(c(1, 0, 0, 1), nrow = 2)) / 2,
    nu_x = 2,
    Mu_y = matrix(0, nrow = 2, ncol = 3),
    Omega_y = solve(matrix(c(1, 0.5, 0, 0.5, 1, 0.5, 0, 0.5, 1), nrow = 3)),
    Psi_y = solve(matrix(c(1, 0, 0, 1), nrow = 2)) / 2,
    nu_y = 2
  ),
  chains = 2, iter = 300
)

x_smp <- extract(fit, "x")$x
dim(x_smp) <- c(dim(x_smp)[1] * dim(x_smp)[2], dim(x_smp)[3])
x_smp <- as.data.table(x_smp)
setnames(x_smp, c("x1", "x2"))

ggplot(x_smp, aes(x1, x2)) + geom_bin2d(bins = 70)
