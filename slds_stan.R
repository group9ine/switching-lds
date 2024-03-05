library(data.table)
library(rstan)
options(mc.cores = parallel::detectCores() - 2)
rstan_options(auto_write = TRUE)

nascar_full <- fread("data/nascar/dataset.csv", sep = ",")
names(nascar_full) <- c("x", "y")

# small subset
nascar <- nascar_full[seq(1, 1e4, 100)]

plot(nascar$x, nascar$y, type = "b")

# simpler data (triangle wave)
period <- 2 * pi
x <- seq(0, 50, length.out = 400)
triangle <- \(x, period) 2 * abs((x %% period) - 0.5 * period) / period 
trwv <- triangle(x, period) + rnorm(length(x), 0, 0.1) - 0.5

plot(x, trwv, type = "b")

# square wave
generate_square_wave <- function(frequency, duration, sampling_rate) {
  t <- seq(0, duration, by = 1/sampling_rate)
  square_wave <- sign(sin(2 * pi * frequency * t))
  return(square_wave)
}

frequency <- 8   # Frequency of the square wave in Hertz
duration <- 5    # Duration of the square wave in seconds
sampling_rate <- 100  # Sampling rate in Hz

square_wave <- generate_square_wave(frequency, duration, sampling_rate)
square_wave <- square_wave + rnorm(length(square_wave),0,0.05)

plot(seq(0, duration, by = 1/sampling_rate), square_wave, type = 'l',
     main = 'Square Wave', xlab = 'Time (s)', ylab = 'Amplitude')
length(square_wave)

# one layer model
sm <- stan_model(
  file = "stan/hmm.stan",
  model_name = "SLDS",
  allow_optimizations = TRUE
)

fit <- sampling(
  sm, data = list(
    K = 2, M = 1, N = 1, T = length(square_wave),
    y = matrix(square_wave, ncol=1),
    Mu = matrix(c(1,-1)),
    lambda = c(1,1),
    Psi = lapply(1:2, \(i) matrix(1)),
    nu = c(1,1)
  ),
  chains = 6, iter = 2000
)

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

library(ggplot2)

ggplot(x_smp, aes(x1, x2)) + geom_bin2d(bins = 70)
