library(rjags)
library(ggplot2)

## Markov-like process, jumps one step either clockwise or counter-clockwise on
# unit circle
# data = read.csv("data/dataset.csv")
# truez = c(1, sign(diff(atan2(data[, 2], data[, 1]))))
# truez[truez == -1] = 2

data <- read.csv("data/nascar/dataset.csv")
names(data) <- c("x", "y")

## ground truth z, needed in most cases to get a good fit
# truez = rep(1, nrow(data))
# truez[data[, 2] > 0] = 2
# truez[data[, 1] > 4] = 3
# truez[data[, 1] < -4] = 4

# # subsample data if it's repetitive/too big
subset <- seq(1, nrow(data), 200)

# # scale data so minimum distance between points is 10, for a better fit
scale <- sqrt(
  sum(diff(as.matrix(data[subset, ]))^2) / (nrow(data[subset, ]) - 1)
)
D <- 2
K <- 10

mod_data <- NULL
mod_data$x <- data[subset, ] * 10 / scale
mod_data$T <- nrow(mod_data$x)
mod_data$K <- K
mod_data$D <- D

T <- mod_data$T
niter <- 10

plot(ggplot(data[subset, ]) +
  geom_point(aes(x, y)))

mod <- jags.model("jags/r-slds.bug", mod_data)
update(mod, 100000) # burn-in
chain <- coda.samples(mod, c("predx", "predz"), n.iter = niter) # sample
chain.df <- as.data.frame(as.mcmc(chain))

# # retrieve the data from the chain
results.list <- list()
for (i in 1:niter) {
  results.list[[i]] <- list()
  results.list[[i]]$x1 <- NaN * seq(T)
  results.list[[i]]$x2 <- NaN * seq(T)
  results.list[[i]]$z <- NaN * seq(T)
  for (j in 1:T) {
    tryCatch(
      {
        results.list[[i]]$x1[j] <- chain.df[
          i, paste("predx[", j, ",1]", sep = "")
        ]
        results.list[[i]]$x2[j] <- chain.df[
          i, paste("predx[", j, ",2]", sep = "")
        ]
        results.list[[i]]$z[j] <- chain.df[
          i, paste("predz[", j, "]", sep = "")
        ]
      },
      error = function(e) {
        print(i)
        print(j)
        print(e)
      }
    )
  }
  results.list[[i]]$z <- as.factor(results.list[[i]]$z)
}

# look at an example result, to check numbers
# print(coda.samples(mod, c("Ab", "Q", "pi"), n.iter = 1))
print(coda.samples(mod, c("Ab", "Q", "Rc"), n.iter = 1))

# plot
for (i in 1:niter) {
  plot(ggplot() +
    geom_point(aes(
        results.list[[i]]$x1,
        results.list[[i]]$x2,
        col = results.list[[i]]$z
    )) +
    geom_path(aes(results.list[[i]]$x1, results.list[[i]]$x2)))
}

library(data.table)

coda.samples(mod, c("Ab", "Q", "Rc"), n.iter = 1e4) |>
  as.mcmc() |>
  as.data.table() |>
  fwrite("parameters_monza.csv")

print("Done")
