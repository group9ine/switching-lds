library(rstan)
options(mc.cores = parallel::detectCores() - 2)
rstan_options(auto_write = TRUE)

sm <- stan_model("stan/slds.stan", model_name = "SLDS")
