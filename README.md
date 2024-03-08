# Switching Linear Dynamical Systems
Prior:

  sampling(
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
