functions {
  real matrix_normal_prec_lpdf(matrix X, matrix M, matrix U, matrix V) {
    int N = rows(X);
    int P = cols(X);
    real lp = - N * P * log(2 * pi())
      + N * log_determinant_spd(V)
      + P * log_determinant_spd(U)
      - trace_gen_quad_form(V, U, X - M);

    return 0.5 * lp;
  }
}

data {
  int<lower=1> K;  // number of hidden states
  int<lower=1> N;  // number of observed features
  int<lower=1> T;  // length of the time series
  vector[N] y[T];

  // prior for the transition matrix rows
  vector<lower=0>[K] alpha[K];

  // for Ab, Q
  matrix[N, N + 1] Mu;
  cov_matrix[N + 1] Omega;  // given as precision = inverse cov
  cov_matrix[N] Psi;
  real<lower=N - 1> nu;
}

parameters {
  simplex[K] pi[K];  // transition matrix to be soft-maxed

  // linear parameters for y
  matrix[N, N + 1] Ab[K];
  cov_matrix[N] Q[K];  // precision = inverse cov
}

model {
  // assigning priors to linear parameters
  for (k in 1:K) {
    Q[k] ~ wishart(nu, Psi);
    Ab[k] ~ matrix_normal_prec(Mu, Q[k], Omega);
  }

  // subset Ab and Cd for linear dynamics
  matrix[N, N] A[K] = Ab[, , 1:N];
  vector[N] b[K] = Ab[, , N + 1];

  vector[K] gamma[T];  // gamma[t, k] = p(z[t] = k, y[1:t]) 

  for (k in 1:K) {
    gamma[1, k] += dirichlet_lpdf(pi[k] | alpha[k]);
  }
  
  for (t in 2:T) {
    for (k in 1:K) {
      gamma[t, k] = log_sum_exp(gamma[t - 1] + log(to_vector(pi[, k])))
        + multi_normal_prec_lpdf(y[t] | A[k] * y[t - 1] + b[k], Q[k]);
    }
  }
  
  target += log_sum_exp(gamma[T]);
}
