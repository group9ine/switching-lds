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
  int<lower=1> M;  // dimension of the continuous hidden states
  int<lower=1> N;  // number of observed features
  int<lower=1> T;  // length of the time series
  vector[N] y[T];

  // prior for the initial continuous hidden state
  vector[M] mu;
  cov_matrix[M] Sigma;

  // prior for the transition matrix rows
  vector<lower=0>[K] alpha[K];

  // for Ab, Q
  matrix[M, M + 1] Mu_x;
  cov_matrix[M + 1] Omega_x;  // given as precision = inverse cov
  cov_matrix[M] Psi_x;
  real<lower=M - 1> nu_x;

  // for Cd, S
  matrix[N, M + 1] Mu_y;
  cov_matrix[M + 1] Omega_y;
  cov_matrix[N] Psi_y;
  real<lower=N - 1> nu_y;
}

parameters {
  simplex[K] pi[K];  // transition matrix to be soft-maxed

  // linear parameters for x
  matrix[M, M + 1] Ab[K];
  cov_matrix[M] Q[K];  // precision = inverse cov

  // linear parameters for y
  matrix[N, M + 1] Cd;
  cov_matrix[N] S;  // precision = inverse cov
  
  vector[M] x[T];  // continuous hidden states
}

model {
  // assigning priors to linear parameters
  for (k in 1:K) {
    Q[k] ~ wishart(nu_x, Psi_x);
    Ab[k] ~ matrix_normal_prec(Mu_x, Q[k], Omega_x);
  }

  S ~ wishart(nu_y, Psi_y);
  Cd ~ matrix_normal_prec(Mu_y, S, Omega_y);

  // subset Ab and Cd for linear dynamics over x
  matrix[M, M] A[K] = Ab[, , 1:M];
  vector[M] b[K] = Ab[, , M + 1];
  matrix[N, M] C = Cd[, 1:M];
  vector[N] d = Cd[, M + 1];

  vector[K] gamma[T];  // gamma[t, k] = p(z[t] = k, x[1:t], y[1:t]) 

  gamma[1] = rep_vector(multi_normal_lpdf(x[1] | mu, Sigma)
                        + multi_normal_prec_lpdf(y[1] | C * x[1] + d, S), K);
  for (k in 1:K) {
    gamma[1, k] += dirichlet_lpdf(pi[k] | alpha[k]);
  }
  
  for (t in 2:T) {
    for (k in 1:K) {
      gamma[t, k] = log_sum_exp(gamma[t - 1] + log(to_vector(pi[, k])))
        + multi_normal_prec_lpdf(x[t] | A[k] * x[t - 1] + b[k], Q[k])
        + multi_normal_prec_lpdf(y[t] | C * x[t] + d, S);
    }
  }
  
  target += log_sum_exp(gamma[T]);
}
