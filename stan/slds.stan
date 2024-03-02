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
  matrix[N, N + 1] Mu[K];
  cov_matrix[N + 1] Omega[K];  // given as precision = inverse cov
  cov_matrix[N] Psi[K];
  real<lower=N - 1> nu[K];
}

parameters {
  simplex[K] pi[K];  // transition matrix to be soft-maxed

  // linear parameters for y
  matrix[N, N] A[K];
  vector[N] b[K];
  cov_matrix[N] Q[K];  // precision = inverse cov
}

model {
  // assigning priors to linear parameters
  for (k in 1:K) {
    Q[k] ~ wishart(nu[k], Psi[k]);
    append_col(A[k], b[k]) ~ matrix_normal_prec(Mu[k], Q[k], Omega[k]);
  }

  vector[K] gamma[T];  // gamma[t, k] = p(z[t] = k, y[1:t]) 

  for (k in 1:K) {
    gamma[1, k] = dirichlet_lpdf(pi[k] | alpha[k]);
  }
  
  for (t in 2:T) {
    for (k in 1:K) {
      gamma[t, k] =
        log_sum_exp(gamma[t - 1] + log(to_vector(pi[, k])))
        + multi_normal_prec_lpdf(y[t] | A[k] * y[t - 1] + b[k], Q[k]);
    }
  }
  
  target += log_sum_exp(gamma[T]);
}

generated quantities {
  array[T] int<lower=1, upper=K> z_star;
  real log_p_z_star;

  {
    array[T, K] int back_ptr;
    vector[K] eta[T];

    for (k in 1:K) {
      eta[1, k] = dirichlet_lpdf(pi[k] | alpha[k]);
    }

    for (t in 2:T) {
      for (k in 1:K) {
        eta[t, k] = negative_infinity();
        for (j in 1:K) {
          real logp;
          logp = eta[t - 1, j] + log(pi[j, k])
                 + multi_normal_prec_lpdf(y[t] | A[k] * y[t - 1] + b[k], Q[k]);

          if (logp > eta[t, k]) {
            back_ptr[t, k] = j;
            eta[t, k] = logp;
          }
        }
      }
    }

    log_p_z_star = max(eta[T]);

    for (k in 1:K) {
      if (eta[T, k] == log_p_z_star) {
        z_star[T] = k;
      }
    }
    for (t in 1:(T - 1)) {
      z_star[T - t] = back_ptr[T - t + 1, z_star[T - t + 1]];
    }
  }
}
