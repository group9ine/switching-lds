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

  // for Ab, Q
  matrix[N, N + 1] Mu[K];
  cov_matrix[N + 1] Omega[K];
  cov_matrix[N] Psi[K];
  real<lower=N - 1> nu[K];
}

parameters {
  // transition matrix
  simplex[K] pi[K];

  // linear parameters for y
  matrix[N, N] A[K];
  vector[N] b[K];
  cov_matrix[N] Q[K];
}

model {
  // assigning priors to linear parameters
  for (k in 1:K) {
    Q[k] ~ wishart(nu[k], Psi[k]);
    append_col(A[k], b[k]) ~ matrix_normal_prec(Mu[k], Q[k], Omega[k]);
  }

  vector[K] gamma[T];  // gamma[t, k] = p(z[t] = k, y[1:t]) 

  gamma[1] = rep_vector(-log(K), K);
  
  for (t in 2:T) {
    for (k in 1:K) {
      gamma[t, k] = log_sum_exp(gamma[t - 1] + log(to_vector(pi[, k])))
        + multi_normal_prec_lpdf(y[t] | A[k] * y[t - 1] + b[k], Q[k]);
    }
  }
  
  target += log_sum_exp(gamma[T]);
}

generated quantities {
  int<lower=1, upper=K> z_star[T];
  real log_p_z_star;

  {
    int back_ptr[T, K];
    vector[K] eta[T];

    eta[1] = rep_vector(-log(K), K);

    real max_logp;
    vector[K] logp;
 
    for (t in 2:T) {
      for (k in 1:K) {
        max_logp = negative_infinity();
        logp = eta[t - 1] + log(to_vector(pi[, k]));

        for (j in 1:K) {
          if (logp[j] > max_logp) {
            max_logp = logp[j];
            back_ptr[t, k] = j;
          }
        }
        
        eta[t, k] = max_logp
          + multi_normal_prec_lpdf(y[t] | A[k] * y[t - 1] + b[k], Q[k]);
      }
    }

    log_p_z_star = max(eta[T]);

    for (k in 1:K) {
      if (eta[T, k] == log_p_z_star) {
        z_star[T] = k;
        break;
      }
    }
    
    for (t in 1:(T - 1)) {
      z_star[T - t] = back_ptr[T - t + 1, z_star[T - t + 1]];
    }
  }
}
