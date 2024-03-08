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

  // for Ab, Q
  matrix[M, M + 1] Mu_x[K];
  cov_matrix[M + 1] Omega_x[K];
  cov_matrix[M] Psi_x[K];
  real<lower=M - 1> nu_x[K];

  // for Cd, S
  matrix[N, M + 1] Mu_y;
  cov_matrix[M + 1] Omega_y;
  cov_matrix[N] Psi_y;
  real<lower=N - 1> nu_y;
}

parameters {
  // transition matrix
  simplex[K] pi[K];

  // linear parameters for x
  matrix[M, M] A[K];
  vector[M] b[K];
  cov_matrix[M] Q[K];

  // linear parameters for y
  matrix[N, M] C;
  vector[N] d;
  cov_matrix[N] S;
  
  // continuous hidden states
  vector[M] x[T];
}

model {
  // assigning priors to linear parameters
  for (k in 1:K) {
    Q[k] ~ wishart(nu_x[k], Psi_x[k]);
    append_col(A[k], b[k]) ~ matrix_normal_prec(Mu_x[k], Q[k], Omega_x[k]);
  }

  S ~ wishart(nu_y, Psi_y);
  append_col(C, d) ~ matrix_normal_prec(Mu_y, S, Omega_y);

  vector[K] gamma[T];  // gamma[t, k] = p(z[t] = k, x[1:t], y[1:t]) 

  gamma[1] = rep_vector(multi_normal_prec_lpdf(x[1] | mu, Sigma)
                        + multi_normal_prec_lpdf(y[1] | C * x[1] + d, S)
                        - log(K), K);
  
  for (t in 2:T) {
    for (k in 1:K) {
      gamma[t, k] = log_sum_exp(gamma[t - 1] + log(to_vector(pi[, k])))
        + multi_normal_prec_lpdf(x[t] | A[k] * x[t - 1] + b[k], Q[k])
        + multi_normal_prec_lpdf(y[t] | C * x[t] + d, S);
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

    eta[1] = rep_vector(multi_normal_lpdf(x[1] | mu, Sigma)
                        + multi_normal_prec_lpdf(y[1] | C * x[1] + d, S)
                        -log(K), K);

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
          + multi_normal_prec_lpdf(x[t] | A[k] * x[t - 1] + b[k], Q[k])
          + multi_normal_prec_lpdf(y[t] | C * x[t] + d, S);
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
