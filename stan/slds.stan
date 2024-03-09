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
  array[T] vector[N] y;

  // for Ab, Q
  array[K] matrix[N, N + 1] Mu;
  array[K] cov_matrix[N + 1] Omega;
  array[K] cov_matrix[N] Psi;
  array[K] real<lower=N - 1> nu;
}

parameters {
  // transition matrix
  array[K] simplex[K] pi;

  // linear parameters for y
  array[K] matrix[N, N] A;
  array[K] vector[N] b;
  array[K] cov_matrix[N] Q;
}

model {
  // assigning priors to linear parameters
  for (k in 1:K) {
    Q[k] ~ wishart(nu[k], Psi[k]);
    append_col(A[k], b[k]) ~ matrix_normal_prec(Mu[k], Q[k], Omega[k]);
  }

  // gamma[t, k] = p(z[t] = k, y[1:t])
  array[T] vector[K] gamma;

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
  array[T] int z_star;
  real log_p_z_star;

  {
    array[T, K] int back_ptr;
    array[T] vector[K] eta;

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
