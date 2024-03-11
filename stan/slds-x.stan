functions {
  // returns the log-pdf of a matrix normal, with precision matrices
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
  array[T] vector[N] y;

  // prior for the initial continuous hidden state
  vector[M] mu;
  cov_matrix[M] Sigma;

  // for Ab, Q
  array[K] matrix[M, M + 1] Mu_x;
  array[K] cov_matrix[M + 1] Omega_x;
  array[K] cov_matrix[M] Psi_x;
  array[K] real<lower=M - 1> nu_x;

  // for Cd, S
  matrix[N, M + 1] Mu_y;
  cov_matrix[M + 1] Omega_y;
  cov_matrix[N] Psi_y;
  real<lower=N - 1> nu_y;
}

parameters {
  // transition matrix
  array[K] simplex[K] pi;

  // linear parameters for x
  array[K] matrix[M, M] A;
  array[K] vector[M] b;
  array[K] cov_matrix[M] Q;

  // linear parameters for y
  matrix[N, M] C;
  vector[N] d;
  cov_matrix[N] S;
  
  // continuous hidden states
  array[T] vector[M] x;
}

model {
  // assigning priors to linear parameters
  for (k in 1:K) {
    Q[k] ~ wishart(nu_x[k], Psi_x[k]);
    append_col(A[k], b[k]) ~ matrix_normal_prec(Mu_x[k], Q[k], Omega_x[k]);
  }

  S ~ wishart(nu_y, Psi_y);
  append_col(C, d) ~ matrix_normal_prec(Mu_y, S, Omega_y);

  // gamma[t, k] = log p(z[t] = k, x[1:t], y[1:t])
  array[T] vector[K] gamma;

                        // prior on x[1]
  gamma[1] = rep_vector(multi_normal_prec_lpdf(x[1] | mu, Sigma)
                        // log-likelihood log p(y[1] | x[1])
                        + multi_normal_prec_lpdf(y[1] | C * x[1] + d, S)
                        // uniform prior log p(z[1] = k)
                        - log(K), K);
  
  // forward algorithm main loop
  for (t in 2:T) {
    for (k in 1:K) {
      // create vector with logged entries, then do
      // log(exp(v1) + ... + exp(vk)) with log_sum_exp
      gamma[t, k] = log_sum_exp(gamma[t - 1] + log(to_vector(pi[, k])))
        // log-likelihood log p(x[t] | z[t] = k, x[t - 1])
        + multi_normal_prec_lpdf(x[t] | A[k] * x[t - 1] + b[k], Q[k])
        // log-likelihood log p(y[t] | x[t])
        + multi_normal_prec_lpdf(y[t] | C * x[t] + d, S);
    }
  }
 
  // final marginalization over z
  target += log_sum_exp(gamma[T]);
}

generated quantities {
  // predicted maximum-probability sequence of hidden states
  array[T] int z_star;

  {
    real log_p_z_star;
    array[T, K] int back_ptr;
    array[T] vector[K] eta;

                        // prior on x[1]
    eta[1] = rep_vector(multi_normal_lpdf(x[1] | mu, Sigma)
                        // log-likelihood log p(y[1] | x[1])
                        + multi_normal_prec_lpdf(y[1] | C * x[1] + d, S)
                        // uniform prior log p(z[1] = k)
                        -log(K), K);

    real max_logp;
    vector[K] logp;
 
    // Viterbi algorithm main loop
    for (t in 2:T) {
      for (k in 1:K) {
        max_logp = negative_infinity();
        logp = eta[t - 1] + log(to_vector(pi[, k]));

        // search for the max over logp and save the state in back_ptr
        for (j in 1:K) {
          if (logp[j] > max_logp) {
            max_logp = logp[j];
            back_ptr[t, k] = j;
          }
        }
 
        // complete eta with the log-likelihood part
        eta[t, k] = max_logp
          + multi_normal_prec_lpdf(x[t] | A[k] * x[t - 1] + b[k], Q[k])
          + multi_normal_prec_lpdf(y[t] | C * x[t] + d, S);
      }
    }

    // final maximization + loop to find the argmax
    log_p_z_star = max(eta[T]);
    for (k in 1:K) {
      if (eta[T, k] == log_p_z_star) {
        z_star[T] = k;
        break;
      }
    }
 
    // fill z_star by traversing the back_ptr array
    for (t in 1:(T - 1)) {
      z_star[T - t] = back_ptr[T - t + 1, z_star[T - t + 1]];
    }
  }
}
