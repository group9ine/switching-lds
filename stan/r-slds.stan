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

  // returns log p_sb(k | nu)
  // i.e. the k-th component of the stick-breaking function
  vector logp_stick_break(vector nu) {
    int K = size(nu) + 1;
    real sum_denoms = 0.0;  // sum of log(1 + exp(nu_j))
    vector[K] p;
    for (k in 1:(K - 1)) {
      sum_denoms += log1p_exp(nu[k]);
      p[k] = nu[k] - sum_denoms;
    }
    p[K] = -sum_denoms;

    return p;
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

  // for R, r
  array[K] matrix[K - 1, N + 1] Mu_r;
  array[K] cov_matrix[K - 1] Sigma_r;
  array[K] cov_matrix[N + 1] Omega_r;
}

parameters {
  // linear parameters for y
  array[K] matrix[N, N] A;
  array[K] vector[N] b;
  array[K] cov_matrix[N] Q;

  // linear parameters for nu
  array[K] matrix[K - 1, N] R;
  array[K] vector[K - 1] r;
}

model {
  // assigning priors to linear parameters
  for (k in 1:K) {
    Q[k] ~ wishart(nu[k], Psi[k]);
    append_col(A[k], b[k]) ~ matrix_normal_prec(Mu[k], Q[k], Omega[k]);
    append_col(R[k], r[k]) ~ matrix_normal_prec(Mu_r[k], Sigma_r[k], Omega_r[k]);
  }

  // gamma[t, k] = log p(z[t] = k, y[1:t])
  array[T] vector[K] gamma;

  // initialize with prior log p(z[1] = k), uniform
  gamma[1] = rep_vector(-log(K), K);
  
  // forward algorithm main loop
  for (t in 2:T) {
    for (k in 1:K) {
      gamma[t, k] =
        // create vector with logged entries, then do
        // log(exp(v1) + ... + exp(vk)) with log_sum_exp
        log_sum_exp(gamma[t - 1] + logp_stick_break(R[k] * y[t - 1] + r[k]))
        // log-likelihood log p(y[t] | z[t] = k, x[t])
        + multi_normal_prec_lpdf(y[t] | A[k] * y[t - 1] + b[k], Q[k]);
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

    // initialize the recursion with the uniform prior log p(z[1] = k)
    eta[1] = rep_vector(-log(K), K);

    real max_logp;
    vector[K] logp;
 
    // Viterbi algorithm main loop
    for (t in 2:T) {
      for (k in 1:K) {
        max_logp = negative_infinity();
        logp = eta[t - 1] + logp_stick_break(R[k] * y[t - 1] + r[k]);

        // search for the max over logp and save the state in back_ptr
        for (j in 1:K) {
          if (logp[j] > max_logp) {
            max_logp = logp[j];
            back_ptr[t, k] = j;
          }
        }
        
        // complete eta with the log-likelihood part
        eta[t, k] = max_logp
          + multi_normal_prec_lpdf(y[t] | A[k] * y[t - 1] + b[k], Q[k]);
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
