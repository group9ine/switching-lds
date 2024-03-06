data {
  int<lower=1> K;  // number of hidden states
  int<lower=1> N;  // number of observed features
  int<lower=1> T;  // length of the time series
  vector[N] y[T];

  // for Ab, Q
  vector[N] Mu[K];
  real lambda[K]; 
  cov_matrix[N] Psi[K];
  real<lower=N - 1> nu[K];
}

parameters {
  simplex[K] pi[K];  // transition matrix to be soft-maxed

  // linear parameters for y
  vector[N] b[K];
  cov_matrix[N] Q[K];  // precision = inverse cov
}

model {
  // assigning priors to linear parameters
  for (k in 1:K) {
    Q[k] ~ inv_wishart(nu[k], Psi[k]);
    b[k] ~ multi_normal(Mu[k], Q[k]/lambda[k]);
  }

  vector[K] gamma[T];  // gamma[t, k] = p(z[t] = k, y[1:t]) 

  gamma[1] = rep_vector(-log(K), K);
  
  for (t in 2:T) {
    for (k in 1:K) {
      gamma[t, k] =
        log_sum_exp(gamma[t - 1] + log(to_vector(pi[, k])))
        + multi_normal_lpdf(y[t] | b[k], Q[k]);
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

    eta[1] = rep_vector(-log(K), K);
 
    for (t in 2:T) {
      for (k in 1:K) {
        real tmp_logp;
        real max_logp = negative_infinity();
        for (j in 1:K) {
          tmp_logp = eta[t - 1, j] + log(pi[j, k]);
          if (tmp_logp > max_logp) {
            max_logp = tmp_logp;
            back_ptr[t, k] = j;
          }
        }
        
        eta[t, k] = max_logp
                    + multi_normal_lpdf(y[t] | b[k], Q[k]);
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
