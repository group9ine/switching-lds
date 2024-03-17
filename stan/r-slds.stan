functions {
  // returns the stick-breaking log-pmf
  vector log_p_sb(vector x) {
    int K = size(x) + 1;
    real sum_denoms = 0.0;
    vector[K] log_p;
    for (k in 1:(K - 1)) {
      sum_denoms += log1p_exp(x[k]);
      log_p[k] = x[k] - sum_denoms;
    }
    log_p[K] = -sum_denoms;

    return log_p;
  }
}

data {
  int K;  // number of hidden states
  int N;  // number of observed features
  int T;  // length of the time series
  array[T] vector[N] y;  // observed data

  // PRIOR PARAMETERS
  // Mu_~     = mean of the multivariate normal
  // lambda_~ = Cauchy variance for the covariance matrix scale
  // kappa_~  = parameter of the LKJ correlation distribution

  // A = linear dynamics roto-translation matrices
  array[K] matrix[N, N] Mu_A;
  real lambda_A;
  real kappa_A;

  // b = linear dynamics biases
  array[K] vector[N] mu_b;
  real lambda_b;
  real kappa_b;

  // Q = linear dynamics covariances
  real lambda_Q;
  real kappa_Q;

  // R = roto-translation matrix for the stick-breaking vector 
  matrix[K - 1, N] Mu_R;
  real lambda_R;
  real kappa_R;

  // r = Markov bias for the stick-breaking vector
  vector[K - 1] mu_r;
  real lambda_r;
  real kappa_r;
}

parameters {
  // Z_~          = (0, 1)-multivariate-normally distributed random variable
  // L_~          = Cholesky factor of the correlation matrix (~ LKJ)
  // sigma_~_unif = uniform [0, pi/2] random variable, then take the tan of it

  array[K] matrix[N, N] Z_A;
  array[K] cholesky_factor_corr[N] L_A;
  array[K] vector<lower=0, upper=pi() / 2>[N] sigma_A_unif;

  array[K] vector[N] z_b;
  array[K] cholesky_factor_corr[N] L_b;
  array[K] vector<lower=0, upper=pi() / 2>[N] sigma_b_unif;

  array[K] cholesky_factor_corr[N] L_Q;
  array[K] vector<lower=0, upper=pi() / 2>[N] sigma_Q_unif;

  matrix[K - 1, N] Z_R;
  cholesky_factor_corr[K - 1] L_R;
  vector<lower=0, upper=pi() / 2>[K - 1] sigma_R_unif;

  vector[K - 1] z_r;
  cholesky_factor_corr[K - 1] L_r;
  vector<lower=0, upper=pi() / 2>[K - 1] sigma_r_unif;
}

transformed parameters {
  array[K] matrix[N, N] A;
  array[K] vector<lower=0>[N] sigma_A;

  array[K] vector[N] b;
  array[K] vector<lower=0>[N] sigma_b;

  array[K] cholesky_factor_cov[N] Q;
  array[K] vector<lower=0>[N] sigma_Q;

  matrix[K - 1, N] R;
  vector<lower=0>[N] sigma_R;

  vector[K - 1] r;
  vector<lower=0>[N] sigma_r;

  for (k in 1:K) {
    // first transform and scale the uniform sigma, so it becomes ~ Cauchy,
    // then sample each parameter as Mu + cholesky_cov * Z, where the Cholesky
    // factor of the covariance matrix is given by diag(sigma) * L, where L
    // follows a LKJ correlation distribution. sigma sets the covariance scale,
    // while L sets the amount of correlation (bigger kappa --> less
    // correlation)

    sigma_A[k] = kappa_A * tan(sigma_A_unif[k]);
    A[k] = Mu_A[k] + diag_pre_multiply(sigma_A[k], L_A[k]) * Z_A[k];

    sigma_b[k] = kappa_b * tan(sigma_b_unif[k]);
    b[k] = mu_b[k] + diag_pre_multiply(sigma_b[k], L_b[k]) * z_b[k];

    sigma_Q[k] = kappa_Q * tan(sigma_Q_unif[k]);
    Q[k] = diag_pre_multiply(sigma_Q[k], L_Q[k]);
  }

  sigma_R = kappa_R * tan(sigma_R_unif);
  R = Mu_R + diag_pre_multiply(sigma_R, L_R) * Z_R;

  sigma_r = kappa_r * tan(sigma_r_unif);
  r = mu_r + diag_pre_multiply(sigma_r, L_r) * z_r;
}

model {
  // define the prior distributions, we only need them for Z and L now
  for (k in 1:K) {
    to_vector(Z_A[k]) ~ std_normal();
    L_A[k] ~ lkj_corr_cholesky(lambda_A);

    z_b[k] ~ std_normal();
    L_b[k] ~ lkj_corr_cholesky(lambda_b);

    L_Q[k] ~ lkj_corr_cholesky(lambda_Q);
  }

  to_vector(Z_R) ~ std_normal();
  L_R ~ lkj_corr_cholesky(lambda_R);

  z_r ~ std_normal();
  L_r ~ lkj_corr_cholesky(lambda_r);

  // FORWARD ALGORITHM
  // log_pk is the recursively evaluated p(z[t] = k, y[1:t])
  array[T] vector[K] log_pk;
  // uniform prior p(z[1] = k)
  log_pk[1] = rep_vector(-log(K), K);

  // main loop
  for (t in 2:T) {
    for (k in 1:K) {
      log_pk[t, k] =
        // accumulate log_pk at previous step weighted by transition probability
        log_sum_exp(log_pk[t - 1] + log_p_sb(R * y[t - 1] + r))
        // log-likelihood y[t] | y[t-1], a multivariate normal
        + multi_normal_cholesky_lpdf(y[t] | A[k] * y[t - 1] + b[k], Q[k]);
    }
  }

  // get log_pk at the last time step and marginalize over k
  target += log_sum_exp(log_pk[T]);
}

generated quantities {
  // VITERBI ALGORITHM
  // declare z sequence to return outside the following block, so it's part of
  // the output
  array[T] int z_star;

  {
    // save arg-maxes in a back pointer: will be reversely traversed later
    array[T, K] int back_ptr;
    // recursively built quantity
    // max_z[1:t-1] log p(z[1:t-1], z[t] = k, y[1:t])
    array[T] vector[K] max_log_pk;

    // uniform prior p(z[1] = k)
    max_log_pk[1] = rep_vector(-log(K), K);

    // temp variables for max search
    real max_over_zt;
    vector[K] tmp;

    for (t in 2:T) {
      for (k in 1:K) {
        max_over_zt = negative_infinity();
        // max_log_pk at previous timestep weighted by transition probabilities
        tmp = max_log_pk[t - 1] + log_p_sb(R * y[t - 1] + r);

        // search for the arg-max and save it in the back pointer
        for (j in 1:K) {
          if (tmp[j] > max_over_zt) {
            max_over_zt = tmp[j];
            back_ptr[t, k] = j;
          }
        }

        // update with log-likelihood y[t] | y[t-1]
        max_log_pk[t, k] = max_over_zt
          + multi_normal_cholesky_lpdf(y[t] | A[k] * y[t - 1] + b[k], Q[k]);
      }
    }

    // maximize over the last time step to get
    // argmax_z[1:T] p(z[1:T], y[1:T])
    real log_p_z_star = max(max_log_pk[T]);
    for (k in 1:K) {
      if (max_log_pk[T, k] == log_p_z_star) {
        z_star[T] = k;
        break;
      }
    }

    // reconstruct the MAP sequence by traversing the back pointer array
    for (t in 1:(T - 1)) {
      z_star[T - t] = back_ptr[T - t + 1, z_star[T - t + 1]];
    }
  }
}
