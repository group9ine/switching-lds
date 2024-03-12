functions {
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
  int K;
  int N;
  int T;
  array[T] vector[N] y;

  array[K] matrix[N, N + 1] Mu_y;
  array[K] matrix[K - 1, N + 1] Mu_r;
}

transformed data {
  array[T] vector[N + 1] y_hat;
  for (t in 1:T)
    y_hat[t] = append_row(y[t], 1);
}

parameters {
  array[K] matrix[N, N + 1] Z_y;
  array[K] cholesky_factor_corr[N] L_y;
  array[K] vector<lower=0>[N] sigma_y;

  array[K] cholesky_factor_corr[N] L_v;
  array[K] vector<lower=0>[N] sigma_v;

  array[K] matrix[K - 1, N + 1] Z_r;
  array[K] cholesky_factor_corr[K - 1] L_r;
  array[K] vector<lower=0>[K - 1] sigma_r;
}

transformed parameters {
  array[K] matrix[N, N + 1] A;
  array[K] cholesky_factor_cov[N] Q;
  array[K] matrix[K - 1, N + 1] R;

  for (k in 1:K) {
    A[k] = Mu_y[k] + diag_pre_multiply(sigma_y[k], L_y[k]) * Z_y[k];
    Q[k] = diag_pre_multiply(sigma_v[k], L_v[k]);
    R[k] = Mu_r[k] + diag_pre_multiply(sigma_r[k], L_r[k]) * Z_r[k];
  }
}

model {
  for (k in 1:K) {
    to_vector(Z_y[k]) ~ std_normal();
    L_y[k] ~ lkj_corr_cholesky(2);
    sigma_y[k] ~ lognormal(0, 1);

    L_v[k] ~ lkj_corr_cholesky(2); 
    sigma_v[k] ~ lognormal(0, 1);

    to_vector(Z_r[k]) ~ std_normal();
    L_r[k] ~ lkj_corr_cholesky(2);
    sigma_r[k] ~ lognormal(0, 1);
  }

  array[T] vector[K] log_pk;
  log_pk[1] = rep_vector(-log(K), K);

  for (t in 2:T) {
    for (k in 1:K) {
      log_pk[t, k] = log_sum_exp(log_pk[t - 1] + log_p_sb(R[k] * y_hat[t - 1]))
        + multi_normal_cholesky_lpdf(y[t] | A[k] * y_hat[t - 1], Q[k]);
    }
  }

  target += log_sum_exp(log_pk[T]);
}
