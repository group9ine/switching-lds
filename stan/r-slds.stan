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
  array[T] vector[N] y_hat;
  for (t in 1:T)
    y_hat[t] = append_row(y, 1);
}

parameters {
  array[K] matrix[N, N + 1] Z_y;
  array[K] cholesky_factor_corr[N] L_Omega_y;
  array[K] vector<lower=0, upper=pi() / 2>[N] tau_unif_y;

  array[K] cholesky_factor_corr[N] L_Omega_v;
  array[K] vector<lower=0, upper=pi() / 2>[N] tau_unif_v;

  array[K] matrix[K - 1, N + 1] Z_r;
  array[K] cholesky_factor_corr[K - 1] L_Omega_r;
  array[K] vector<lower=0, upper=pi() / 2>[K - 1] tau_unif_r;
}

transformed parameters {
  array[K] matrix[N, N + 1] A;
  array[K] vector<lower=0>[N] tau_y;

  array[K] cholesky_factor_corr[N] L_Q;
  array[K] vector<lower=0>[N] tau_v;

  array[K] matrix[K - 1, N + 1] R;
  array[K] vector<lower=0>[K - 1] tau_r;

  for (k in 1:K) {
    tau_y[k] = 2.5 * tan(tau_unif_y[k]);
    A[k] = Mu_y[k] + diag_pre_multiply(tau_y[k], L_Sigma_y[k]) * Z_y[k];

    tau_v[k] = 2.5 * tan(tau_unif_v[k]);
    L_Q[k] = diag_pre_multiply(tau_v[k], L_Omega_v[k]);

    tau_r[k] = 2.5 * tan(tau_unif_r[k]);
    R[k] = Mu_r[k] + diag_pre_multiply(tau_r[k], L_Sigma_r[k]) * Z_r[k];
  }
}

model {
  to_vector(Z_y) ~ std_normal();
  L_Omega_y ~ lkj_corr_cholesky(2);

  L_Omega_v ~ lkj_corr_cholesky(2); 

  to_vector(Z_r) ~ std_normal();
  L_Omega_r ~ lkj_corr_cholesky(2);

  array[T] vector[K] log_pk;
  log_pk[1] = rep_vector(-log(K), K);

  for (t in 2:T) {
    for (k in 1:K) {
      log_pk[t, k] = log_sum_exp(log_pk[t - 1] + log_p_sb(R[k] * y_hat[t - 1]))
        + multi_normal_cholesky_lpdf(y_hat[t] | A[k] * y_hat[t - 1], L_Q[k]);
    }
  }

  target += log_sum_exp(log_pk[T]);
}
