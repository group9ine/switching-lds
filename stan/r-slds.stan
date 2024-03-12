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

  array[K] matrix[N, N] Mu_A;
  real lambda_A;
  real kappa_A;

  array[K] vector[N] mu_b;
  real lambda_b;
  real kappa_b;

  real lambda_Q;
  real kappa_Q;

  array[K] matrix[K - 1, N] Mu_R;
  real lambda_R;
  real kappa_R;

  array[K] vector[K - 1] mu_r;
  real lambda_r;
  real kappa_r;
}

parameters {
  array[K] matrix[N, N] Z_A;
  array[K] cholesky_factor_corr[N] L_A;
  array[K] vector<lower=0, upper=pi() / 2>[N] sigma_A_unif;

  array[K] vector[N] z_b;
  array[K] cholesky_factor_corr[N] L_b;
  array[K] vector<lower=0, upper=pi() / 2>[N] sigma_b_unif;

  array[K] cholesky_factor_corr[N] L_Q;
  array[K] vector<lower=0, upper=pi() / 2>[N] sigma_Q_unif;

  array[K] matrix[K - 1, N] Z_R;
  array[K] cholesky_factor_corr[K - 1] L_R;
  array[K] vector<lower=0, upper=pi() / 2>[K - 1] sigma_R_unif;

  array[K] vector[K - 1] z_r;
  array[K] cholesky_factor_corr[K - 1] L_r;
  array[K] vector<lower=0, upper=pi() / 2>[K - 1] sigma_r_unif;
}

transformed parameters {
  array[K] matrix[N, N] A;
  array[K] vector<lower=0>[N] sigma_A;
  array[K] vector[N] b;
  array[K] vector<lower=0>[N] sigma_b;
  array[K] cholesky_factor_cov[N] Q;
  array[K] vector<lower=0>[N] sigma_Q;
  array[K] matrix[K - 1, N] R;
  array[K] vector<lower=0>[N] sigma_R;
  array[K] vector[K - 1] r;
  array[K] vector<lower=0>[N] sigma_r;

  for (k in 1:K) {
    A[k] = Mu_A[k] + diag_pre_multiply(sigma_A[k], L_A[k]) * Z_A[k];
    sigma_A[k] = kappa_A * tan(sigma_A_unif[k]);

    b[k] = mu_b[k] + diag_pre_multiply(sigma_b[k], L_b[k]) * z_b[k];
    sigma_b[k] = kappa_b * tan(sigma_b_unif[k]);

    Q[k] = diag_pre_multiply(sigma_Q[k], L_Q[k]);
    sigma_Q[k] = kappa_Q * tan(sigma_Q_unif[k]);

    R[k] = Mu_R[k] + diag_pre_multiply(sigma_R[k], L_R[k]) * Z_R[k];
    sigma_R[k] = kappa_R * tan(sigma_R_unif[k]);

    r[k] = mu_r[k] + diag_pre_multiply(sigma_r[k], L_r[k]) * z_r[k];
    sigma_r[k] = kappa_r * tan(sigma_r_unif[k]);
  }
}

model {
  for (k in 1:K) {
    to_vector(Z_A[k]) ~ std_normal();
    L_A[k] ~ lkj_corr_cholesky(lambda_A);

    to_vector(z_b[k]) ~ std_normal();
    L_b[k] ~ lkj_corr_cholesky(lambda_b);

    L_Q[k] ~ lkj_corr_cholesky(lambda_Q);

    to_vector(Z_R[k]) ~ std_normal();
    L_R[k] ~ lkj_corr_cholesky(lambda_R);

    to_vector(z_r[k]) ~ std_normal();
    L_r[k] ~ lkj_corr_cholesky(lambda_r);
  }

  array[T] vector[K] log_pk;
  log_pk[1] = rep_vector(-log(K), K);

  for (t in 2:T) {
    for (k in 1:K) {
      log_pk[t, k] =
        log_sum_exp(log_pk[t - 1] + log_p_sb(R[k] * y[t - 1] + r[k]))
        + multi_normal_cholesky_lpdf(y[t] | A[k] * y[t - 1] + b[k], Q[k]);
    }
  }

  target += log_sum_exp(log_pk[T]);
}

generated quantities {
  array[T] int z_star;

  {
    real log_p_z_star;
    array[T, K] int back_ptr;
    array[T] vector[K] max_log_pk;

    max_log_pk[1] = rep_vector(-log(K), K);

    real max_over_zt;
    vector[K] tmp;

    for (t in 2:T) {
      for (k in 1:K) {
        max_over_zt = negative_infinity();
        tmp = max_log_pk[t - 1] + log_p_sb(R[k] * y[t - 1] + r[k]);

        for (j in 1:K) {
          if (tmp[j] > max_over_zt) {
            max_over_zt = tmp[j];
            back_ptr[t, k] = j;
          }
        }

        max_log_pk[t, k] = max_over_zt
          + multi_normal_cholesky_lpdf(y[t] | A[k] * y[t - 1] + b[k], Q[k]);
      }
    }

    log_p_z_star = max(max_log_pk[T]);
    for (k in 1:K) {
      if (max_log_pk[T, k] == log_p_z_star) {
        z_star[T] = k;
        break;
      }
    }

    for (t in 1:(T - 1)) {
      z_star[T - t] = back_ptr[T - t + 1, z_star[T - t + 1]];
    }
  }
}
