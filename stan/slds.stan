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

  // initial hidden states
  int init_z;
  vector[M] init_x;

  // prior parameters
  vector[K] alpha;  // for pi

  // for Ab, Q
  matrix[M, M + 1] Mu_x;
  cov_matrix[M + 1] Omega_x;  // given as precision = inverse cov
  cov_matrix[M] Psi_x;
  real<lower=M - 1> nu_x;

  // for Cd, S
  matrix[N, M + 1] Mu_y;
  cov_matrix[M + 1] Omega_y;
  cov_matrix[N] Psi_y;
  real<lower=N - 1> nu_y;
}

parameters {
  simplex[K] trans[K];  // transition matrix

  // linear parameters for x
  matrix[M, M + 1] Ab[K];
  cov_matrix[M] Q[K];  // precision = inverse cov

  // linear parameters for y
  matrix[N, M + 1] Cd[K];
  cov_matrix[N] S[K];  // precision = inverse cov
}

model {
  int z[T];        // discrete latent states
  vector[M] x[T];  // continuous latent states

  // subset Ab and Cd for linear dynamics over x
  matrix[M, M] A[K];
  vector[M] b[K];
  matrix[N, M] C[K];
  vector[N] d[K];

  // assigning priors
  for (k in 1:K) {
    trans[k] ~ dirichlet(alpha);

    Q[k] ~ wishart(nu_x, Psi_x);
    Ab[k] ~ matrix_normal_prec(Mu_x, Q[k], Omega_x);
    A[k] = Ab[k][, :M];
    b[k] = Ab[k][, M + 1];

    S[k] ~ wishart(nu_y, Psi_y);
    Cd[k] ~ matrix_normal_prec(Mu_y, S[k], Omega_y);
    C[k] = Cd[k][, :M];
    d[k] = Cd[k][, M + 1];
  }

  // initialize hidden states
  z[1] = init_z;
  x[1] = init_x;
  y[1] ~ multi_normal_prec(C[z[1]] * x[1] + d[z[1]], S[z[1]]);

  for (t in 2:T) {
    if (sum(trans[z[t - 1]] != 1.0)) print("ERROR");
    z[t] ~ categorical(trans[z[t - 1]]);
    x[t] ~ multi_normal_prec(A[z[t]] * x[t - 1] + b[z[t]], Q[z[t]]);
    y[t] ~ multi_normal_prec(C[z[t]] * x[t] + d[z[t]], S[z[t]]);
  }
}
