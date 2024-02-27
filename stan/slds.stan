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

  // prior parameters
  vector[K] alpha;  // for pi

  // for A, Q
  matrix[M, M + 1] Mu_x;
  matrix[M + 1, M + 1] Omega_x;  // given as precision = inverse cov
  matrix[M, M] Psi_x;
  real<lower=M - 1> nu_x;

  // for C, S
  matrix[N, M + 1] Mu_y;
  matrix[M + 1, M + 1] Omega_y;
  matrix[N, N] Psi_y;
  real<lower=N - 1> nu_y;
}

parameters {
  simplex[K] trans[K];  // transition matrix

  // linear parameters for x
  matrix[M, M + 1] Ab[K]
  matrix[M, M] Q[K];  // precision = inverse cov

  // linear parameters for y
  matrix[N, M + 1] Cd[K];
  matrix[N, N] S[K];
}

transformed parameters {
  matrix[M, M] A[K] = Ab[:][, :M];
  vector[M] b[K] = Ab[:][, M + 1];

  matrix[N, M] C[K] = Cd[:][, :M];
  vector[N] d[K] = Cd[:][, M + 1];
}

model {
  int z[T];        // discrete latent states
  vector[M] x[T];  // continuous latent states

  // assigning priors
  for (k in 1:K) {
    trans[k] ~ dirichlet(alpha);

    Q[k] ~ wishart(nu_x, Psi_x);
    A[k] ~ matrix_normal_prec(Mu_x, Q[k], Omega_x);

    S[k] ~ wishart(nu_y, Psi_y);
    C[k] ~ matrix_normal_prec(Mu_y, S[k], Omega_y);
  }

  for (t in 2:T) {
    z[t] ~ categorical(trans[z[t - 1]]);
    x[t] ~ multi_normal_prec(A[z[t]] * x[t - 1] + b[z[t]]);
    y[t] ~ multi_normal_prec(C[z[t]] * x[t] + d[z[t]]);
  }
}
