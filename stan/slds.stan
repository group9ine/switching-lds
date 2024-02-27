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
  int<lower=0> K;  // number of hidden states
  int<lower=0> M;  // dimension of the continuous hidden states
  int<lower=0> N;  // number of observed features
  int<lower=0> T;  // length of the time series
  vector[N] y[T];

  // prior parameters
  real alpha;  // for pi

  // for A, Q
  matrix[M, M + 1] Mu_x;
  matrix[M + 1, M + 1] Lambda_x;  // given as precision = inverse cov
  matrix[M, M] Psi_x;
  real<lower=M - 1> nu_x;

  // for C, S
  matrix[N, M + 1] Mu_y;
  matrix[M + 1, M + 1] Lambda_y;
  matrix[N, N] Psi_y;
  real<lower=N - 1> nu_y;
}

parameters {
  simplex[K] pi;

  // linear parameters for x
  array[K] matrix[M, M + 1] A;
  array[K] matrix[M, M] Q;  // precision = inverse cov

  // linear parameters for y
  array[K] matrix[N, M + 1] C;
  array[K] matrix[N, N] S;
}

model {
  // assigning priors
  for (k in 1:K) {
    pi[k] ~ dirichlet(alpha);

    Q[k] ~ wishart(nu_x, Psi_x);
    A[k] ~ matrix_normal_prec(Mu_x, Q[k], Lambda_x);

    S[k] ~ wishart(nu_y, Psi_y);
    C[k] ~ matrix_normal_prec(Mu_y, S[k], Lambda_y);
  }
}
