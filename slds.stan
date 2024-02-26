data {
  int<lower=0> K;  // number of hidden states
  int<lower=0> M;  // dimension of the continuous hidden states
  int<lower=0> N;  // number of observed features
  int<lower=0> T;  // length of the time series
  array[T] vector[N] y;
}

parameters {
  // linear params for x
  array[K] matrix[M, M] A;
  array[K] matrix[M, M] Q;
  array[K] vector[M] b;
  // linear params for y
  array[K] matrix[N, M] C;
  array[K] matrix[N, N] S;
  array[K] vector[N] d;
  simplex[K] pi;
}

model {
  // priors
  pi ~ dirichlet(rep_vector(1, K))  // alpha = 1

  
  vector[K] z;

