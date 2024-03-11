# Model description

A switching linear dynamical system -- also known as *switching state
space model* -- is defined, borrowing the notation from @Linderman17, by
the set of discrete-time stochastic equations $$\begin{aligned}
  x_{t} &= A_{z_{t}} x_{t - 1} + b_{z_{t}} + v_{t}, \\
  y_{t} &= C_{z_{t}} x_{t} + d_{z_{t}} + w_{t},
\end{aligned}$$ where $v_{t} \in \R^{M}$ and $w_{t} \in \R^{N}$ are
Gaussian-distributed random vectors with mean zero and variance
$Q_{z_{t}}$, $S_{z_{t}}$ respectively. The vectors $y_{t} \in \R^{N}$,
$t = 1, \ldots, T$ may represent a time series of observations, while
the $x_{t} \in \R^{M}$ are a set of continuous latent states linked
together by linear dynamics defined by the matrices
$A_{k} \in \R^{M \times M}$ and bias vectors $b_{k} \in \R^{M}$. The
transformation between $x$ and $y$ is also linear, through the matrices
$C_{k}
\in \R^{N \times M}$ and bias vectors $d_{k} \in \R^{N}$.

The linear parameters $A_{k}$, $b_{k}$, $C_{k}$, $d_{k}$ form a discrete
set of $K$ elements, and a discrete latent variable
$z_{t} \in \{1, \ldots, K\}$ sets the specific instances in use at time
step $t$. The variable $z$ evolves over time as a Markov process,
meaning that $z_{t}$ is conditionally independent of all previous states
except for its immediate predecessor $z_{t-1}$:
$$p(z_{t} \given z_{t-1}, z_{t-2}, \ldots, z_{1}) = p(z_{t} \given z_{t-1}).$$
We will denote the probability to transition from $z_{t-1} = j$ to
$z_{t} = k$ with $\pi_{jk}$. The transition $z_{t-1} \to z_{t}$
effectively modifies the linear dynamics from $x_{t-1}$ to $x_{t}$ and
the linear transformation from $x_{t}$ to $y_{t}$, *switching* from one
regime to another.

Given a set of data points $y_{t}$, the goal is then to infer the
posterior distribution of the parameter set
$$\theta = \{ \pi_{k}, A_{k}, b_{k}, Q_{k}, C_{k}, d_{k}, S_{k} \},$$
where $x_{1:T}$ denotes the whole sequence
$x_{1}, x_{2}, \ldots, x_{T}$, and $\pi_{k}$ the *k*th row of the
transition matrix.

# Model implementation

To set up a Monte Carlo sampling scheme, we chose to work with the Stan
[@Stan24] programming language and its implementation in R through the
package [rstan]{.sans-serif} [@RStan24]. Sampling in Stan is done by
default using a variant of the Hamiltonian Monte Carlo scheme called
"No-U-Turn sampler" or [@Carpenter17].

The model cannot be implemented directly as it is, in the sense of
specifying a categorical likelihood for the transition
$z_{t} \to z_{t+1}$ and Gaussian likelihoods for $x_{t} \given x_{t-1}$
and $y_{t} \given x_{t}$, because Stan does not allow the definition of
integer parameters: so, one should marginalize over the hidden discrete
states. Besides Stan's limitations in this regard, the resulting
strategy -- known in the literature as *forward algorithm* -- is more
efficient than the straightforward implementation in sampling
low-probability states, and is commonly used in similar inference
problems involving hidden Markov models or other state space models
[@Damiano2018].

## The forward algorithm {#sec:forward-algorithm}

The basic idea behind the forward algorithm is to exploit a recursive
relationship to build the full likelihood: indeed, consider the quantity
$$\gamma_{t}(k) \coloneq p(z_{t} = k, x_{1:t}, y_{1:t}).$$ By summing
over the $z$ states at $t-1$ first and then using the chain rule
repeatedly, we can write $$\begin{split}
    \gamma_{t}(k) &= \sum_{j=1}^{K}
      p(z_{t} = k, z_{t-1} = j, x_{1:t}, y_{1:t}) \\
      &= \!\begin{multlined}[t]
        p(y_{t} \given z_{t} = k, x_{t}) \,
        p(x_{t} \given z_{t} = k, x_{t-1}) \\
        \cdot \sum_{j=1}^{K} \pi_{jk}\,
          p(z_{t-1} = j, x_{1:t-1}, y_{1:t-1}),
      \end{multlined}
  \end{split}
  \label{eq:gamma-relation}$$ where we have recognized the conditional
probability $p(z_{t} = k \given z_{t-1}
= j, x_{1:t-1}, y_{1:t-1}) = p(z_{t} = k \given z_{t-1} = j)$ as the
element $(j, k)$ of the transition matrix $\pi$. The first two terms
outside the sum are the likelihoods of $y_{t}$ and $x_{t}$, and because
of the model definition they only depend on $z_{t}$, $x_{t}$ and
$x_{t-1}$. Also, they are simply Gaussian densities: $$\begin{aligned}
  \lkl_{k}(y_{t}) &\coloneq p(y_{t} \given z_{t} = k, x_{t})
    = \nrm(C_{k} x_{t} + d_{k}, S_{k}), \\
  \lkl_{k}(x_{t}) &\coloneq p(x_{t} \given z_{t} = k, x_{t-1})
    = \nrm(A_{k} x_{t-1} + b_{k}, Q_{k}).
\end{aligned}$$ Then, the remaining terms in the sum in
Eq. [\[eq:gamma-relation\]](#eq:gamma-relation){reference-type="eqref"
reference="eq:gamma-relation"} are nothing else than $\gamma_{t-1}(j)$,
giving us the recursive relation we needed:
$$\gamma_{t}(k) = \lkl_{k}(y_{t})\, \lkl_{k}(x_{t})
    \sum_{j=1}^{K} \pi_{jk} \gamma_{t-1}(j).$$

Indeed, to retrieve the full joint likelihood of the sequences $x_{1:T}$
and $y_{1:T}$ we only need to marginalize $\gamma$ at the last time step
$T$ over the discrete states $k = 1, 2, \ldots, K$:
$$p(x_{1:T}, y_{1:T}) = \sum_{k=1}^{K} p(z_{T} = k, x_{1:T}, y_{1:T})
    = \sum_{k=1}^{K} \gamma_{T} (k).$$ To recursively build $\gamma_{t}$
up to time $T$ we need $\mathcal{O}(TK^{2})$ operations, because of the
double marginalization over $z_{t}$ and $z_{t-1}$. To initialize the
recursion, $$\begin{split}
    \gamma_{1}(k) &= p(z_{1} = k, x_{1}, y_{1}) \\
                  &= \lkl_{1}(y_{1}) \,
                    p(x_{1} \given z_{1} = k) \, p(z_{1} = k).
  \end{split}
  \label{eq:gamma-1}$$ The last two terms are the prior distributions on
$x_{1}$ and $z_{1}$. We chose a multivariate Gaussian for the first and
a uniform distribution over the $K$ states for the second.

At this point, we also need the prior distributions for the dynamical
parameters. Following the suggestion from @Linderman17, we chose
matrix-normal-inverse-Wishart priors: $$\begin{aligned}
  (A_{k}, b_{k}), Q_{k} &\sim \MNIW(M_{x}, \Omega_{x}, \Psi_{x}, \nu_{x}) \\
  (C_{k}, d_{k}), S_{k} &\sim \MNIW(M_{y}, \Omega_{y}, \Psi_{y}, \nu_{y}).
\end{aligned}$$ Here $M_{x} \in \R^{M \times (M + 1)}$ and
$M_{y} \in \R^{N \times (M + 1)}$ are the mean matrices of the matrix
normals, $\Omega_x, \Omega_y \in \R^{(M + 1)
\times (M + 1)}$ their between-column covariance matrices, while
$\Psi_x \in
\R^{M \times M}$ and $\Psi_{y} \in \R^{N \times N}$ are the scale
matrices of the inverse Wisharts and $\nu_{x}, \nu_{y}$ their degrees of
freedom. The returned random matrices with $M + 1$ columns are then
split between the matrices $A_{k}$ and $C_{k}$ and their corresponding
bias vectors $b_{k}$ and $d_{k}$.

At the time of writing, Stan has not yet implemented a matrix normal
distribution function [@Lee17]. Therefore, we have resorted to
implementing it by defining the logarithm of the probability density.
The explicit form of the matrix normal distribution with parameters
$M \in \R^{p
\times q}$, $\Sigma \in \R^{p \times p}$, $\Omega \in \R^{q \times q}$
is [@DeWaal06] $$\begin{gathered}
  p(X \given M, \Sigma, \Omega) =
    \frac{1}{(2 \pi)^{pq/2} \abs{\Omega}^{p/2} \abs{\Sigma}^{q/2}} \\
    \cdot \exp\left\{-\frac{1}{2}\tr\left[\Omega^{-1} (X - M)^\tran
                     \Sigma^{-1} (X - M)\right]\right\}.
\end{gathered}$$

## Reconstructing the hidden states

Since we marginalize out the $z$ sequence during the sampling procedure,
we need a way to recover them probabilistically. One way to do it is to
search *a posteriori* for the most likely hidden sequence of $z$ states
conditioned to the observed $y_{1:T}$ and inferred $x_{1:T}$. This is
done through the so-called "Viterbi algorithm" [@Stan24], which is based
on a recursive relation much like the forward algorithm.

Indeed, consider the quantity $$\eta_{t}(k) \coloneq \argmax_{z_{1:t-1}}
    p(z_{1:t-1}, z_{t} = k, x_{1:t}, y_{1:t}).$$ If we proceed similarly
to [\[eq:gamma-relation\]](#eq:gamma-relation){reference-type="eqref"
reference="eq:gamma-relation"}, and using also the fact that
$$\max_{a,b} [f(a) g(a, b)] = \max_{a}[f(a) \max_{b} g(a, b)],$$ we can
expand the definition as $$\begin{gathered}
  \eta_{t}(k) = \argmax_{j \in \{1, \ldots, K\}}
    [\, p(y_{t} \given z_{t} = k, x_{t}) \,
    p(x_{t} \given z_{t} = k, x_{t-1}) \\
    \cdot \pi_{jk} \argmax_{z_{1:t-2}}
    p(z_{1:t-2}, z_{t-1} = j, x_{1:t-1}, y_{1:t-1})].
\end{gathered}$$ We then recognize the last factor as $\eta$ at the
previous time step $t-1$, giving us the recursive relation
$$\eta_{t}(k) = \lkl_{k}(y_{t}) \lkl_{k}(x_{t})
  \argmax_{j \in \{1, \ldots, K\}} [ \pi_{jk} \, \eta_{t-1}(j)].$$

Regarding the initial value, there is no $z_{t-1}$ to maximize over, so
we get $$\eta_{1}(k) = p(z_{1} = k, x_{1}, y_{1}),$$ which is the same
initialization of $\gamma_{1}(k)$ as shown in the previous section,
Eq. [\[eq:gamma-1\]](#eq:gamma-1){reference-type="eqref"
reference="eq:gamma-1"}. Once we have the value of $\eta$ at time $T$ we
can maximize over $z_{T}$ and recover the maximum-probability sequence
$\hat{z}_{1:T}$: $$\hat{z}_{1:T} = 
  \argmax_{z_{1:T}} p(z_{1:T}, x_{1:T}, y_{1:T}) =
  \argmax_{k \in \{1, \ldots, K\}} \eta_{T}(k).$$ The procedure is thus
substantially the same as the forward algorithm, but with maximization
replacing summation.

# Adding recursion

A possible improvement over the standard model is the one proposed by
@Barber06, referred to by @Linderman17 as *recurrent* . The recursivity
here consists in adding a link between the current discrete hidden state
$z_{t}$ and the continuous hidden state at the previous time step
$x_{t-1}$. In this way the regime switching can have a non-Markovian
dependence on the continuous latent state, greatly enhancing the
descriptive power in situations where there is strong multi-time step
correlation in the linear dynamics.

To model the connection $z_{t} \given x_{t-1}$, @Linderman17 propose a
linear transformation combined with a stick-breaking process to
construct a vector $\pi_{z_{t}}$ of transition probabilities.
Specifically, let $\nu_{t} \in
\R^{K-1}$, defined by $$\nu_{t} = R_{z_{t-1}} x_{t-1} + r_{z_{t-1}},$$
where $R_{k} \in \R^{K-1 \times M}$ and $r_{k} \in \R^{K-1}$. The
relative magnitudes of the matrix $R$ and the bias vector $r$ control
the weight given to the recursive influence ($R$) compared to the pure
Markov dynamics between the $z$ states ($r$) [@Linderman17]. The
transition probabilities of $z_{t}$ conditioned to $x_{t-1}$ are then
given by $$z_{t} \given x_{t-1} \sim \pi_{\mathrm{SB}}(\nu_{t}),
  \label{eq:nu-xtm1}$$ where
$\pi_{\mathrm{SB}} \colon \R^{K-1} \to [0,1]^{K}$ is the stick-breaking
function, mapping the vector $\nu_{t}$ to a normalized probability
vector. Its *k*th component is defined as
$$\pi_{\mathrm{SB}}^{(k)}(\nu) =
  \begin{cases}
    \sigma(\nu_{k}) \prod_{j = 1}^{k} (1 - \sigma(\nu_{j}))
    \quad &\text{if } k \leq K - 1, \\
    \prod_{j = 1}^{K - 1} (1 - \sigma(\nu_{j})) \quad &\text{if } k = K,
  \end{cases}$$ where $\sigma(x) = 1 / (1 + e^{-x})$ denotes the sigmoid
function.

The forward algorithm described in
Section [2.1](#sec:forward-algorithm){reference-type="ref"
reference="sec:forward-algorithm"} can readily accommodate this change
in the transition probabilities. Indeed, recovering the expansion of
$\gamma_{t}(k)$ from
Eq. [\[eq:gamma-relation\]](#eq:gamma-relation){reference-type="eqref"
reference="eq:gamma-relation"}, we just need to adapt the term
$p(z_{t} = k \given z_{t-1} = j, x_{1:t-1},
y_{1:t-1})$. The same is valid for the Viterbi algorithm, of course.
$p(z_{t} =
k \given z_{t-1} = j)$ becomes
$p(z_{t} = k \given z_{t-1} = j, x_{t-1})$, which is the *k*th component
of the stick-breaking function as defined above. Working with
logarithms, we can write this term quite simply as
$$\log p(z_{t} = k \given z_{t-1} = j, x_{t-1}) = \nu_{k} - \sum_{j=1}^{k}
  \log(1 + e^{\nu_{j}}),$$ with $\nu = R_{j}x_{t-1} + r_{j}$ setting the
dependence on $x_{t-1}$.

We also need additional priors for $R_{k}$ and $r_{k}$. In keeping with
the priors for the other sets of linear parameters, we chose a matrix
normal distribution:
$$(R_{k}, r_{k}) \sim \MN(M_{r}, \Sigma_{r}, \Omega_{r}).$$ Here
$M_{r} \in \R^{(K-1) \times (M+1)}$ is the mean matrix, while
$\Sigma_{r}
\in \R^{(K-1) \times (K-1)}$ and
$\Omega_{r} \in \R^{(M+1) \times (M+1)}$ are the between-row and
between-column covariance matrix respectively.
