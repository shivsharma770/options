# Monte Carlo Pricer

This document explains the Monte Carlo (MC) pricer shipped in
`ore::pricing::MonteCarloEngine`, the statistical theory that
underpins it, the variance-reduction choice we've made, and how it
compares to the deterministic engines from Milestones 4 and 8.

## 1. Risk-neutral simulation

Under the risk-neutral measure the underlying \(S\) follows geometric
Brownian motion:

$$
dS/S \;=\; (r - q)\,dt + \sigma\,dW,
$$

with \(W\) a standard Brownian motion, \(r\) the continuously-compounded
risk-free rate, and \(q\) the continuously-compounded dividend yield.
The Itô-solution to this SDE at time \(T\) has closed form:

$$
S(T) \;=\; S(0)\,\exp\!\Bigl((r - q - \tfrac{1}{2}\sigma^2)\,T
                             + \sigma \sqrt{T}\,Z\Bigr),
\qquad Z \sim \mathcal{N}(0, 1).
$$

For a European vanilla option the payoff depends only on \(S(T)\),
never on the intermediate path. This engine therefore performs
**one draw per path**:

```cpp
double s_T   = spot * std::exp(drift * T + sigma * sqrt(T) * Z);
double payoff = std::max(callput_sign * (s_T - strike), 0.0);
```

Multi-step path simulation (needed for barriers, Asians, lookbacks,
early-exercise MC) is a future milestone.

## 2. The estimator, the LLN, and the CLT

The fair value of the option is the risk-neutral discounted expectation
of the payoff:

$$
V \;=\; e^{-rT}\,\mathbb{E}\bigl[\text{payoff}(S(T))\bigr].
$$

With \(N\) i.i.d. samples \(\{S_i(T)\}\), the Monte Carlo estimator is

$$
\hat{V} \;=\; e^{-rT}\,\frac{1}{N}\sum_{i=1}^{N} \text{payoff}(S_i(T)),
$$

which is unbiased. Two classical theorems govern its behaviour:

- **Law of Large Numbers.** \(\hat V \to V\) almost surely as
  \(N \to \infty\).
- **Central Limit Theorem.** For large \(N\),
  \[
    \sqrt{N}\,(\hat V - V) \;\xrightarrow{d}\; \mathcal{N}(0,\,\sigma_{\text{payoff}}^2),
  \]
  so the standard error of \(\hat V\) is
  \[
    \operatorname{SE}(\hat V) \;=\; \sigma_{\text{payoff}} / \sqrt{N}.
  \]
  The convergence rate is \(O(1/\sqrt{N})\) — halving the SE costs 4×
  more paths. This is the fundamental scaling of Monte Carlo.

## 3. Confidence intervals

Given the CLT, a \((1 - \alpha)\)-confidence interval for the true
price is

$$
\Bigl[\hat V - z_{1 - \alpha/2}\,\operatorname{SE},\;
      \hat V + z_{1 - \alpha/2}\,\operatorname{SE}\Bigr].
$$

For 95 %, \(z = 1.9599639\ldots\) The engine exposes this constant as
`MonteCarloEngine::kZ95` so consumers can produce consistent CIs from
a bare `standard_error` when necessary.

**Where does the CI live?** On `PricingResult::confidence_interval_95`
as `std::optional<std::pair<double, double>>`. This is a small,
backward-compatible extension of `PricingResult`: every previous engine
already leaves optional fields empty, and the extra field costs nothing
when unused. The 95 % level is the trader-desk default; other levels
can be derived on the fly from `standard_error` and any Gaussian
quantile.

## 4. Welford's online algorithm

We must compute the sample mean and sample variance *without storing
every path payoff* — a million-path run at 8 bytes/path is 8 MB, but
scaling to 100 M paths would be 800 MB. **Welford (1962)** gives a
one-pass, numerically stable recurrence:

```
mean = 0, M2 = 0, n = 0
for each x:
    n     += 1
    delta = x - mean              // deviation from OLD mean
    mean += delta / n              // update mean in place
    delta_new = x - mean           // deviation from NEW mean
    M2   += delta * delta_new      // update second-moment aggregate
sample_variance = M2 / (n - 1)     // Bessel correction
```

**Why the two-delta form?** The naïve "sum of squares" approach

```
sum   += x
sum_sq += x*x
mean  = sum / n
var   = sum_sq / n - mean * mean
```

catastrophically loses precision for prices with a high mean-to-variance
ratio: `sum_sq - n * mean²` subtracts two large nearly-equal numbers,
so many leading digits cancel. On a million-path ATM option this can
lose 6 digits of relative accuracy in the variance. Welford's
formulation never subtracts near-equal quantities and maintains full
double-precision accuracy for any \(N\).

**Memory footprint.** Three doubles (`mean`, `M2`, `n`) for the whole
simulation. That is the "online" in "online accumulation".

## 5. Antithetic variates

Antithetic variates is one of the simplest and most effective variance
reduction techniques. For every draw \(Z\), we also use \(-Z\) — a
free extra sample because negating is a single flop and requires no
new RNG state.

The estimator becomes

$$
\hat V_{\text{anti}} \;=\; e^{-rT}\,\frac{1}{N}\sum_{i=1}^{N} \tfrac{1}{2}\bigl(\text{payoff}(S_i^{+}) + \text{payoff}(S_i^{-})\bigr),
$$

where \(S_i^{+}\) uses \(Z_i\) and \(S_i^{-}\) uses \(-Z_i\).

The variance of one antithetic pair is

$$
\operatorname{Var}\bigl(\tfrac{1}{2}(Y_+ + Y_-)\bigr) \;=\; \tfrac{1}{2}\bigl(\sigma^2 + \operatorname{Cov}(Y_+, Y_-)\bigr).
$$

For monotone payoffs (vanilla calls, puts, digitals) the payoffs at
\(Z\) and \(-Z\) are **negatively correlated**, so
\(\operatorname{Cov}(Y_+, Y_-) < 0\) and the pair variance is strictly
less than \(\sigma^2\). We get better-than-\(\sqrt{2}\) variance
reduction at the cost of one extra `exp` per pair.

**Limitations.**
- For non-monotone payoffs (e.g. straddles, symmetric butterflies)
  antithetic variates can *increase* variance because
  \(\operatorname{Cov}(Y_+, Y_-) > 0\). We enable them by default only
  because this milestone supports vanilla calls and puts, which are
  guaranteed monotone.
- The "effective sample size" is still \(N\) Welford samples, not
  \(2N\); the standard error is computed over the paired samples. So
  the CIs remain valid.

Antithetic variates is controlled by `Config::antithetic_variates`
(default `true`).

## 6. Common random numbers for Greeks

Bump-and-revalue Greeks with independent seeds are catastrophic —
the noise in \(V(x + h)\) and \(V(x - h)\) is uncorrelated, so their
difference has *more* noise than either individually, and dividing by
\(2h\) magnifies that noise even further.

**Common Random Numbers (CRN)** fix this: use the same RNG seed for
both bumped runs. The two prices then share their per-path noise
exactly, and the finite-difference isolates the derivative signal.

Concretely, the engine constructs a fresh `MersenneTwisterNormalGenerator`
seeded with `Config::seed` before every bumped run. Since the seed is
the same, all runs consume the same sequence of \(Z\)'s.

Bump sizes match `BinomialTreeEngine` for consistency:

| Greek | Method | Bump size |
|-------|--------|-----------|
| Δ, Γ  | central diff in \(S\) (share three MC runs) | \(h = 0.01\,S\) |
| ν     | central diff in \(\sigma\)                  | \(h = 10^{-3}\) |
| ρ     | central diff in \(r\)                       | \(h = 10^{-4}\) |
| Θ     | forward diff toward expiration              | \(h = 1/365\)  |

Cost: 8 additional MC runs of the same size. Disabled by default via
`Config::compute_greeks = false`.

## 7. Computational complexity

| Aspect | Cost |
|--------|------|
| **Time** | \(O(N)\) — one exponential, one payoff per path (two of each per pair in antithetic mode) |
| **Memory** | \(O(1)\) — three Welford aggregates and one RNG state |
| **Greeks** | 8× the base cost when enabled, otherwise 0 |
| **RNG dispatch** | ~2 ns per call (virtual). Negligible in practice. |

## 8. Comparison: Black-Scholes vs Binomial vs Monte Carlo

| Aspect | Black-Scholes | Binomial (CRR) | Monte Carlo |
|--------|---------------|----------------|-------------|
| **Exercise style** | European | European + American | European (this milestone) |
| **Time complexity** | \(O(1)\) | \(O(N^2)\) | \(O(N)\) |
| **Memory** | \(O(1)\) | \(O(N)\) | \(O(1)\) |
| **Error rate** | 0 (exact within model) | \(O(1/N)\) (sawtooth) | \(O(1/\sqrt{N})\) (stochastic) |
| **Path-dependent payoffs** | No | Weakly (lattice-friendly rules) | Yes (with multi-step; future milestone) |
| **Curse of dimensionality** | N/A | Suffers for multi-factor | Immune — cost is dimension-independent |
| **Confidence intervals** | N/A (exact) | N/A (deterministic) | Native — CLT gives them for free |
| **Greeks** | Closed form | Bump-and-revalue (~8×) | Bump-and-revalue + CRN (~8×) |
| **Parallelism** | N/A | Sequential (backward induction) | Embarrassingly parallel |

**When to use which:**
- **Black-Scholes**: European vanillas, high-throughput pricing, IV
  calibration (thousands of prices per second).
- **Binomial**: American vanillas, or when you want a deterministic
  price with monotonic convergence to reason about.
- **Monte Carlo**: Non-vanilla payoffs (Asians, barriers, lookbacks
  once we implement path simulation), multi-factor models
  (stochastic vol, jump-diffusion), or any problem where the state
  space grows faster than trees can handle.

## 9. Design decisions worth flagging

- **`std::normal_distribution` transform is implementation-defined.**
  libstdc++, libc++, and MSVC each use a slightly different variant
  of Box-Muller / Ziggurat. Prices are reproducible within one build
  environment but not across compilers. This is acceptable for a
  research engine; a production engine would ship a custom transform.
- **RNG lives behind a small abstract interface.** The `NormalGenerator`
  base in `ore::numerics` is the single point of extension for later
  quasi-random methods (Sobol, Halton) or better normal transforms
  (Ziggurat, inverse CDF via the Beasley-Springer-Moro algorithm).
  `MonteCarloEngine` does not know or care which concrete generator
  it holds.
- **Seed at engine level, not at call level.** The engine's Config
  owns the seed; every `price()` call constructs its own generator
  seeded from that value. This means the same engine + same inputs
  produces the same price on every call, which is what reproducibility
  should mean.
- **Greeks are off by default.** They cost 8× a base run and most
  research uses want just the price + CI first. Turn them on
  explicitly when needed.

## 10. Explicitly out of scope

Per the milestone brief, this pricer implements **none** of:

- Multi-step path simulation (barriers, Asians, lookbacks, Bermudans).
- Longstaff-Schwartz for American options via MC.
- Control variates or importance sampling.
- Quasi-Monte Carlo (Sobol, Halton) — the RNG abstraction is ready
  for it; adding a concrete `SobolGenerator` is a future task.
- Brownian bridge construction, stratification, or moment matching.
- Stochastic-volatility or local-volatility models.

These are all natural next steps built on the foundation laid here.

## 11. References

- Boyle, P. (1977). "Options: A Monte Carlo Approach". *Journal of
  Financial Economics* 4(3), 323-338. **The original MC option
  pricing paper.**
- Glasserman, P. (2003). *Monte Carlo Methods in Financial
  Engineering*. Springer. **The standard reference.**
- Welford, B.P. (1962). "Note on a method for calculating corrected
  sums of squares and products". *Technometrics* 4(3), 419-420.
- Chan, T.F., Golub, G.H., LeVeque, R.J. (1983). "Algorithms for
  Computing the Sample Variance: Analysis and Recommendations".
  *The American Statistician* 37(3).
- Kloeden, P.E. and Platen, E. (1992). *Numerical Solution of
  Stochastic Differential Equations*. Springer. (For future
  multi-step path simulation.)
- Hull, J.C. (2018). *Options, Futures, and Other Derivatives*,
  10th Ed., Chapter 21 (Monte Carlo).
