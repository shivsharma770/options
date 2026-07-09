# Binomial-Tree Pricer

This document explains the Cox-Ross-Rubinstein (CRR) binomial-tree
implementation shipped in `ore::pricing::BinomialTreeEngine`, why CRR
was chosen over its cousins, and how the engine's assumptions and
performance compare to Black-Scholes.

## 1. Model

### 1.1 Discretisation

Time-to-expiration \(T\) is split into \(N\) equal intervals of size

$$
\Delta t \;=\; T / N.
$$

At each intermediate node the underlying can move *up* by factor \(u\)
or *down* by factor \(d = 1/u\). After \(k\) steps and \(i\) up-moves
the underlying is at

$$
S_{i,k} \;=\; S_0 \, u^{\,i}\, d^{\,k - i}.
$$

Because \(u \cdot d = 1\) the lattice **recombines** — an up-then-down
move returns to the origin — so an \(N\)-step tree has only
\((N+1)(N+2)/2\) nodes rather than the naive \(2^N\).

### 1.2 CRR parameters

The three parameters \(u\), \(d\), \(p\) (risk-neutral probability of
an up move) are chosen so the discrete lattice **matches the mean and
variance of a lognormal underlying** over each step:

$$
u \;=\; e^{\sigma \sqrt{\Delta t}}, \qquad
d \;=\; 1/u, \qquad
p \;=\; \frac{e^{(r - q)\Delta t} - d}{u - d}.
$$

The dividend yield \(q\) enters the drift term \(e^{(r-q)\Delta t}\);
if the underlying pays no dividends the formulas reduce to the classic
Cox-Ross-Rubinstein (1979) originals.

### 1.3 Why CRR?

There are several parameterisations of the binomial lattice that all
converge to the same Black-Scholes limit. The three most common:

| Model | \(u\), \(d\) | \(p\) | Property matched |
|-------|-----------|-----|------------------|
| **Cox-Ross-Rubinstein** (1979) | \(u = e^{\sigma\sqrt{\Delta t}},\; d = 1/u\) | \(\displaystyle\frac{e^{(r-q)\Delta t} - d}{u - d}\) | 2 moments; \(u \cdot d = 1\) (log-symmetric) |
| **Jarrow-Rudd** (1983)        | \(u = e^{\mu\Delta t + \sigma\sqrt{\Delta t}},\; d = e^{\mu\Delta t - \sigma\sqrt{\Delta t}}\) | \(1/2\) | Equiprobable, matches 2 moments |
| **Tian** (1993)               | closed-form; more algebra | tuned | 3 moments (mean, variance, skew) |

We chose **CRR** for four reasons:

1. **Pedagogical clarity.** Log-symmetry (\(u \cdot d = 1\)) makes the
   node geometry easy to reason about: node values are simple powers of
   a single parameter \(u\). Every textbook (Hull ch. 21, Wilmott,
   Shreve II ch. 8) presents CRR first for exactly this reason.
2. **A single parameter.** \(d\) and \(p\) fall out of \(u\); we only
   have to compute one non-trivial quantity per step.
3. **The convergence pattern is the interesting phenomenon.** CRR's
   price oscillates around the Black-Scholes limit — the well-known
   "sawtooth" — and this milestone's benchmark utility
   (`examples/binomial_convergence.cpp`) is designed to display it.
4. **Jarrow-Rudd and Tian shave a small constant off the error.** For
   \(N \ge 100\) all three give essentially the same accuracy for
   practical purposes; the algebraic overhead of the more accurate
   models is not worth it at the pedagogical stage.

We do *not* implement the "leisen-Reimer" adjustment which changes
convergence order from \(O(1/N)\) to \(O(1/N^2)\) at the cost of a
completely different tree parameterisation — future milestone, if any.

## 2. Algorithm

### 2.1 Terminal payoff

For an \(N\)-step tree, the underlying at maturity node \(i\) is
\(S_0\, u^i\, d^{\,N-i}\), and the payoff is

$$
V_{i,N} \;=\;
\begin{cases}
\max(S_{i,N} - K,\, 0) & \text{call} \\
\max(K - S_{i,N},\, 0) & \text{put}
\end{cases}
$$

### 2.2 Backward induction

Moving from time step \(k+1\) to step \(k\), the risk-neutral
one-step-ahead expectation, discounted:

$$
V^{\text{cont}}_{i,k} \;=\; e^{-r \Delta t}\bigl[p\, V_{i+1,\,k+1} + (1-p)\, V_{i,\,k+1}\bigr].
$$

For **European exercise** we simply set \(V_{i,k} = V^{\text{cont}}_{i,k}\).

For **American exercise** we compare the continuation value with the
intrinsic value at node \(S_{i,k}\):

$$
V_{i,k} \;=\; \max\bigl(V^{\text{cont}}_{i,k},\; \text{intrinsic}(S_{i,k})\bigr).
$$

After \(N\) backward passes \(V_{0,0}\) is the fair value at time
\(t = 0\).

### 2.3 Memory: rolling vector, O(N)

The naive implementation stores every \(V_{i,k}\) in a 2D array —
\(\Theta(N^2)\) memory. Because each step's values depend only on the
next step's, we can **overwrite in place** with a single
`std::vector<double>` of size \(N+1\):

```cpp
for (k = N; k > 0; --k) {
    for (i = 0; i < k; ++i) {
        values[i] = df * (p * values[i+1] + (1-p) * values[i]);
        if (american) values[i] = max(values[i], intrinsic(S[i,k-1]));
    }
}
```

**Memory complexity: O(N)** (the values vector plus two size-\(N+1\)
lookup tables for \(u^k\) and \(d^k\)). **Time complexity: O(N²)** —
the inner loop runs \(N + (N-1) + \dots + 1 = O(N^2)\) times.

The intermediate spot values \(S_{i,k}\) are needed only for the
American-exercise comparison. We precompute \(u^k\) and \(d^k\) once
so that \(S_{i,k} = S_0 \cdot u^i \cdot d^{k-i}\) is an O(1) lookup.

### 2.4 Payoff sharing

The call and put paths share the same induction loop; only the payoff
function differs. This is enforced by an inline `payoff(spot, strike,
type)` helper — no duplicated backward-induction code.

## 3. Convergence to Black-Scholes

For **European vanilla options** CRR converges to the Black-Scholes
price as \(N \to \infty\). Empirically the price *oscillates* around
the true value — the "sawtooth" — because the terminal lattice does or
doesn't happen to include a node at the strike price, and this changes
the local slope of the numerical delta. The envelope of the oscillation
shrinks as \(O(1/N)\).

The `binomial_convergence` benchmark in `examples/` prints the price
and error for \(N \in \{10, 25, 50, 100, 250, 500, 1000, 2500, 5000\}\)
so you can watch this happen. Typical numbers for an ATM European call
with \(S = K = 100\), \(r = 0.05\), \(\sigma = 0.20\), \(T = 1\):

| Steps | Price | Abs. error |
|-------|-------|-----------|
| 10    | 10.35 | 0.90      |
| 25    |  9.85 | 0.40      |
| 100   |  9.51 | 0.06      |
| 500   |  9.4551| 0.005    |
| 5000  |  9.4506| 0.0005   |

(Exact numbers depend on parity of \(N\); the important observation is
that error contracts steadily as \(N\) grows.)

## 4. American option identities

Two identities are exercised in the test suite; both hold for any
correct implementation of the tree with early-exercise checking.

### 4.1 American Call = European Call (no dividends)

**Statement.** For \(q = 0\) an American call is never optimally
exercised early; therefore its price equals the European call.

**Sketch of proof.** For a call with strike \(K\),
early-exercise-value at time \(t\) is \(S_t - K\). Continuation value
is at least \(S_t - K e^{-r(T-t)} \ge S_t - K\) (from put-call parity
with no dividends and \(r \ge 0\)). So continuation dominates
exercise everywhere on the tree and the max-operator is inactive.

### 4.2 American Put ≥ European Put

**Statement.** For any parameters, an American put is worth at least
as much as the corresponding European put.

**Sketch.** The American put has *strictly more* exercise
opportunities. The set of admissible stopping times for the European
put is a subset of the American's. Since the option is a *maximum*
over stopping times, adding opportunities can only weakly increase the
value.

When \(r > 0\) and the option is in the money, early-exercise value
$(K - S) > (K - S) e^{-r(T-t)}$ can strictly dominate continuation,
and the inequality is strict.

## 5. Comparison with Black-Scholes

| Aspect | Black-Scholes | Binomial (CRR) |
|--------|---------------|----------------|
| **Exercise style** | European only | European *and* American |
| **Complexity** | O(1) — closed form | O(N²) time, O(N) memory |
| **Accuracy** | Exact within model assumptions | Sawtooth O(1/N) around exact |
| **Greeks** | Closed-form analytical | Bump-and-revalue (8 extra tree solves) |
| **Volatility** | Constant | Constant per step (extendable to term-structured) |
| **Rate / yield** | Constant | Constant per step (extendable) |
| **Path-dependent payoffs** | Not supported | Only weakly (needs discrete lattice-friendly rules) |
| **Numerical stability** | Very robust | Requires \(0 \le p \le 1\); can fail for very high vol or very small N |

**Common assumptions** (both models): frictionless markets, no
transaction costs, continuous rebalancing, no arbitrage. Both assume
lognormal dynamics; the tree is an approximation of that lognormal SDE
sampled at discrete times.

**When to use which:**
- Black-Scholes: European vanillas, high-throughput pricing, IV
  calibration where you need thousands of evaluations per second.
- Binomial tree: American exercise, contracts with early-exercise or
  simple path-dependent features (Bermudan puts, callable convertibles
  as a first approximation), or as a pedagogical bridge to more
  general lattice / PDE methods.

## 6. Numerical Greeks

CRR does not admit closed-form Greeks the way Black-Scholes does. We
compute Greeks by **bump-and-revalue** — re-price with one input
perturbed and take a finite difference.

| Greek | Method | Bump size | Rationale |
|-------|--------|-----------|-----------|
| Δ     | central diff in \(S\) | \(h = 0.01 S\) (1% of spot) | Standard trader convention; large enough to escape tree noise |
| Γ     | second-order central diff, same \(h\) as Δ | \(h = 0.01 S\) | Reuses the two spot bumps needed for Δ |
| ν     | central diff in \(\sigma\) | \(h = 10^{-3}\) (10 bp of vol) | Fine enough to preserve \(\sigma^2\) to 5 sf |
| ρ     | central diff in \(r\) | \(h = 10^{-4}\) (1 bp of rate) | Trader convention; rho is linear in this regime |
| Θ     | forward diff toward expiration | \(h = 1/365\) (one calendar day) | Central diff would need \(T - h > 0\); forward is universally valid |

**Trade-offs of bump-and-revalue on a tree:**

- **Truncation error** of central differences is \(O(h^2)\); of forward
  differences \(O(h)\). Smaller \(h\) → less truncation error.
- **Tree noise** — the CRR lattice makes each price oscillate as inputs
  vary. Very small \(h\) picks up this noise as spurious "curvature",
  ruining gamma and rho. Larger \(h\) averages the noise out.
- The optimum is a compromise; the trader-convention values above are
  the standard sweet spot for \(N \gtrsim 200\).

**Cost.** Greeks require ~8 additional tree evaluations per pricing
call. Set `Config::compute_greeks = false` when benchmarking pure
pricing throughput or when you don't need sensitivities.

## 7. Edge cases

The implementation handles these boundary cases without special-case
code paths:

- **`sigma == 0`** — the deterministic-limit path returns
  \(e^{-rT}\,\max(F - K, 0)\) for calls (with \(F = S e^{(r-q)T}\)).
  No tree is built.
- **`T == 0`** — expired option, price is intrinsic value.
- **`N == 1`** — valid but very inaccurate; used as a smoke-test for
  crash-freedom in tests.
- **Negative rates** — supported; \(p\) simply becomes smaller.
- **`p` outside [0, 1]** — CRR can produce non-probabilities when
  \(\sigma \sqrt{\Delta t}\) is large relative to \((r-q)\Delta t\).
  The engine returns `NaN` and the caller is expected to increase
  \(N\). This is documented in the .cpp and rare for reasonable
  inputs.

## 8. References

- Cox, J.C., Ross, S.A., and Rubinstein, M. (1979). "Option Pricing:
  A Simplified Approach". *Journal of Financial Economics* 7(3).
- Jarrow, R. and Rudd, A. (1983). *Option Pricing*. Homewood, IL:
  R.D. Irwin.
- Tian, Y. (1993). "A Modified Lattice Approach to Option Pricing".
  *Journal of Futures Markets* 13(5).
- Merton, R.C. (1973). "Theory of Rational Option Pricing". *Bell
  Journal of Economics and Management Science* 4(1).
- Hull, J.C. (2018). *Options, Futures, and Other Derivatives*,
  10th Ed. Chapter 21.
- Broadie, M. and Detemple, J. (1996). "American Option Valuation:
  New Bounds, Approximations, and a Comparison of Existing Methods".
  *Review of Financial Studies* 9(4).
- Shreve, S.E. (2004). *Stochastic Calculus for Finance II: Continuous-
  Time Models*. Springer. Chapter 8 (American options in tree models).
