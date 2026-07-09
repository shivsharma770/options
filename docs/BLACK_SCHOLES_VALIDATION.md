# Black-Scholes-Merton — validation catalog

This document records every numeric benchmark that
`ore::pricing::BlackScholesEngine` is validated against, along with the
source of each expected value. The goal is transparency: any reader can
look up the source, reproduce the reference computation independently,
and confirm we aren't marking our own homework.

> All benchmarks live in `tests/pricing/test_black_scholes.cpp`. See the
> "Validation philosophy" note in `tests/pricing/README.md` for the
> reasoning behind what we do and don't validate against.

## 1. Conventions

Every case below is stated in the *engine's* natural units, which are
also the units documented in `include/ore/core/market_snapshot.hpp` and
`include/ore/pricing/greeks.hpp`:

- Rates and yields: continuously compounded, decimal (`0.05`, not `5.0`).
- Volatility: annualized standard deviation of log-returns, decimal
  (`0.20`, not `20.0`).
- Time to expiry: years on an ACT/365 fixed basis.
- Delta: `dV/dS`, unitless.
- Gamma: `d^2 V / dS^2`, per share per share.
- Vega: `dV/d sigma` per unit of vol (i.e. per `1.0` = 100 vol points).
  Traders' `Vega per 1% vol` is our `vega * 0.01`.
- Theta: `dV/dt` per year. Traders' `Theta per day` is our `theta / 365`.
- Rho: `dV/dr` per unit of rate. Traders' `Rho per bp` is our
  `rho * 1e-4`.

## 2. Numeric benchmarks

### Case A — Hull, Example 15.6

Source: Hull, J. (2018), *Options, Futures, and Other Derivatives*,
10th Edition, Prentice Hall, Example 15.6, p. 342.

| Parameter | Value |
|---|---|
| Spot `S`             | 42.0 |
| Strike `K`           | 40.0 |
| Risk-free rate `r`   | 0.10 |
| Dividend yield `q`   | 0.0 |
| Volatility `sigma`   | 0.20 |
| Time to expiry `T`   | 0.5 years |
| Type                 | Call |

Expected: **`Call = 4.759`** (Hull's rounded 3-decimal value).

Pinned in `HullExample_15_6_Call` with tolerance `5e-3` (matches Hull's
own precision).

### Case B — Canonical ATM, no dividends

Source: derived from the closed form using known high-precision values
for the standard normal CDF. Cross-referenced against QuantLib 1.31's
`AnalyticEuropeanEngine` and R's `RQuantLib::EuropeanOption` (both
independently used the same closed-form reference implementation).

| Parameter | Value |
|---|---|
| Spot `S`           | 100.0 |
| Strike `K`         | 100.0 |
| Risk-free rate `r` | 0.05 |
| Dividend yield `q` | 0.0 |
| Volatility `sigma` | 0.20 |
| Time to expiry `T` | 1.0 years |

Reference constants used in the derivation:

| Symbol | Value | Source |
|---|---|---|
| `d1`             | `0.35` | Exact by construction. |
| `d2`             | `0.15` | Exact by construction. |
| `Phi(0.35)`      | `0.6368306511756191` | `scipy.stats.norm.cdf(0.35)`. |
| `Phi(0.15)`      | `0.5596176923702426` | `scipy.stats.norm.cdf(0.15)`. |
| `phi(0.35)`      | `0.3752403469169692` | `scipy.stats.norm.pdf(0.35)`. |
| `exp(-0.05)`     | `0.9512294245007140` | `math.exp(-0.05)` (IEEE-754). |

Expected prices and Greeks:

| Quantity | Call | Put | Formula |
|---|---|---|---|
| Price   | `10.4506`     | `5.5735`      | `S Phi(d1) - K exp(-rT) Phi(d2)` / `K exp(-rT) Phi(-d2) - S Phi(-d1)` |
| Delta   | `0.63683065`  | `-0.36316935` | `Phi(d1)` / `Phi(d1) - 1` |
| Gamma   | `0.01876202`  | `0.01876202`  | `phi(d1) / (S sigma sqrt(T))` |
| Vega    | `37.52403`    | `37.52403`    | `S phi(d1) sqrt(T)` |
| Theta   | `-6.41403`    | `-1.65790`    | See engine .cpp for the two-term formula. |
| Rho     | `53.23248`    | `-41.89046`   | `KT exp(-rT) Phi(d2)` / `-KT exp(-rT) Phi(-d2)` |

Pinned in `BlackScholesGreeksTest::CanonicalCall_AllFive` and
`::CanonicalPut_AllFive`. Delta, Gamma, and Vega are pinned to 1e-8 or
better because they involve a single `Phi` or `phi` evaluation; Theta
and Rho to 1e-4 because they involve products of three high-precision
constants that only fit ~10 decimal digits of accuracy in the printed
literals used here.

### Case C — Deep ITM call, dividends off

| Parameter | Value |
|---|---|
| Spot `S`           | 500.0 |
| Strike `K`         | 100.0 |
| Risk-free rate `r` | 0.05 |
| Dividend yield `q` | 0.0 |
| Volatility `sigma` | 0.20 |
| Time to expiry `T` | 1.0 |
| Type               | Call |

Expected: `price ≈ S * exp(-qT) - K * exp(-rT) = 500 - 95.1229... = 404.877...`

Rationale: `d1 ≈ 8.4`, `d2 ≈ 8.2`. `Phi(8.2) = Phi(8.4) = 1.0` to
double precision. The call reduces to its Black-Scholes lower bound.
Same holds for `Delta = e^{-qT}`.

Pinned in `BlackScholesEdgeCaseTest::DeepITMCall_ApproachesStockMinusDiscountedStrike`
with `1e-6` tolerance.

### Case D — Deep OTM put

Symmetric: `S = 500, K = 100, ..., type = Put`. Expected `price ≈ 0`,
`Delta ≈ 0`. Pinned in
`BlackScholesEdgeCaseTest::DeepOTMPut_ApproachesZero`.

## 3. Non-numeric benchmarks (analytic identities)

These identities hold **exactly** (to machine precision) for any correct
implementation of Black-Scholes, regardless of the specific numeric
inputs. Failing any of them is a model-independent bug indicator.

### 3.1 Put-call parity

$$ C - P = S e^{-qT} - K e^{-rT} $$

Verified over a `3 x 3 x 3 x 2 x 2 x 3 = 324`-point grid of `(S, K, r, q,
sigma, T)` tuples in `BlackScholesIdentitiesTest::PutCallParity`.
Tolerance: `1e-12`.

### 3.2 Delta identity

$$ \Delta_C - \Delta_P = e^{-qT} $$

Verified across a `(q, T)` grid in `CallPutDeltaRelationship`.
Tolerance: `1e-14`.

### 3.3 Gamma equality

$$ \Gamma_C = \Gamma_P $$

Verified across spot values in `GammaEquality`. Tolerance: bit-exact
(`EXPECT_DOUBLE_EQ`).

### 3.4 Vega equality

$$ \mathcal{V}_C = \mathcal{V}_P $$

Verified across volatility values in `VegaEquality`. Tolerance:
bit-exact (`EXPECT_DOUBLE_EQ`).

### 3.5 Rho identity

$$ \rho_C - \rho_P = K T e^{-rT} $$

Verified across rate values in `RhoIdentity`. Tolerance: `1e-10`.

## 4. Finite-difference Greeks validation

Each analytical Greek is compared against a central finite-difference
estimate over the entire pricing pipeline. This is one of the strongest
correctness checks: agreement to `~1e-6` relative implies that every
component (formula, `NormalDistribution::cdf`, `NormalDistribution::pdf`,
`std::exp`, `std::log`) is internally consistent.

| Greek | FD bump size `h`      | Order | Tolerance (abs / rel) |
|---|---|---|---|
| Delta | `1e-3 * S`            | First-order central | `1e-6 / 1e-5` |
| Gamma | `1e-2 * S`            | Second-order central | `1e-6 / 1e-4` |
| Vega  | `1e-4` (absolute vol) | First-order central | `1e-4 / 1e-5` |
| Theta | `1e-4` (absolute T)   | First-order central, sign-flipped | `1e-4 / 1e-4` |
| Rho   | `1e-5` (absolute rate)| First-order central | `1e-4 / 1e-5` |

`Gamma` uses a larger `h` because second-order differences amplify
rounding. `Vega`, `Theta`, `Rho` use absolute bumps because those inputs
are already O(1).

Tests in `BlackScholesFiniteDiffTest` sweep spot, volatility, time-to-
expiry, and rate values across a small grid for each Greek.

## 5. Recommended additional reference sources (for future milestones)

None of these are currently pinned, but they're the sources we should
compare against as more engines are added:

- **QuantLib** — `QuantLib/test-suite/europeanoption.cpp`. Approximately
  50 vetted numeric test cases with pre-computed expected values.
- **Hull's OFOD10e.xls** — the reference Excel workbook Hull publishes
  alongside the textbook, with hundreds of test rows.
- **Wilmott, P.** (2007), *Paul Wilmott on Quantitative Finance*, 2nd
  Edition, Wiley. Volume 1, Chapter 8 includes the Greeks fully
  tabulated for several worked examples.
- **Haug, E. G.** (2007), *The Complete Guide to Option Pricing
  Formulas*, 2nd Edition, McGraw-Hill. Chapter 1 has closed-form
  worked cases including with continuous dividends and cost-of-carry
  variations.

## 6. Reproducing the reference values

Every reference `Phi` and `phi` value in this document was produced by
`scipy.stats.norm.{cdf,pdf}` at machine precision. The equivalent R
computation is `pnorm(x)` and `dnorm(x)`. The closed-form Black-Scholes
values can be reproduced in one line:

```python
from scipy.stats import norm
from math import exp, log, sqrt

def bs_call(S, K, r, q, sigma, T):
    d1 = (log(S/K) + (r - q + sigma*sigma/2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * exp(-q*T) * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)

print(bs_call(100, 100, 0.05, 0.0, 0.20, 1.0))
# 10.450583572185565
```

Or in R:

```r
library(RQuantLib)
EuropeanOption(type = "call", underlying = 100, strike = 100,
               dividendYield = 0.0, riskFreeRate = 0.05,
               maturity = 1.0, volatility = 0.20)$value
# [1] 10.45058
```

If a future refactor breaks one of the pinned values, running either
snippet in a fresh interpreter is enough to confirm whether the
regression is on our side or in the reference.
