# Volatility Analytics

This document explains the theoretical background of the volatility
analytics module (`ore::analytics::volatility_analytics`), the design
choices behind the shipped data structures, and the known limitations of
the current implementation.

## The three views of implied volatility

Given a calibrated `CalibrationReport` the module produces three
complementary representations of the same underlying object — the
implied-volatility function \(\sigma(K, T)\).

| View            | Domain                             | Structure               |
|-----------------|-------------------------------------|-------------------------|
| Smile           | strike, at fixed expiration         | `VolatilitySmile`       |
| Term structure  | expiration, at fixed strike (spot)  | `TermStructure`         |
| Surface         | (strike, expiration) grid           | `VolatilitySurface`     |

Each is a *view* of the same population of calibrated contracts; none is
"more correct" than the others. Which one to use depends on the question
being asked:

* *"Are OTM puts expensive today?"* → look at the smile.
* *"Is the market pricing more risk over three months than over one?"* → look at the term structure.
* *"Is my portfolio's vega exposure concentrated?"* → look at the surface.

## Why markets are not flat-volatility

Black-Scholes assumes a single, constant \(\sigma\). Empirically, when
one solves for the \(\sigma\) that reproduces each observed option
price, the result varies with \(K\) and \(T\). The stylised facts are:

1. **Equity index skew.** For SPX, SPY, and similar cap-weighted equity
   indices, low-strike puts trade at *higher* IV than high-strike calls
   ("skew" or "smirk"). Drivers:
     * **Leverage effect** (Christie 1982; Bekaert & Wu 2000). Falling
       equity value increases financial leverage, which mechanically
       raises the volatility of the remaining equity claim.
     * **Crash-o-phobia** (Bates 1991, Rubinstein 1994). Investors pay a
       premium for downside protection since portfolio insurance became
       widespread after the 1987 crash.
     * **Systematic supply/demand** (Bollen & Whaley 2004). Retail
       hedgers are natural buyers of puts and sellers of covered calls,
       creating persistent order-flow pressure.
2. **Currency and commodity smile.** FX and commodity options usually
   show a more *symmetric* smile — high IV on both wings, lower at ATM.
   Drivers are jump risk (currency devaluations, supply shocks) and the
   more symmetric return distribution of these underlyings.
3. **Term structure.** In calm markets the term structure is upward-
   sloping (long-dated vol > short-dated vol), reflecting uncertainty
   compounding with time. In stress it inverts: short-dated vol spikes
   above long-dated vol as the market anticipates near-term realised
   volatility. See Cboe's VIX-vs-VIX9D charts around any major event.
4. **Volatility of volatility.** The smile itself moves and changes
   shape over time (stochastic vol; Heston 1993). A single snapshot
   under-represents the true uncertainty in the market's forward IV.

The upshot is that Black-Scholes is best viewed as a *quoting
convention* — a bijection between prices and a single number per
contract, useful for interpolating and comparing quotes — rather than
a literal generative model of the market.

## Smile construction

For every calibrated contract we compute one point per (strike, type)
pair. When both a call and a put are calibrated at the same strike,
both are retained; a parallel `types` vector lets consumers filter.

### Moneyness definitions

Three conventions are supported (see `enum class Moneyness`):

| Definition   | Formula              | When to use                                   |
|--------------|----------------------|-----------------------------------------------|
| Simple       | \(K/S\)              | Intuitive for humans; not symmetric around ATM |
| LogSimple    | \(\ln(K/S)\)         | **Default.** Symmetric, additive, natural in BS |
| LogForward   | \(\ln(K/F)\), \(F = S e^{(r-q)T}\) | Removes cost of carry; best across terms |

`LogSimple` maps ATM to 0 and gives you a linear axis to draw skew
against. `LogForward` is the correct axis when you want to compare
smiles at different maturities (the drift removes the systematic shift
from carry).

## Skew metrics

For each smile the module computes five diagnostic numbers:

| Metric              | Definition                                                                   |
|---------------------|------------------------------------------------------------------------------|
| **ATM IV**          | IV at `K = spot`, linearly interpolated in strike between OTM points        |
| **25-delta call IV**| IV of the call whose BS delta = 0.25, linearly interpolated in delta space |
| **25-delta put IV** | IV of the put whose BS delta = -0.25, likewise                              |
| **Risk reversal**   | `25d_call_IV - 25d_put_IV`  → measures asymmetric skew                       |
| **Butterfly**       | `0.5 * (25d_call + 25d_put) - ATM_IV` → measures wing convexity              |

**Why 25-delta rather than a fixed strike?** Delta is dimensionless and
comparable across expirations; a fixed \(K\) is not (a 10% OTM call at
1M is a very different beast from a 10% OTM call at 1Y). 25-delta is
the traditional FX-market convention now widespread in equities.

**Interpolation.** We interpolate linearly in delta space between the
two calibrated calls (or puts) that bracket the target delta:

$$
\text{iv}_{25\Delta} = \text{iv}_{a} + \frac{0.25 - \delta_a}{\delta_b - \delta_a}\,(\text{iv}_b - \text{iv}_a)
$$

with \(\delta_a, \delta_b\) the two bracketing deltas (in the sorted
sequence) and \(\text{iv}_a, \text{iv}_b\) the corresponding calibrated
IVs. Delta itself is computed from the calibrated IV of each point
using the Black-Scholes formula from the pricing engine — so no
smoothing or fitting is imposed.

## Term structure construction

The ATM-IV term structure is:

1. Group all calibrated contracts by expiration.
2. For each expiration, build the (strike -> OTM IV) curve using the
   classical single-value-per-strike convention (K < spot → put; else
   call).
3. Linearly interpolate in strike to \(K = \text{spot}\).
4. Missing brackets (spot outside the calibrated strike range) → `NaN`.

The result is a maturities-vs-IV pair of vectors, monotonically
ascending in maturity, that can be plotted as-is.

## Surface construction

The surface is a rectangular grid:

* Rows: expirations, ascending.
* Columns: the sorted union of every strike observed at any expiration.
* Cell \([i][j]\) is the calibrated IV of the OTM contract at
  \((T_i, K_j)\), or `NaN` if no such contract was calibrated.

No interpolation is applied. Cells left `NaN` are exactly the strikes
that were not listed at that expiration in the input chain — this is
the honest representation of the raw data. Interpolation across the
surface belongs to the *next* milestone (arbitrage-free smile fitting,
SVI, or non-parametric splines are all viable candidates).

## Statistics

`compute_statistics` returns nine numbers over any vector of IVs (or
subset of the report):

* `count`, `min`, `max`
* `mean`, `median`, `stddev` (population, `n` divisor)
* `p10`, `p25`, `p75`, `p90`

Percentiles use numpy's linear-interpolation method:

$$
p\text{-th percentile} = \text{sorted}[\lfloor r \rfloor] + \{r\}\,(\text{sorted}[\lceil r \rceil] - \text{sorted}[\lfloor r \rfloor]),\quad r = \frac{p}{100}(n-1)
$$

This guarantees the returned percentile lies in \([\min, \max]\) and
agrees with `numpy.percentile(default)` and pandas `.quantile()` for
any input.

Non-finite values are silently dropped before computation. An empty
input returns all-zeros.

## CSV format

All four exporters emit *long-format* CSV, so pandas can `pivot()` to
recover the natural wide form without any manual reshaping.

| Exporter    | Columns                                                                                              |
|-------------|------------------------------------------------------------------------------------------------------|
| Smile       | `expiration, time_to_expiry, strike, option_type, moneyness_convention, moneyness, implied_volatility` |
| Term        | `expiration, time_to_expiry, atm_iv`                                                                 |
| Surface     | `expiration, time_to_expiry, strike, implied_volatility`                                             |
| Skew        | `expiration, time_to_expiry, atm_iv, call_25delta_iv, put_25delta_iv, risk_reversal, butterfly`       |

Numeric fields use `%.17g` (round-trip-preserving IEEE 754). Missing
optional values are empty fields (pandas reads them as `NaN`). Dates
are ISO 8601 (`YYYY-MM-DD`), matching the loader.

## Limitations

This milestone deliberately excludes:

1. **No interpolation of the surface.** If the input chain does not
   list \(K = 105\) at \(T = 3\text{M}\), the surface cell is `NaN`.
   The next milestone owns interpolation.
2. **No arbitrage checks on the fitted surface.** The individual per-
   option calibration in Milestone 6 checks BS arbitrage bounds; there
   is no cross-strike or cross-expiration no-arbitrage enforcement here
   (no calendar-spread test, no butterfly-inequality test).
3. **No parametric fitting.** No SVI, SABR, Heston, or similar. The
   output is empirical: it reports what the market said, not a smoothed
   or de-noised interpretation.
4. **No local or stochastic volatility.** \(\sigma(K, T)\) here is the
   *implied* volatility surface — the input to those models, not those
   models' outputs.
5. **Provider-IV comparison lives in Milestone 6.** This module works
   only from our own computed IV, so any provider-vs-us skew analysis
   must be done against `CalibrationReport::results` directly.
6. **Single snapshot.** Historical / time-series analytics (realised
   vs. implied vol, term-structure regime detection) are not part of
   this milestone.

## References

* Black, F. and Scholes, M. (1973). "The Pricing of Options and
  Corporate Liabilities". *Journal of Political Economy* 81(3).
* Hull, J. (2018). *Options, Futures, and Other Derivatives*, 10th Ed.
  Ch. 20 covers smile and term-structure conventions.
* Rebonato, R. and McKay, K. (2009). *The SABR/LIBOR Market Model*.
  Broader treatment of smile fitting.
* Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*.
  The standard reference for how the surface behaves empirically.
* Christie, A. (1982). "The Stochastic Behavior of Common Stock
  Variances: Value, Leverage and Interest Rate Effects". *Journal of
  Financial Economics* 10.
* Bates, D. (1991). "The Crash of '87: Was It Expected? The Evidence
  from Options Markets". *Journal of Finance* 46.
* Rubinstein, M. (1994). "Implied Binomial Trees". *Journal of Finance*
  49.
* Bollen, N. and Whaley, R. (2004). "Does Net Buying Pressure Affect
  the Shape of Implied Volatility Functions?". *Journal of Finance* 59.
* Heston, S. (1993). "A Closed-Form Solution for Options with
  Stochastic Volatility with Applications to Bond and Currency
  Options". *Review of Financial Studies* 6.
