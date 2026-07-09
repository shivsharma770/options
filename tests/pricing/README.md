# Pricing test suite — validation philosophy

This directory is empty for now. When the first pricing engine (Black-Scholes)
lands here, its tests must follow the principles below. This note exists so
we do not accidentally end up validating our own bugs against downloaded
market data.

## What pricing tests validate against

Pricing engines are validated **against known analytic answers**, not
against observed market prices. Concretely:

1. **Textbook values.** For each engine, the test suite pins numeric outputs
   to canonical worked examples from published references — e.g. Hull,
   *Options, Futures, and Other Derivatives*, chapter examples; the
   Wilmott books; the AS-247/AS-241 test tables from the *Journal of the
   Royal Statistical Society*. Every test that pins a numeric value must
   cite the source.

2. **Put-call parity.**
   \f[ C - P = S e^{-qT} - K e^{-rT} \f]
   For any Black-Scholes / binomial / Monte-Carlo run, computing both a
   call and a put on the same input and checking parity is a
   model-independent sanity test. Deviations bigger than a few ULP
   indicate a bug.

3. **Known Greeks identities.**
   - \f$ \Delta_C - \Delta_P = e^{-qT} \f$
   - Deep-ITM European call: \f$ \Delta \to e^{-qT} \f$, \f$ \Gamma \to 0 \f$
   - At expiry: \f$ V \to \max(0, S - K) \f$ for a call, \f$ \max(0, K - S) \f$ for a put
   - Gamma is symmetric in moneyness: \f$ \Gamma(S, K) = \Gamma(K, S) \f$
     when \f$ q = 0 \f$ and \f$ r = 0 \f$.
   Each identity is a free correctness test that every engine must satisfy.

4. **Cross-engine agreement.** Once we have more than one engine, they must
   agree on European vanillas to within their respective error bounds:
   binomial(N → ∞) → Black-Scholes; Monte Carlo(paths → ∞) → Black-Scholes;
   etc.

5. **Published benchmark cases.** For engines with well-known reference
   values (e.g. binomial tree tests from Hull's textbook, or the
   Andersen-Broadie American-option benchmarks), pin those exact values.

## What pricing tests do NOT validate against

- **Yahoo Finance / Bloomberg / market quotes.** Market prices reflect
  supply, demand, order-book microstructure, dividends, hard-to-borrow
  cost, and a dozen other frictions our engines deliberately ignore. A
  Black-Scholes price will not match the market mid, and a mismatch is
  not a bug. Market data belongs in *research* notebooks, not in the
  correctness test suite.

- **Regressions against our own past runs.** Golden-file comparisons that
  pin whatever the engine happens to output are worse than useless — they
  freeze bugs into the reference. Only pin values that have an
  independent source.

## Numerical tolerances

Use `ore::numerics::approximately_equal` for every floating-point
comparison, with tolerances chosen deliberately per test:

- **Analytic identities** (parity, symmetry): `abs_tol = 1e-12`,
  `rel_tol = 1e-10`. Anything looser and a bug can slip through.
- **Textbook values** (typically quoted to 4-6 decimals in Hull etc.):
  `abs_tol = 1e-4` is usually enough.
- **Convergence tests** (binomial N → ∞, MC paths → ∞): rate rather than
  absolute value — halve N, expect error to at most halve (binomial) or
  scale by \f$ \sqrt{2} \f$ (MC).

## Structure

Once populated, this directory should mirror `include/ore/pricing/`:

```text
tests/pricing/
    test_black_scholes.cpp         # BS closed-form: textbook values, parity, Greeks
    test_binomial.cpp              # Binomial: convergence to BS, early-exercise cases
    test_monte_carlo.cpp           # MC: BS agreement, SE scaling, seed reproducibility
```

Fixture data (input CSVs) should be small, hand-written, and cite the source
in a header comment.
