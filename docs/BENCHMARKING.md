# Benchmarking Framework

This document describes the `ore::benchmark` module — the cross-engine
timing and accuracy framework that compares `BlackScholesEngine`,
`BinomialTreeEngine`, and `MonteCarloEngine` on a common set of
standardised option-pricing problems.

The framework does **not** implement any new pricing algorithms. It is
an evaluation and reporting layer on top of the existing pricing engines.

---

## 1. Scope

The framework answers four operational questions:

1. **Accuracy.** How far does each engine's price deviate from the
   analytical Black-Scholes reference?
2. **Convergence.** How does that deviation shrink as the resource
   parameter (tree steps, MC paths) grows?
3. **Runtime.** How does per-call latency scale with the same parameter?
4. **Greek accuracy.** How well do numerical (Binomial, MC) Greeks
   track the analytical (BS) Greeks?

Every question is answered on the same 12 benchmark cases, so a single
CSV dump is enough to draw every plot the Python side produces.

---

## 2. Module layout

```
include/ore/benchmark/
    benchmark.hpp              # umbrella
    benchmark_case.hpp         # BenchmarkCase + standard_suite()
    benchmark_runner.hpp       # Runner + Row + Report + memory helper

src/benchmark/
    benchmark_case.cpp
    benchmark_runner.cpp
```

`ore::benchmark` depends on `ore::pricing` and `ore::core`. It does
**not** depend on `ore::analytics` — benchmarking is a peer of
analytics, not a consumer.

---

## 3. Standard benchmark suite

`standard_benchmark_suite()` returns 12 European cases. All cases share
a valuation date of **2026-01-01** and a synthetic underlying (`BENCH`).

| # | Slug | Regime tested | S | K | r | q | σ | T (y) | Type |
|---|------|---------------|---|---|---|---|---|-------|------|
| 1 | `atm_call` | Baseline ATM | 100 | 100 | 5% | 0% | 20% | 1 | Call |
| 2 | `atm_put` | Put/call parity control | 100 | 100 | 5% | 0% | 20% | 1 | Put |
| 3 | `itm_call` | Deep intrinsic | 100 | 80 | 5% | 0% | 20% | 1 | Call |
| 4 | `otm_call` | Tail sensitivity | 100 | 120 | 5% | 0% | 20% | 1 | Call |
| 5 | `itm_put` | ITM put | 100 | 120 | 5% | 0% | 20% | 1 | Put |
| 6 | `otm_put` | OTM put | 100 | 80 | 5% | 0% | 20% | 1 | Put |
| 7 | `short_maturity_call` | Time-discretisation stress | 100 | 100 | 5% | 0% | 20% | 1/12 | Call |
| 8 | `long_maturity_call` | Drift-dominated | 100 | 100 | 5% | 0% | 20% | 5 | Call |
| 9 | `low_vol_call` | Concentrated payoff | 100 | 100 | 5% | 0% | 5% | 1 | Call |
| 10 | `high_vol_call` | Fat-tailed payoff | 100 | 100 | 5% | 0% | 60% | 1 | Call |
| 11 | `dividend_paying_call` | Non-zero yield | 100 | 100 | 5% | 3% | 20% | 1 | Call |
| 12 | `negative_rate_call` | Post-2015 regime | 100 | 100 | −2% | 0% | 20% | 1 | Call |

The slugs are the stable public identifier for each case: they land
verbatim in the CSV `case_name` column and are referenced by name in
the Python plotters.

Every case uses `ExerciseStyle::European` so that `BlackScholesEngine`
can serve as the analytical reference. American-only stress cases will
be added when we ship an analytical American approximation (e.g.
Barone-Adesi-Whaley); until then, mixing them in would silently break
the abs / rel error columns.

---

## 4. Metrics

Each `(engine, case)` pair produces one row in `BenchmarkReport::rows`
with the following columns.

| Column | Semantics |
|---|---|
| `case_name`, `case_description`, `engine_name` | Identifiers. |
| `price` | Fair value reported by the engine. |
| `delta`, `gamma`, `vega`, `theta`, `rho` | From `PricingResult::greeks` — zero if the engine didn't compute them. |
| `runtime_us` | Median wall-clock microseconds over `Config::median_reps` `price()` calls. |
| `iterations` | Copy of `PricingResult::iterations` (empty for BS, `N` for Binomial, sample count for MC). |
| `standard_error` | Copy of `PricingResult::standard_error` (MC only). |
| `ci_95_low`, `ci_95_high` | Copy of `PricingResult::confidence_interval_95`. |
| `reference_price` | Set when a reference engine is present. |
| `absolute_error` | `|price − reference_price|`. |
| `relative_error` | `absolute_error / max(|price|, |reference_price|)` — guarded against near-zero denominators. |
| `estimated_memory_bytes` | See §5. |

Rows that would produce nonsensical values in a column (e.g. BS
`standard_error`) leave the cell empty; pandas parses these as `NaN`.

### Runtime measurement

Runtime is `median-of-K` over `std::chrono::steady_clock`. The default
`K = 3` smooths out routine scheduler jitter without inflating overall
suite runtime. Tests use `K = 1` for perfect determinism. Larger `K`
(9, 11) is appropriate when the runtime column is the primary object
of study — but note that reruns then dominate the total suite time.

### Memory estimate

`estimated_memory_bytes(engine)` is a heuristic — not a true
`getrusage()` reading — designed to reflect the *scaling behaviour* of
each engine's working set:

| Engine | Formula |
|---|---|
| `BlackScholesEngine` | `0` (stateless closed-form) |
| `BinomialTreeEngine` | `3 · (steps + 1) · sizeof(double)` |
| `MonteCarloEngine` | `sizeof(std::mt19937_64) + 64` — independent of paths |

The MC line is the pay-off from choosing Welford's algorithm: the
working set does not grow with path count, so the MC memory column is
*constant* while the runtime column grows linearly. That contrast is
the point.

---

## 5. Accuracy against Black-Scholes

Whenever the input `engines` vector contains a `BlackScholesEngine`
(more precisely, an engine whose `name()` starts with
`Config::reference_engine_prefix`, default `"BlackScholes"`), the
runner:

1. Prices the case with the reference engine once, up front.
2. Fills `reference_price`, `absolute_error`, and `relative_error` on
   every other row for the same case.
3. Sets the reference engine's own row to zero error by definition.

If no engine matches the prefix, all three columns are empty.

The choice of denominator for `relative_error` is deliberate:

$$
\text{rel\_err} = \frac{|p - p^\star|}{\max(|p|, |p^\star|)}
$$

Using the max, rather than `|p^\star|`, avoids the pathological
blow-up when both prices are near zero (deep-OTM contracts). The value
is bounded in [0, 2] and is symmetric in its arguments.

---

## 6. Convergence studies

Two dedicated helpers sweep the resource parameter for a *single* case
and return their own report.

### 6.1 `run_binomial_convergence(case, step_counts)`

Constructs a fresh `BinomialTreeEngine` at each step count in
`step_counts` and prices the given case. Suggested sweep:

```
{ 10, 25, 50, 100, 250, 500, 1000, 2500, 5000 }
```

The `absolute_error` column plotted against `iterations` on a log-log
axis reveals CRR's characteristic **O(1/N) sawtooth convergence** to
Black-Scholes: overall trend $\propto 1/N$ with even-vs-odd `N`
oscillation about it.

### 6.2 `run_monte_carlo_convergence(case, path_counts, seed, antithetic)`

Constructs a fresh `MonteCarloEngine` at each path count. Same seed
across every run so the runs share prefixes of the same random
stream (their errors and standard errors are dependent, but this is
what we want for a plot: it removes the run-to-run noise). Suggested
sweep:

```
{ 1'000, 5'000, 10'000, 50'000, 100'000, 500'000, 1'000'000, 5'000'000 }
```

Plotted on log-log, `standard_error` should show a slope of $-1/2$ and
`absolute_error` should track it (up to the residual sample noise
which itself scales like $1/\sqrt N$).

---

## 7. Runtime scaling

The convergence helpers double as runtime-scaling studies: the
`runtime_us` column of the returned reports is a per-step or per-path
timing table.

Asymptotic expectations:

| Engine | Runtime | Memory |
|---|---|---|
| Black-Scholes | `O(1)` | `O(1)` |
| Binomial (CRR) | `O(N²)` | `O(N)` |
| Monte Carlo | `O(P)` | `O(1)` |

For the runtime plot in Python (`plot_runtime.py`) we recommend a
`log(N) → log(runtime)` scatter; the slope should be `~2` for Binomial
and `~1` for MC.

---

## 8. Greek comparison

The runner does *not* have a separate Greek-comparison helper: the
`greeks.*` columns of the main suite report already carry every
required datum. `plot_greeks.py` pivots the CSV on
`(case_name, engine_name)` and plots the five Greeks side by side per
case.

For MC, `compute_greeks` is off by default in the standard suite so
the runtime column is dominated by pricing, not by 8× bumped runs.
The dedicated `benchmark_all_engines` example enables `compute_greeks`
on both Binomial and MC for the Greek-comparison output.

---

## 9. CSV export

`BenchmarkReport::write_csv(std::ostream&)` produces a single CSV with
the columns listed in §4. Descriptions are always double-quoted; every
numeric column is either a `12.6f`-precision fixed-point number or an
empty cell (`NaN` after pandas parse). Row order matches
`BenchmarkReport::rows`, which the runner guarantees to be case-major:
for each case, every engine's row appears in the order the engines
were passed in.

---

## 10. Python visualisation

Four scripts live in `python/`:

| Script | Consumes | Produces |
|---|---|---|
| `plot_runtime.py` | main suite CSV | Bar chart: median runtime per engine per case. |
| `plot_accuracy.py` | main suite CSV | Bar chart: `absolute_error` per engine per case (log y-axis). |
| `plot_convergence.py` | Binomial and MC convergence CSVs | Log-log convergence plot with reference $1/N$ and $1/\sqrt N$ slopes. |
| `plot_greeks.py` | main suite CSV with Greeks | Grouped bar chart: analytical vs Binomial vs MC Greeks per case. |

All four scripts share a common set of assumptions: pandas + matplotlib
in the environment, and CSV files at the paths passed on the command
line. Output filenames follow the input CSV's stem.

---

## 11. Practical trade-offs

| Aspect | Black-Scholes | Binomial (CRR) | Monte Carlo |
|---|---|---|---|
| Exact closed form | ✓ | — | — |
| American options | ✗ | ✓ | ✗ (this milestone) |
| Path-dependent options | ✗ | limited | ✓ (future) |
| Runtime | O(1), microseconds | O(N²), milliseconds | O(P), milliseconds–seconds |
| Memory | O(1) | O(N) | O(1) (Welford) |
| Convergence rate | exact | O(1/N) with sawtooth | O(1/√P) |
| Statistical noise | none | none | yes — SE column |
| Greek quality | analytical, exact | numerical, mildly noisy | numerical, noisier |

The three engines are complementary, not competing: whenever the
analytical formula applies, use it; when American exercise matters,
use Binomial; when the payoff is path-dependent (a future milestone),
use Monte Carlo. The framework's job is to *quantify* that folk
wisdom, so future engines join the comparison on the same terms.

---

## 12. Limitations

- **Reference bias.** Every abs / rel error column is relative to
  Black-Scholes. When we ship an engine that Black-Scholes can't
  express (barriers, Asians), those rows won't have a reference and
  the columns will be empty.
- **Determinism.** Wall-clock times are inherently noisy on a
  multi-tasking OS. `median_reps` smooths but does not eliminate the
  noise. Use `perf`/`vtune` if you care about *why* an engine is
  slow, not just *that* it is.
- **Memory heuristic.** `estimated_memory_bytes` is a formula, not a
  measurement. It captures the intended working set of each engine
  and *not* short-lived allocator overhead, RNG cache lines shared
  across calls, or PGO-influenced code size.
- **Seed dependence in MC.** The MC convergence sweep uses a shared
  seed across path counts, which makes the errors dependent between
  rows. This is deliberate for plotting (removes cross-run jitter)
  but should be re-thought if the report is ever used for statistical
  hypothesis testing about the MC estimator.

Future milestones (importance sampling, control variates, Sobol
sequences, American Monte Carlo) will slot into the same framework
without changing the report schema.
