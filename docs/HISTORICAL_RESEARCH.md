# Historical Research Framework

This document describes the `ore::research` module: the reusable
infrastructure the project uses to run quantitative studies across
the historical SPY options archive. The framework is deliberately
generic. Pricing, calibration, and analytics milestones each
introduced *new algorithms*; this milestone introduces *the
machinery every future algorithm will run on top of*.

## Contents

- [Design goals](#design-goals)
- [Directory conventions](#directory-conventions)
- [Architecture](#architecture)
- [Execution flow](#execution-flow)
- [Writing a new study](#writing-a-new-study)
- [Built-in studies](#built-in-studies)
- [CSV schemas](#csv-schemas)
- [Parallel execution model](#parallel-execution-model)
- [Performance notes](#performance-notes)
- [Limitations](#limitations)

## Design goals

1. **Framework, not algorithm.** The engine has no opinion about
   pricing, IV solving, or Greeks. Every study encapsulates the
   analytics it needs; the framework only supplies iteration,
   scheduling, timing, error trapping, and CSV export.
2. **Zero-touch reuse.** Every future study should be a subclass of
   `ResearchStudy` and nothing else. No changes to the engine, the
   loader, or the dataset are required to add a new study.
3. **No hardcoded paths.** Every output path flows through
   `HistoricalResearchEngine::Config::output_dir` (default
   `data/generated/research/`). Every study filename flows through
   its own `Config` (default `<study>_<slug>.csv`).
4. **Deterministic parallelism.** `threads = 1` and `threads = N`
   must produce numerically identical results for a well-behaved
   study. The framework is designed around that invariant.
5. **CSV-first outputs.** Every study writes ISO-8601 dates,
   `%.17g` floats, and empty fields for missing optionals — the
   Python plotters, notebooks, and downstream tooling all consume
   the CSVs directly.

## Directory conventions

```
data/
├── historical/
│   └── spy/
│       ├── spy_eod_2010-*/
│       ├── spy_eod_2011-*/
│       ├── ...
│       └── spy_eod_2021-*/
│
└── generated/
    ├── calibration/
    ├── research/          # every ResearchStudy CSV lands here by default
    ├── benchmarks/
    └── plots/
```

`HistoricalLoader::load_from_eod(root, ticker)` recursively scans
`root` for any file with a `.txt` or `.csv` extension. The
`spy_eod_*` directory names are a convention of the upstream
distribution — the loader itself does not pattern-match on them,
so a future `qqq_eod_2022-*` directory drops in without a code
change.

## Architecture

```
+-----------------------------+
| HistoricalDataset           |
|   HistoricalSnapshot[...]   |
+-------------+---------------+
              |
              v
+-----------------------------+     +---------------------------+
| HistoricalResearchEngine    |<--->| ResearchStudy (abstract)  |
|   - Config: threads,        |     |   - begin(ctx)            |
|     output_dir, progress,   |     |   - process(ctx)          |
|     continue_on_error       |     |   - end(ctx, report)      |
|   - run(study) -> Report    |     |   - clone() / merge()     |
+-------------+---------------+     +---------------------------+
              |
              v
+-----------------------------+
| ResearchContext (per day)   |
|   snapshot, chain, market,  |
|   underlying, date, spot,   |
|   day_index, total_days,    |
|   output_dir                |
+-----------------------------+
```

Every module in the diagram lives in `ore::research`. The engine
depends on `ore::marketdata` (for the dataset) and, by convention,
studies depend on `ore::pricing` and `ore::numerics`; the
framework itself pulls in nothing beyond `ore::marketdata` and
`std::thread`.

## Execution flow

For a single-threaded run:

1. `engine.run(study)` creates a fresh `ResearchReport`.
2. `output_dir` is created (recursively) if it does not exist.
3. `study.begin(ctx0)` is called once — `ctx0` refers to the first
   day the engine will dispatch.
4. For each snapshot in ascending date order:
   1. A `ResearchContext` referring to that snapshot is stack-
      allocated.
   2. `study.process(ctx)` is called. Exceptions are trapped into
      `report.errors` unless `Config::continue_on_error` is `false`.
   3. `report.processed_days` is incremented and the progress
      callback is invoked (if set).
5. `study.end(ctx_last, report)` is called. The study writes CSVs
   (via `ResearchCsvWriter`), fills in `report.processed_contracts`
   / `skipped_contracts`, and appends paths to
   `report.generated_files`.
6. `report.runtime_seconds` is populated and the report is returned.

For a parallel run see [Parallel execution model](#parallel-execution-model).

## Writing a new study

The minimum contract is:

```cpp
class VegaWeightedAtmVolStudy final : public ore::research::ResearchStudy {
public:
    std::string_view name() const noexcept override {
        return "VegaWeightedAtmVolStudy";
    }

    void process(const ore::research::ResearchContext& ctx) override {
        for (const auto& oms : ctx.chain().options()) {
            // ...compute whatever the study wants...
        }
    }

    void end(const ore::research::ResearchContext& ctx,
             ore::research::ResearchReport& report) override
    {
        auto path = ctx.output_dir() / "vega_weighted_atm_vol.csv";
        ore::research::ResearchCsvWriter w(path,
            {"date", "atm_vol", "weighted_atm_vol"});
        // ...write rows...
        report.generated_files.push_back(path);
    }
};
```

That's the whole surface. Every optional hook — `begin`, `clone`,
`merge` — has a sensible default.

To support parallel execution:

```cpp
std::unique_ptr<ResearchStudy> clone() const override {
    return std::make_unique<VegaWeightedAtmVolStudy>(config_);
}

void merge(const ResearchStudy& other) override {
    const auto* rhs = dynamic_cast<const VegaWeightedAtmVolStudy*>(&other);
    if (!rhs) return;
    rows_.insert(rows_.end(), rhs->rows_.begin(), rhs->rows_.end());
}
```

## Built-in studies

Four studies ship in the framework. Each is a working example of
the framework's idioms.

### `IVValidationStudy`

For every viable contract:

1. `mid = (bid + ask) / 2`.
2. Recover `sigma` such that `BS(sigma) = mid` via
   `ImpliedVolatilitySolver`.
3. Compare against `Quote::implied_volatility`.

Reports mean / median / RMSE / worst absolute error and
convergence rate.

### `GreeksValidationStudy`

For every contract with a full set of provider Greeks and a
positive provider IV:

1. Price with `BlackScholesEngine` at `sigma = provider_iv`.
2. Convert BS Greeks to vendor units (theta / 365, vega × 0.01).
3. Compare against the provider's Greeks.

Reports MAE, RMSE, and 50/95/99th-percentile absolute errors per
Greek.

### `PricingValidationStudy`

Round-trip test: `mid → IV solver → BS → repriced → residual`. Any
residual larger than the IV solver's tolerance flags a numerical
inconsistency between the pricing engine and the IV solver.

Reports mean absolute residual, RMSE, worst residual, and
fraction of contracts within tolerance.

### `HistoricalCalibrationStudy`

The bridge between `ore::analytics` and the research framework: it
runs the *existing* `OptionChainCalibrator` and the
`volatility_analytics` free functions (`build_smiles`,
`build_term_structure`, `build_surface`, `compute_skew_metrics`)
over every trading day and flattens their per-day outputs into
long-format, date-tagged CSVs. It adds no new pricing model, IV
solver, or interpolation — it only supplies the "iterate the
archive, aggregate, export the time series" glue those single-day
analytics could not do on their own.

Each `process(ctx)` calibrates today's chain, builds the four
volatility structures over the resulting `CalibrationReport`, and
appends compact per-day rows. `end()` writes five CSVs —
`calibration.csv`, `smiles.csv`, `term_structure.csv`,
`surface.csv`, `skew.csv` — each toggleable via `Config`. Reports
days processed, contracts calibrated / skipped / failed, mean
convergence rate, and mean provider-vs-computed `|IV error|`.

Run it end-to-end with `examples/historical_calibration.cpp`
(`example_historical_calibration`).

## CSV schemas

Every study emits ISO-8601 dates and `%.17g` floating-point
numbers. Missing optionals are empty fields — which pandas parses
as `NaN`, exactly what every downstream analysis wants.

### `iv_validation.csv`

```
date,expiration,strike,option_type,mid_price,provider_iv,
computed_iv,absolute_error,relative_error,iterations,
solver_method,converged
```

- `option_type` ∈ {`Call`, `Put`}
- `solver_method` ∈ {`Newton`, `Bisection`}
- `converged` ∈ {`0`, `1`}
- `computed_iv`, `absolute_error`, `relative_error` are empty on
  non-convergent solves.

### `greeks_validation.csv`

```
date,expiration,strike,option_type,provider_iv,
provider_delta,computed_delta,delta_error,
provider_gamma,computed_gamma,gamma_error,
provider_theta,computed_theta,theta_error,
provider_vega,computed_vega,vega_error,
provider_rho,computed_rho,rho_error
```

Individual Greek triples are empty if the vendor did not publish
that Greek. `computed_*` values are in the vendor's unit convention.

### `pricing_validation.csv`

```
date,expiration,strike,option_type,mid_price,
recovered_iv,repriced,residual,converged
```

`recovered_iv`, `repriced`, and `residual` are empty on non-
convergent solves.

### `HistoricalCalibrationStudy` outputs

Every file leads with a `date` column so pandas can `groupby('date')`
without post-processing. All are long-format.

#### `calibration.csv`

```
date,expiration,strike,option_type,bid,ask,last,mid_price,
provider_iv,computed_iv,absolute_error,relative_error,
solver_status,iterations,used_bisection,solver_residual,skip_reason
```

- `provider_iv` / `computed_iv` / `absolute_error` / `relative_error`
  are empty when unavailable (skipped or non-convergent).
- By default only calibrated contracts are written; set
  `Config::include_skipped_in_calibration` to keep skipped/failed
  rows for skip-reason diagnostics.

#### `smiles.csv`

```
date,expiration,time_to_expiry,strike,option_type,
moneyness_convention,moneyness,implied_volatility
```

`moneyness_convention` ∈ {`K/S`, `ln(K/S)`, `ln(K/F)`} per
`Config::moneyness_convention`.

#### `term_structure.csv`

```
date,expiration,time_to_expiry,atm_iv
```

`atm_iv` is empty when no calibrated strikes bracket spot at that
expiration (pandas reads it as `NaN`).

#### `surface.csv`

```
date,expiration,time_to_expiry,strike,implied_volatility
```

`implied_volatility` is empty for uncalibrated grid cells — the
surface is *not* interpolated. `df.pivot(index='expiration',
columns='strike', values='implied_volatility')` recovers the 2D grid
for one date.

#### `skew.csv`

```
date,expiration,time_to_expiry,atm_iv,call_25delta_iv,
put_25delta_iv,risk_reversal,butterfly
```

Any metric is empty when the chain lacks the strikes to compute it.

## Parallel execution model

`Config::threads = N` (for `N > 1`) causes the engine to:

1. Call `study.begin(ctx0)` once on the primary study.
2. Partition the dataset into `N` contiguous, ascending date
   ranges. Chunk `i` receives every snapshot in `[low_i, high_i)`.
3. Spawn `N` worker threads. Each worker:
   1. Calls `study.clone()` to obtain a fresh per-thread study
      instance.
   2. Processes its chunk in ascending date order.
4. `.join()` every worker.
5. Merge every clone back into the primary in ascending worker
   order (`0, 1, ..., N-1`) via `study.merge(*clone)`.
6. Call `study.end(ctx_last, report)` on the primary.

**Determinism.** Provided the study's `merge` implementation is
order-preserving on its accumulator (or, equivalently, produces an
order-independent aggregate), the parallel run is numerically
identical to a single-threaded run over the same dataset. All
four built-in studies satisfy this property:

- `rows_` is appended-to in ascending-date order by every worker;
  merging in worker order produces the same vector the serial run
  would have produced.
- Aggregate statistics (`mean_absolute_error`, `rmse`, ...) are
  rebuilt from `rows_` in `end()`, so they inherit determinism
  from the vector.

**Fallback.** If a study returns `nullptr` from `clone()`, the
engine falls back to serial execution silently. This is the safe
default: studies that have not been designed for parallelism (or
that rely on iteration order in their accumulator) will still
work — just single-threaded.

**Progress callback.** Under parallel execution, the callback is
serialised through an internal mutex. It is safe to call any
non-reentrant code from the callback, but frequent calls will
contend on the mutex.

## Performance notes

- **`std::span` for `between()`.**
  `HistoricalDataset::between(start, end)` returns a zero-copy
  `std::span`. Studies that iterate over a date range should
  prefer `between()` to `filter()` (which copies).
- **Avoid copying option chains.**
  `HistoricalSnapshot` and `OptionChain` are movable but copies
  are not cheap. Studies that need to mutate a chain locally
  should make one working vector per invocation, not per row.
- **Locking cost.**
  The built-in studies fold per-day results into their accumulator
  under a mutex. The critical section is a single
  `std::vector::insert` per day; it is dwarfed by the per-day
  computation. Studies that want to squeeze out the last bit of
  parallel scaling can push per-thread accumulators onto a
  thread-local structure and merge only in `merge()`.

## Limitations

- The current EOD parser assumes the OptionsDX / Delta-Neutral
  archive schema. Other archives (Cboe LiveVol, OptionMetrics) can
  be added by writing a sibling loader that emits the same
  `HistoricalSnapshot` shape — the research framework itself does
  not need to change.
- Risk-free rates and dividend yields are configured once per
  load, not per date. When historical rate/dividend curves are
  wired in, they will land at the loader layer.
- The framework does not attempt to fan out across machines. When
  a run gets large enough to need that, the natural boundary is
  by date-range: the driver can run `HistoricalResearchEngine`
  independently on disjoint date ranges and merge the CSVs
  downstream.
