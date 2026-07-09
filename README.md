# OptionResearchEngine

A modular C++20 quantitative finance research engine for pricing, analyzing,
and researching options. This is an **academic software engineering
project**, not a trading bot.

> The CMake project is named `OptionResearchEngine`; the compiled library is
> `ore` (aliased `ore::ore`) and the C++ namespace root is `ore::`.

## Status

Core engine complete. The pricing engines (Black-Scholes, binomial tree,
Monte Carlo), the finance-agnostic numerics (root-finding, RNG, normal
distribution), the historical market-data loaders, the analytics layer
(option-chain IV calibration plus volatility smiles / term structure /
surface / skew), and the `ore::research` historical study framework are
all implemented, unit-tested, and driven by the executables under
`examples/`. The most recent milestone wires the analytics layer into the
research framework via `HistoricalCalibrationStudy` — see
[docs/HISTORICAL_RESEARCH.md](docs/HISTORICAL_RESEARCH.md).

## Layout

```
CMakeLists.txt            Top-level CMake project
cmake/                    CMake helper modules (compiler warnings, etc.)
include/ore/              Public C++ headers, one folder per module
    core/                 Financial contract & market value types
    pricing/              Black-Scholes, binomial tree, Monte Carlo, IV solver
    numerics/             Finance-agnostic numerical utilities (solvers, RNG)
    marketdata/           Historical / Yahoo option-chain loaders
    analytics/            Option-chain IV calibration & volatility structures
    research/             Historical study framework (ore::research)
    portfolio/            Portfolio & positions (planned)
    utils/                Cross-cutting utilities (planned)
src/                      C++ implementation files (mirrors include/)
tests/                    GoogleTest unit tests (FetchContent, mirrors include/)
examples/                 Executable examples (opt-in via ORE_BUILD_EXAMPLES)
python/                   Data downloaders, notebooks, plots (Milestone 2+)
data/                     Local market data & engine outputs
    raw/                    Downloaded, unmodified market data
    processed/              Normalised, C++-parser-ready datasets
    generated/              Outputs (vol surfaces, benchmark results)
docs/                     Design docs
```

## Architectural principles

- **Contract vs. underlying vs. market state are three separate types.**
  `Option` holds only contract fields (strike, expiration date, payoff type,
  exercise style, and its `Underlying`). `Underlying` is identity only
  (symbol, exchange, asset type). `MarketSnapshot` holds market inputs
  (spot, rates, yields, valuation date). Time-to-maturity is **derived** at
  pricing time, never stored.
- **Modern C++20 chrono types.** Dates are `std::chrono::year_month_day`;
  timestamps are `std::chrono::system_clock::time_point`. No custom date
  classes.
- **Value semantics for data types.** `Underlying`, `Option`, `Quote`,
  `MarketSnapshot`, `OptionMarketSnapshot`, `PricingResult` are aggregate
  value types — no inheritance, no virtuals, copyable, movable, comparable
  where appropriate.
- **Interfaces are the only virtual types.** `PricingEngine` today; more
  will be added under `marketdata/` and `analytics/` as they materialise.
- **Numerics is independent of finance.** `ore::numerics` (normal
  distribution, root-finding, RNG) never depends on any finance module.
  The dependency graph strictly points *finance → numerics*.
- **Rate & yield conventions:** continuous compounding, decimal
  (i.e. `0.05`, not `5.0`), ACT/365F day count (default).
- **Error handling:** deliberately deferred. Once the CSV/Parquet parser
  lands in Milestone 3, we will choose between exceptions, `std::expected`
  (C++23), or a small `Result<T>` — driven by real code, not a preemptive
  design.

## Build (Windows / MSVC)

Requires:

- **CMake ≥ 3.20** (`winget install Kitware.CMake`)
- **Visual Studio 2022 Build Tools** with the "Desktop development with C++"
  workload (`winget install Microsoft.VisualStudio.2022.BuildTools`)
- Internet access on first configure (GoogleTest is fetched via CMake)

From the repository root, from a **"x64 Native Tools Command Prompt for VS 2022"**
(or after running `vcvars64.bat`):

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

Alternatively, using Ninja (faster, single-config):

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

### CMake options

| Option                | Default | Description                                     |
| --------------------- | ------- | ----------------------------------------------- |
| `ORE_BUILD_TESTS`     | `ON`    | Build the GoogleTest-based unit test target.    |
| `ORE_BUILD_EXAMPLES`  | `OFF`   | Build executable examples under `examples/`.    |

## Roadmap

- **M1** — architecture scaffold, CMake, GoogleTest, interfaces. ✅
- **M2** — market-data downloaders / loaders → files under `data/`. ✅
- **M3** — C++ CSV parser producing `Option` / `Quote` / `OptionMarketSnapshot`
  objects, with unit tests. ✅
- **M4** — Black-Scholes, binomial tree, Monte Carlo, IV solver, option-chain
  calibration, vol smiles / term structure / surface / skew, and the
  `ore::research` historical study framework. ✅ ← *current*
- **M5+** — Greeks surfaces, surface interpolation / parametric fitting,
  delta hedging, portfolio analytics, pybind11 bindings, notebooks.

## Deferred design decisions (tracked, not yet resolved)

- **Error handling model** (exceptions vs. `std::expected` vs. `Result<T>`)
  — chosen alongside the Milestone 3 parser.
- **Analytics sub-organization** — the analytics module will subdivide into
  `statistics/`, `risk/`, `greeks/` etc. Whether `pricing/` stays a
  top-level module or moves under `analytics/pricing/` is left open until
  the first pricing engine lands.
- **On-disk format** for downloaded data — CSV vs. Parquet, decided at M2.
