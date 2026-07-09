# Executable examples for the OptionResearchEngine project.
#
# Only compiled when configured with `-DORE_BUILD_EXAMPLES=ON`.

# --- Binomial-tree convergence benchmark ------------------------------------
#
# Prints a Steps / Tree Price / BS Price / Abs Error / Runtime table for
# an ATM European call, demonstrating CRR's sawtooth convergence to
# Black-Scholes. See docs/BINOMIAL_TREE.md and the source file for
# details.
add_executable(example_binomial_convergence binomial_convergence.cpp)
target_link_libraries(example_binomial_convergence PRIVATE ore::ore)
ore_set_warnings(example_binomial_convergence)

# --- Monte Carlo convergence benchmark --------------------------------------
#
# Prints a Paths / Price / BS Error / Std Error / 95% CI / Runtime table
# for an ATM European call, demonstrating the O(1/sqrt(N)) convergence
# of Monte Carlo estimates. See docs/MONTE_CARLO.md.
add_executable(example_monte_carlo_convergence monte_carlo_convergence.cpp)
target_link_libraries(example_monte_carlo_convergence PRIVATE ore::ore)
ore_set_warnings(example_monte_carlo_convergence)

# --- End-to-end benchmark suite driver --------------------------------------
#
# Runs the standard 12-case suite across every pricing engine and the two
# convergence sweeps (Binomial and Monte Carlo). Writes three CSV files to
# the current working directory and prints a compact summary to stdout.
# The Python plotters in `python/` consume the CSVs directly. See
# docs/BENCHMARKING.md.
add_executable(example_benchmark_all_engines benchmark_all_engines.cpp)
target_link_libraries(example_benchmark_all_engines PRIVATE ore::ore)
ore_set_warnings(example_benchmark_all_engines)

# --- Historical dataset loader ---------------------------------------------
#
# Reads a ticker's historical option-chain tree from
# `data/historical/raw/<TICKER>/` and prints a summary of the loaded dataset
# (trading days, contracts, date range, missing dates, parse failures).
# Runs in lenient mode: malformed days are recorded but do not abort.
# See docs/HISTORICAL_DATA.md.
add_executable(example_load_historical_dataset load_historical_dataset.cpp)
target_link_libraries(example_load_historical_dataset PRIVATE ore::ore)
ore_set_warnings(example_load_historical_dataset)

# --- Historical research engine driver -------------------------------------
#
# Runs the three built-in research studies (`IVValidationStudy`,
# `GreeksValidationStudy`, `PricingValidationStudy`) against a
# multi-year OptionsDX / Delta-Neutral SPY archive stored under
# `data/historical/spy/`. Outputs land in `data/generated/research/`.
# See docs/HISTORICAL_RESEARCH.md.
add_executable(example_historical_research historical_research.cpp)
target_link_libraries(example_historical_research PRIVATE ore::ore)
ore_set_warnings(example_historical_research)

# --- Historical calibration + volatility-structure exporter ---------------
#
# Drives `HistoricalCalibrationStudy` over the SPY archive, exporting
# per-day calibration diagnostics plus long-format volatility smiles,
# term structure, surface, and 25-delta skew CSVs to
# `data/generated/research/`. The Python plotters in `python/` consume
# these directly after adding a `date` filter. See
# docs/HISTORICAL_RESEARCH.md.
add_executable(example_historical_calibration historical_calibration.cpp)
target_link_libraries(example_historical_calibration PRIVATE ore::ore)
ore_set_warnings(example_historical_calibration)
