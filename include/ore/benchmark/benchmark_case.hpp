/**
 * @file benchmark_case.hpp
 * @brief `BenchmarkCase` — a single row of the standardised benchmark
 *        suite, and the factory that produces the canonical 12-case set.
 *
 * A `BenchmarkCase` bundles an option, a market snapshot, and metadata
 * (name, description) that describes what the case exercises. The
 * standard suite covers moneyness (ATM/ITM/OTM × call/put), maturity
 * (short/long), volatility regime (low/high), dividends, and negative
 * rates.
 *
 * The suite is *deterministic*: `standard_benchmark_suite()` always
 * returns the same 12 cases in the same order, so CSV outputs across
 * runs are diffable and reproducible.
 */
#pragma once

#include <string>
#include <vector>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>

namespace ore::benchmark {

/**
 * @brief One benchmark case = one contract + one market snapshot + metadata.
 *
 * The `name` is a snake_case slug that is embedded into CSV rows, so it
 * must remain stable across runs of the standard suite. `description`
 * is free text intended for human consumption in the docs and in the
 * runner's stdout dump.
 */
struct BenchmarkCase {
    std::string name;
    std::string description;
    ore::core::Option         option;
    ore::core::MarketSnapshot market;
};

/**
 * @brief The canonical 12-case benchmark suite.
 *
 * Cases (all with a shared valuation date of 2026-01-01 unless noted):
 *
 *   1. atm_call            — S=K=100, r=5%, q=0, sigma=20%, T=1y
 *   2. atm_put             — put analogue of 1
 *   3. itm_call            — K=80  (deep ITM call)
 *   4. otm_call            — K=120 (deep OTM call)
 *   5. itm_put             — K=120 (deep ITM put)
 *   6. otm_put             — K=80  (deep OTM put)
 *   7. short_maturity_call — T=1 month (~0.0833y)
 *   8. long_maturity_call  — T=5y
 *   9. low_vol_call        — sigma=5% ATM 1y
 *  10. high_vol_call       — sigma=60% ATM 1y
 *  11. dividend_paying_call — q=3% ATM 1y
 *  12. negative_rate_call   — r=-2% ATM 1y
 *
 * Every case uses `ExerciseStyle::European` so `BlackScholesEngine`
 * can serve as an analytical reference. American-only cases are a
 * future addition (they would break the reference contract).
 */
[[nodiscard]] std::vector<BenchmarkCase> standard_benchmark_suite();

} // namespace ore::benchmark
