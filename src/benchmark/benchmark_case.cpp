#include <ore/benchmark/benchmark_case.hpp>

#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/types.hpp>
#include <ore/core/underlying.hpp>

namespace ore::benchmark {

namespace {

using ore::core::AssetType;
using ore::core::ExerciseStyle;
using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionType;
using ore::core::Underlying;

Underlying benchmark_underlying() {
    return Underlying{
        .symbol     = "BENCH",
        .exchange   = "SYNTHETIC",
        .asset_type = AssetType::Equity,
    };
}

// Convenience: expiration = valuation + `years` (approximate — 365.25
// days). Individual cases can override this by constructing their own
// dates. Suite-wide valuation is 2026-01-01.
std::chrono::year_month_day years_after(
    std::chrono::year_month_day base,
    double years)
{
    const auto base_days = std::chrono::sys_days{base};
    const auto delta_days = std::chrono::days{
        static_cast<int>(std::lround(years * 365.25))
    };
    return std::chrono::year_month_day{base_days + delta_days};
}

// Builder: assemble a full BenchmarkCase from the parts that actually
// vary between cases. Keeps the suite body readable.
BenchmarkCase make_case(
    std::string name,
    std::string description,
    double spot, double strike,
    double rate, double dividend_yield, double vol,
    double years, OptionType type)
{
    const auto valuation = std::chrono::year_month_day{
        std::chrono::year{2026}/1/1
    };
    Option option{
        .underlying = benchmark_underlying(),
        .strike     = strike,
        .expiration = years_after(valuation, years),
        .type       = type,
        .exercise   = ExerciseStyle::European,
    };
    MarketSnapshot market{
        .spot           = spot,
        .risk_free_rate = rate,
        .dividend_yield = dividend_yield,
        .volatility     = vol,
        .valuation_date = valuation,
    };
    return BenchmarkCase{
        .name        = std::move(name),
        .description = std::move(description),
        .option      = std::move(option),
        .market      = market,
    };
}

}  // namespace

std::vector<BenchmarkCase> standard_benchmark_suite() {
    // Everything derives from these baseline defaults:
    //   spot = 100, r = 5%, q = 0, sigma = 20%, T = 1y, Call
    // Each case perturbs *one* dimension so the effect is legible.
    //
    // Descriptions favour precision over marketing prose — they will
    // land verbatim in the CSV header comments and in `docs/BENCHMARKING.md`.
    return {
        make_case(
            "atm_call",
            "At-the-money 1y European call. Baseline. K=S=100, sigma=20%.",
            /*spot*/100.0, /*K*/100.0, /*r*/0.05, /*q*/0.0, /*sigma*/0.20,
            /*T*/1.0, OptionType::Call),
        make_case(
            "atm_put",
            "At-the-money 1y European put. Put/call parity control.",
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Put),
        make_case(
            "itm_call",
            "20% in-the-money call. K=80. Tests deep-intrinsic pricing.",
            100.0, 80.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call),
        make_case(
            "otm_call",
            "20% out-of-the-money call. K=120. Small option value, "
            "sensitive to tail behaviour.",
            100.0, 120.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call),
        make_case(
            "itm_put",
            "20% in-the-money put. K=120. Puts often show meaningful "
            "early-exercise premium for later engines.",
            100.0, 120.0, 0.05, 0.0, 0.20, 1.0, OptionType::Put),
        make_case(
            "otm_put",
            "20% out-of-the-money put. K=80.",
            100.0, 80.0, 0.05, 0.0, 0.20, 1.0, OptionType::Put),
        make_case(
            "short_maturity_call",
            "One-month ATM call, T=1/12y. Stresses time-discretisation "
            "in binomial and gives MC very small SE.",
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0 / 12.0, OptionType::Call),
        make_case(
            "long_maturity_call",
            "Five-year ATM call. Big option value; the terminal drift "
            "term dominates the SDE.",
            100.0, 100.0, 0.05, 0.0, 0.20, 5.0, OptionType::Call),
        make_case(
            "low_vol_call",
            "Low-volatility ATM call, sigma=5%. Payoff distribution is "
            "concentrated; MC variance is small.",
            100.0, 100.0, 0.05, 0.0, 0.05, 1.0, OptionType::Call),
        make_case(
            "high_vol_call",
            "High-volatility ATM call, sigma=60%. Fat-tailed payoff "
            "distribution; MC variance is large.",
            100.0, 100.0, 0.05, 0.0, 0.60, 1.0, OptionType::Call),
        make_case(
            "dividend_paying_call",
            "ATM call on a 3% continuously-yielding underlying. Forward "
            "drifts below spot; call value is lower than the no-div case.",
            100.0, 100.0, 0.05, 0.03, 0.20, 1.0, OptionType::Call),
        make_case(
            "negative_rate_call",
            "ATM call at r=-2%. Post-2015 rates regime. Discount factor "
            "> 1; put/call parity still holds.",
            100.0, 100.0, -0.02, 0.0, 0.20, 1.0, OptionType::Call),
    };
}

} // namespace ore::benchmark
