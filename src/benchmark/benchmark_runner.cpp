#include <ore/benchmark/benchmark_runner.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <ostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ore/benchmark/benchmark_case.hpp>
#include <ore/pricing/binomial_tree_engine.hpp>
#include <ore/pricing/black_scholes_engine.hpp>
#include <ore/pricing/greeks.hpp>
#include <ore/pricing/monte_carlo_engine.hpp>
#include <ore/pricing/pricing_engine.hpp>
#include <ore/pricing/pricing_result.hpp>

namespace ore::benchmark {

namespace {

using ore::pricing::BinomialTreeEngine;
using ore::pricing::BlackScholesEngine;
using ore::pricing::MonteCarloEngine;
using ore::pricing::PricingEngine;
using ore::pricing::PricingResult;

/**
 * @brief Time a single `price(option, market)` call. Returns the median
 *        elapsed microseconds and the last `PricingResult` produced.
 *
 * We do `reps` timed executions and take the median so that scheduler
 * jitter does not blow up the metric. The result value is the *last*
 * run's — since engines are pure functions of their inputs this is
 * equivalent to any other run's, and taking the last avoids double
 * work.
 */
struct TimedResult {
    PricingResult result;
    double        median_microseconds;
};

TimedResult time_price_call(
    const PricingEngine& engine,
    const ore::core::Option& option,
    const ore::core::MarketSnapshot& market,
    std::size_t reps)
{
    if (reps == 0) {
        // Zero reps is meaningless; treat as one for a well-defined
        // return value. Never happens through the public API — Config
        // clamps this — but the internal helper stays defensive.
        reps = 1;
    }

    std::vector<double> times_us;
    times_us.reserve(reps);
    PricingResult last{};

    for (std::size_t i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        last = engine.price(option, market);
        const auto t1 = std::chrono::steady_clock::now();
        const auto dt_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        times_us.push_back(dt_us);
    }

    // Median: nth_element on the middle index. Odd counts return the
    // middle sample; even counts average the two middles.
    const auto n = times_us.size();
    const auto mid = n / 2;
    std::nth_element(times_us.begin(), times_us.begin() + static_cast<std::ptrdiff_t>(mid), times_us.end());
    double median = times_us[mid];
    if (n % 2 == 0) {
        // Everything left of `mid` is <= times_us[mid]; take the max
        // of that slice as the "lower middle".
        const auto lower_middle = *std::max_element(
            times_us.begin(),
            times_us.begin() + static_cast<std::ptrdiff_t>(mid));
        median = 0.5 * (lower_middle + times_us[mid]);
    }
    return {std::move(last), median};
}

/**
 * @brief Copies the pricing-side fields of `PricingResult` into the
 *        benchmark row. Doesn't touch the timing / error / memory
 *        columns — those are the runner's job.
 */
void fill_from_result(BenchmarkRow& row, const PricingResult& r) {
    row.price = r.price;
    row.greeks = r.greeks;
    row.iterations = r.iterations;
    row.standard_error = r.standard_error;
    row.confidence_interval_95 = r.confidence_interval_95;
}

/**
 * @brief Locate the reference engine within `engines` — the *first*
 *        one whose `name()` starts with `prefix`. Returns nullptr if
 *        `prefix` is empty or no match is found.
 */
const PricingEngine* find_reference(
    std::span<const std::unique_ptr<PricingEngine>> engines,
    std::string_view prefix) noexcept
{
    if (prefix.empty()) return nullptr;
    for (const auto& e : engines) {
        if (!e) continue;
        const auto n = e->name();
        if (n.size() >= prefix.size() &&
            std::string_view(n).substr(0, prefix.size()) == prefix)
        {
            return e.get();
        }
    }
    return nullptr;
}

/**
 * @brief Set the (`absolute_error`, `relative_error`) pair on a row
 *        against a reference price. Relative error uses the *larger*
 *        of the two magnitudes as the denominator so we don't blow up
 *        near zero.
 */
void apply_reference_error(BenchmarkRow& row, double reference_price) {
    row.reference_price = reference_price;
    const double abs_err = std::fabs(row.price - reference_price);
    row.absolute_error = abs_err;
    const double denom = std::max(std::fabs(row.price), std::fabs(reference_price));
    // If both prices are (near) zero the ratio is undefined; leave
    // relative_error as nullopt so plots don't render a spike. Uses a
    // small threshold rather than exact zero to guard against
    // subnormal denominators.
    if (denom > 1e-14) {
        row.relative_error = abs_err / denom;
    }
}

/**
 * @brief Render an optional numeric column into the CSV stream.
 */
template <typename T>
void write_optional(std::ostream& os, const std::optional<T>& v) {
    if (v.has_value()) os << *v;
    // else: emit nothing so pandas parses it as NaN.
}

} // namespace

//
// BenchmarkReport::write_csv --------------------------------------------------
//

void BenchmarkReport::write_csv(std::ostream& os) const {
    // Header order is documented in docs/BENCHMARKING.md — do not
    // reorder without updating the Python plotters that index by name.
    os << "case_name,case_description,engine_name,"
       << "price,delta,gamma,vega,theta,rho,"
       << "runtime_us,iterations,standard_error,"
       << "ci_95_low,ci_95_high,"
       << "reference_price,absolute_error,relative_error,"
       << "estimated_memory_bytes\n";

    // Fixed precision so CSV diffs are meaningful. `defaultfloat` for
    // integers, `std::fixed` with 12 decimals for floats — enough for
    // pandas comparisons but not so noisy that human readers drown.
    auto original_precision = os.precision();
    auto original_flags = os.flags();

    for (const auto& r : rows) {
        // Wrap description in double quotes so commas inside are safe.
        os << r.case_name << ','
           << '"' << r.case_description << '"' << ','
           << r.engine_name << ',';

        os.precision(12);
        os.setf(std::ios::fixed, std::ios::floatfield);
        os << r.price << ','
           << r.greeks.delta << ','
           << r.greeks.gamma << ','
           << r.greeks.vega << ','
           << r.greeks.theta << ','
           << r.greeks.rho << ',';

        os << r.runtime_us << ',';
        os.unsetf(std::ios::floatfield);
        os.precision(original_precision);

        write_optional(os, r.iterations);
        os << ',';
        os.precision(12);
        os.setf(std::ios::fixed, std::ios::floatfield);
        write_optional(os, r.standard_error);
        os << ',';
        if (r.confidence_interval_95.has_value()) {
            os << r.confidence_interval_95->first;
        }
        os << ',';
        if (r.confidence_interval_95.has_value()) {
            os << r.confidence_interval_95->second;
        }
        os << ',';
        write_optional(os, r.reference_price);
        os << ',';
        write_optional(os, r.absolute_error);
        os << ',';
        write_optional(os, r.relative_error);
        os << ',';
        os.unsetf(std::ios::floatfield);
        os.precision(original_precision);

        os << r.estimated_memory_bytes << '\n';
    }

    os.flags(original_flags);
    os.precision(original_precision);
}

void BenchmarkReport::write_csv(const std::filesystem::path& path) const {
    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error(
            "BenchmarkReport::write_csv: cannot open " + path.string());
    }
    write_csv(ofs);
}

//
// BenchmarkRunner::run -------------------------------------------------------
//

BenchmarkReport BenchmarkRunner::run(
    std::span<const std::unique_ptr<PricingEngine>> engines,
    std::span<const BenchmarkCase>                  cases) const
{
    BenchmarkReport report{};
    report.rows.reserve(engines.size() * cases.size());

    // Cache the reference engine pointer once per invocation; we
    // resolve *by name prefix*, not by index, so it's stable across
    // reorderings of `engines`.
    const auto* reference = find_reference(engines, config_.reference_engine_prefix);

    for (const auto& case_ : cases) {
        // Pre-compute the reference price for this case (if any) so
        // every non-reference engine's row can attach errors below.
        std::optional<double> reference_price{};
        if (reference != nullptr) {
            reference_price = reference->price(case_.option, case_.market).price;
        }

        for (const auto& engine_ptr : engines) {
            if (!engine_ptr) continue;  // Defensive: skip null slots.
            const auto& engine = *engine_ptr;

            BenchmarkRow row{};
            row.case_name = case_.name;
            row.case_description = case_.description;
            row.engine_name = std::string(engine.name());
            row.estimated_memory_bytes = estimated_memory_bytes(engine);

            const auto timed = time_price_call(
                engine, case_.option, case_.market, config_.median_reps);
            fill_from_result(row, timed.result);
            row.runtime_us = timed.median_microseconds;

            if (reference_price.has_value() && &engine != reference) {
                apply_reference_error(row, *reference_price);
            } else if (reference_price.has_value() && &engine == reference) {
                // Reference engine's row: error is zero by definition.
                row.reference_price = *reference_price;
                row.absolute_error = 0.0;
                row.relative_error = 0.0;
            }

            report.rows.push_back(std::move(row));
        }
    }

    return report;
}

BenchmarkReport BenchmarkRunner::run_binomial_convergence(
    const BenchmarkCase& case_,
    std::span<const std::size_t> step_counts) const
{
    BenchmarkReport report{};
    report.rows.reserve(step_counts.size());

    // Reference (optional): a single BlackScholesEngine for the whole
    // sweep. If the config prefix isn't "BlackScholes" we still use
    // BS as the analytic ground truth — the convergence study only
    // makes sense against the closed-form answer.
    std::optional<double> reference_price{};
    BlackScholesEngine bs{};
    if (!config_.reference_engine_prefix.empty()) {
        reference_price = bs.price(case_.option, case_.market).price;
    }

    for (std::size_t steps : step_counts) {
        BinomialTreeEngine::Config config{};
        config.steps = steps;
        // Greeks off for convergence sweeps: 8x extra tree evaluations
        // would triple the runtime column with no analytical payoff.
        config.compute_greeks = false;
        BinomialTreeEngine engine{config};

        BenchmarkRow row{};
        row.case_name = case_.name;
        row.case_description = case_.description;
        row.engine_name = std::string(engine.name());
        row.estimated_memory_bytes = estimated_memory_bytes(engine);

        const auto timed = time_price_call(
            engine, case_.option, case_.market, config_.median_reps);
        fill_from_result(row, timed.result);
        row.runtime_us = timed.median_microseconds;

        if (reference_price.has_value()) {
            apply_reference_error(row, *reference_price);
        }

        report.rows.push_back(std::move(row));
    }

    return report;
}

BenchmarkReport BenchmarkRunner::run_monte_carlo_convergence(
    const BenchmarkCase& case_,
    std::span<const std::size_t> path_counts,
    std::uint64_t seed,
    bool          antithetic) const
{
    BenchmarkReport report{};
    report.rows.reserve(path_counts.size());

    std::optional<double> reference_price{};
    BlackScholesEngine bs{};
    if (!config_.reference_engine_prefix.empty()) {
        reference_price = bs.price(case_.option, case_.market).price;
    }

    for (std::size_t paths : path_counts) {
        MonteCarloEngine::Config config{};
        config.paths = paths;
        config.seed = seed;
        config.antithetic_variates = antithetic;
        config.compute_greeks = false;
        MonteCarloEngine engine{config};

        BenchmarkRow row{};
        row.case_name = case_.name;
        row.case_description = case_.description;
        row.engine_name = std::string(engine.name());
        row.estimated_memory_bytes = estimated_memory_bytes(engine);

        const auto timed = time_price_call(
            engine, case_.option, case_.market, config_.median_reps);
        fill_from_result(row, timed.result);
        row.runtime_us = timed.median_microseconds;

        if (reference_price.has_value()) {
            apply_reference_error(row, *reference_price);
        }

        report.rows.push_back(std::move(row));
    }

    return report;
}

//
// estimated_memory_bytes ------------------------------------------------------
//

std::size_t estimated_memory_bytes(const PricingEngine& engine) noexcept {
    // Black-Scholes: closed-form, stateless. The 32 bytes are a
    // vtable + a padding word for the base subobject — we ignore
    // them because they're constant and irrelevant to workload
    // scaling analysis. Report zero as "no working set".
    if (dynamic_cast<const BlackScholesEngine*>(&engine) != nullptr) {
        return 0;
    }

    // Binomial: rolling vector of terminal values, plus two vectors
    // for u^k and d^k precomputed powers used when marching backwards.
    // Total: 3 * (steps + 1) * sizeof(double). This is the O(N)
    // memory bound that motivates the rolling-vector implementation.
    if (const auto* bt = dynamic_cast<const BinomialTreeEngine*>(&engine)) {
        const std::size_t n = bt->config().steps + 1;
        return 3 * n * sizeof(double);
    }

    // Monte Carlo: Mersenne-Twister state (~2.5 KB for MT19937_64) +
    // the normal-distribution cache (Box-Muller state, few dozen
    // bytes) + a handful of Welford accumulator doubles. Independent
    // of `paths` — Welford is why we picked it.
    if (dynamic_cast<const MonteCarloEngine*>(&engine) != nullptr) {
        return sizeof(std::mt19937_64) + 64;
    }

    // Unknown engine: report 0 rather than raise. Benchmark rows are
    // best-effort diagnostics, not correctness-critical.
    return 0;
}

} // namespace ore::benchmark
