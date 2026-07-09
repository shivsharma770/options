/**
 * @file benchmark_runner.hpp
 * @brief `BenchmarkRunner`, `BenchmarkRow`, `BenchmarkReport` — the
 *        cross-engine timing + accuracy framework.
 *
 * A single run consists of `E` engines × `C` cases and produces `E·C`
 * rows in a `BenchmarkReport`. Each row records the price, the runtime
 * (median of `Config::median_reps` executions), the Greeks (all-zero
 * if the engine didn't compute them), and — when a *reference* engine
 * is present in the input set — the absolute and relative price errors
 * versus that reference.
 *
 * Convergence-study helpers (`run_binomial_convergence`,
 * `run_monte_carlo_convergence`) generate their own rows by sweeping
 * `steps` or `paths` for a single case; the row schema is identical
 * so all reports concatenate cleanly on the CSV side.
 *
 * See `docs/BENCHMARKING.md` for methodology and expected behaviour.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ore/benchmark/benchmark_case.hpp>
#include <ore/pricing/greeks.hpp>
#include <ore/pricing/pricing_engine.hpp>

namespace ore::benchmark {

/**
 * @brief One row of the benchmark report.
 *
 * Copied out of `PricingResult` plus the timing and error columns
 * that only make sense in a cross-engine context. Every column is
 * exposed as a public data member — the whole point is diffable CSV.
 */
struct BenchmarkRow {
    /** Slug (`BenchmarkCase::name`) — stable across runs. */
    std::string case_name{};
    /** Human sentence (`BenchmarkCase::description`) — for CSV comments. */
    std::string case_description{};
    /** `PricingEngine::name()` snapshot, e.g. `"Binomial(CRR, N=500)"`. */
    std::string engine_name{};

    /** Fair value as reported by the engine. */
    double         price{0.0};
    /** Sensitivities. Zero-filled if the engine doesn't populate them. */
    ore::pricing::Greeks greeks{};

    /** Wall-clock runtime of a single `price(option, market)` call,
     *  microseconds, median of `Config::median_reps`. */
    double         runtime_us{0.0};

    /** Copy of `PricingResult::iterations` (BS = empty, Binomial =
     *  N, MC = N_samples). */
    std::optional<std::size_t> iterations{};

    /** Copy of `PricingResult::standard_error`. */
    std::optional<double>                    standard_error{};
    /** Copy of `PricingResult::confidence_interval_95`. */
    std::optional<std::pair<double, double>> confidence_interval_95{};

    /** Reference price used to compute the error columns below, if any. */
    std::optional<double> reference_price{};
    /** `|price - reference_price|`. */
    std::optional<double> absolute_error{};
    /** `|price - reference_price| / max(|price|, |reference_price|)`. */
    std::optional<double> relative_error{};

    /** Rough working-set estimate in bytes for this engine at its
     *  current config. See `estimated_memory_bytes()`. */
    std::size_t estimated_memory_bytes{0};
};

/**
 * @brief Full benchmark report — a list of rows plus a CSV exporter.
 *
 * Deliberately a plain struct with `rows` exposed: consumers routinely
 * filter, sort, and index by (case, engine) pairs. No invariants to
 * uphold; the report is a *record*.
 */
struct BenchmarkReport {
    std::vector<BenchmarkRow> rows{};

    /** Emit the report as a CSV. See the .cpp for the column list. */
    void write_csv(std::ostream& os) const;
    /** File overload. Throws if the path cannot be opened. */
    void write_csv(const std::filesystem::path& path) const;
};

/**
 * @brief Runs pricing engines against benchmark cases.
 *
 * Stateless with respect to the engines and cases it processes; a
 * single `BenchmarkRunner` instance is safe to reuse.
 */
class BenchmarkRunner {
public:
    /**
     * @brief Runner configuration.
     */
    struct Config {
        /** Number of price() calls timed; the median is kept. Larger
         *  values smooth out scheduling noise at proportional cost.
         *  Set to 1 for deterministic tests. */
        std::size_t median_reps{3};

        /** The first engine whose `name()` **starts with** this prefix
         *  is treated as the reference. Default `"BlackScholes"` — that
         *  engine's price becomes the ground truth for the abs/rel
         *  error columns of the other engines. Set to empty to
         *  disable reference-based error computation. */
        std::string_view reference_engine_prefix{"BlackScholes"};
    };

    BenchmarkRunner() = default;
    explicit BenchmarkRunner(Config config) : config_(config) {}

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    /**
     * @brief Run every case against every engine. `engines[0]` is
     *        checked first for the reference-prefix match, then
     *        `engines[1]`, etc.
     *
     * @param engines  Vector of owning pointers; the runner does not
     *                 take ownership. Empty span → empty report.
     * @param cases    Vector of cases to run. Empty span → empty report.
     *
     * @return A `BenchmarkReport` with `engines.size() * cases.size()`
     *         rows. Ordering: for each case, all engines in order.
     */
    [[nodiscard]] BenchmarkReport run(
        std::span<const std::unique_ptr<ore::pricing::PricingEngine>> engines,
        std::span<const BenchmarkCase>                                cases) const;

    /**
     * @brief Sweep a `BinomialTreeEngine` over a set of step counts on
     *        a single case. Useful for convergence plots.
     *
     * Constructs a fresh `BinomialTreeEngine` for each step count. If
     * `Config::reference_engine_prefix` is non-empty, a
     * `BlackScholesEngine` is used to fill the error columns (only
     * meaningful for European cases).
     */
    [[nodiscard]] BenchmarkReport run_binomial_convergence(
        const BenchmarkCase& case_,
        std::span<const std::size_t> step_counts) const;

    /**
     * @brief Sweep a `MonteCarloEngine` over a set of path counts on
     *        a single case. Populates the standard-error and 95 % CI
     *        columns on every row. Same reference-engine logic as
     *        `run_binomial_convergence`.
     *
     * @param seed        Shared seed across every path count. Fixed
     *                    seed → reproducible convergence curve.
     * @param antithetic  Whether to enable antithetic variates for
     *                    every run.
     */
    [[nodiscard]] BenchmarkReport run_monte_carlo_convergence(
        const BenchmarkCase& case_,
        std::span<const std::size_t> path_counts,
        std::uint64_t seed = 42,
        bool          antithetic = true) const;

private:
    Config config_{};
};

/**
 * @brief Rough working-set memory estimate for `engine`, in bytes.
 *
 * Determined by `dynamic_cast` against the three concrete engine
 * types shipped so far:
 *
 *   * `BlackScholesEngine`  → 0 (stateless closed-form).
 *   * `BinomialTreeEngine`  → `3 * (steps + 1) * sizeof(double)`
 *                             — the `values`, `u_pow`, `d_pow` vectors
 *                             that the .cpp allocates.
 *   * `MonteCarloEngine`    → ~2 KB (Mersenne Twister state + normal-
 *                             distribution cache), independent of
 *                             `paths` because we use Welford accumulation.
 *
 * Unknown engine types return 0.
 */
[[nodiscard]] std::size_t estimated_memory_bytes(
    const ore::pricing::PricingEngine& engine) noexcept;

} // namespace ore::benchmark
