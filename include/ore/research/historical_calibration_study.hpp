/**
 * @file historical_calibration_study.hpp
 * @brief Built-in study: drive `OptionChainCalibrator` and
 *        `volatility_analytics` across every trading day in the
 *        dataset, exporting per-day calibration + smile + surface
 *        results as CSVs.
 *
 * ### What this study exists for
 *
 * `OptionChainCalibrator` and the `volatility_analytics` free
 * functions (`build_smiles`, `build_term_structure`, `build_surface`,
 * `compute_skew_metrics`) already know how to turn a *single*
 * `OptionChain` into calibration diagnostics and every classical
 * volatility structure representation. What they cannot do on their
 * own is iterate over a `HistoricalDataset`, aggregate the per-day
 * outputs, and write time-series CSVs. That gap was the reason
 * historical smile / surface / skew studies previously had to be
 * hand-rolled by external drivers.
 *
 * `HistoricalCalibrationStudy` is a thin `ResearchStudy` wrapper that
 * closes the gap without duplicating any calibration or analytics
 * logic. Each `process(ctx)` call:
 *
 *   1. Runs `OptionChainCalibrator::calibrate` on today's chain.
 *   2. Runs `build_smiles`, `build_term_structure`, `build_surface`,
 *      `compute_skew_metrics` on the returned report.
 *   3. Flattens each output into per-day rows tagged with `ctx.date()`.
 *
 * `end()` then writes five long-format CSVs — `calibration.csv`,
 * `smiles.csv`, `term_structure.csv`, `surface.csv`, `skew.csv` —
 * using `ResearchCsvWriter` for consistent formatting with every
 * other study.
 *
 * ### CSV consumption
 *
 * Every emitted CSV starts with a `date` column so pandas can
 * `groupby('date')` (or slice by date range) without any post-
 * processing. Missing optionals are empty fields. All floating-point
 * numbers are `%.17g`.
 *
 * ### Parallel execution
 *
 * The study implements `clone()` and `merge()`. Workers accumulate
 * their own row vectors and the primary merges them in ascending
 * worker order; the engine partitions by contiguous date ranges so
 * the final concatenation is already date-ordered. `end()` performs
 * a final stable sort by date as a defensive guarantee.
 *
 * ### What this study deliberately does not do
 *
 *  - It does not introduce a new pricing model or a new IV solver.
 *  - It does not interpolate or fit the surface. Missing grid cells
 *    stay as `NaN` in the CSV, matching `VolatilitySurface`'s own
 *    "no interpolation" contract.
 *  - It does not compute cross-day statistics (realized vol, IV
 *    rank, ...). Those belong to studies layered on top of this
 *    one's CSVs.
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <ore/analytics/option_chain_calibrator.hpp>
#include <ore/analytics/volatility_analytics.hpp>
#include <ore/core/types.hpp>
#include <ore/numerics/solver_result.hpp>
#include <ore/research/research_study.hpp>

namespace ore::research {

/**
 * @class HistoricalCalibrationStudy
 * @brief Whole-archive calibration + volatility-structure exporter.
 */
class HistoricalCalibrationStudy final : public ResearchStudy {
public:
    /**
     * @brief Study configuration.
     */
    struct Config {
        /** Passed through to `OptionChainCalibrator`. Defaults are
         *  fine for SPY-scale equity chains. */
        ore::analytics::OptionChainCalibrator::Config calibrator{};

        /** Moneyness convention used to populate `smiles.csv`. The
         *  default (`LogSimple = ln(K/S)`) is the standard academic
         *  choice; see `volatility_analytics.hpp`. */
        ore::analytics::Moneyness moneyness_convention{
            ore::analytics::Moneyness::LogSimple};

        /** Output filenames, relative to
         *  `ResearchContext::output_dir()`. Each corresponds to one
         *  emit toggle below. */
        std::string calibration_filename{"calibration.csv"};
        std::string smiles_filename{"smiles.csv"};
        std::string term_structure_filename{"term_structure.csv"};
        std::string surface_filename{"surface.csv"};
        std::string skew_filename{"skew.csv"};

        /** Per-output emit toggles. Setting a flag to `false` skips
         *  the corresponding CSV entirely (the study still runs the
         *  analytics — the cost saving is only I/O). */
        bool export_calibration{true};
        bool export_smiles{true};
        bool export_term_structure{true};
        bool export_surface{true};
        bool export_skew{true};

        /**
         * Include contracts that were *not* calibrated
         * (`skip_reason != None`, or solver failed) in
         * `calibration.csv`. Off by default because it inflates the
         * historical file by up to 10x for archives where deep-OTM
         * strikes are heavily unquoted. Turn on to run diagnostics
         * on the skip-reason mix over time.
         */
        bool include_skipped_in_calibration{false};
    };

    HistoricalCalibrationStudy() = default;
    explicit HistoricalCalibrationStudy(Config config)
        : config_(std::move(config)) {}

    /** @copydoc ResearchStudy::name */
    [[nodiscard]] std::string_view name() const noexcept override {
        return "HistoricalCalibrationStudy";
    }

    void begin(const ResearchContext& ctx) override;
    void process(const ResearchContext& ctx) override;
    void end(const ResearchContext& ctx, ResearchReport& report) override;

    /** @copydoc ResearchStudy::clone */
    [[nodiscard]] std::unique_ptr<ResearchStudy> clone() const override;

    /** @copydoc ResearchStudy::merge */
    void merge(const ResearchStudy& other) override;

    // -------------------------------------------------------------------
    // Row types
    // -------------------------------------------------------------------

    /**
     * @struct CalibrationRow
     * @brief One row of `calibration.csv`. Mirrors the single-day
     *        `CalibrationResult` shape with a leading `date` column
     *        and without the redundant per-contract `contract_symbol`
     *        (it is derivable from expiration+strike+type and would
     *        otherwise bloat the historical CSV).
     */
    struct CalibrationRow {
        std::chrono::year_month_day date{};
        std::chrono::year_month_day expiration{};
        double                      strike{0.0};
        ore::core::OptionType       type{ore::core::OptionType::Call};
        double                      bid{0.0};
        double                      ask{0.0};
        double                      last{0.0};
        double                      mid_price{0.0};
        std::optional<double>       provider_iv{};
        std::optional<double>       computed_iv{};
        std::optional<double>       absolute_error{};
        std::optional<double>       relative_error{};
        ore::numerics::SolverStatus solver_status{
            ore::numerics::SolverStatus::MaxIterationsReached};
        std::size_t                 iterations{0};
        bool                        used_bisection{false};
        double                      solver_residual{0.0};
        ore::analytics::SkipReason  skip_reason{ore::analytics::SkipReason::None};
    };

    /**
     * @struct SmileRow
     * @brief One row of `smiles.csv` (one point of one smile on one date).
     */
    struct SmileRow {
        std::chrono::year_month_day date{};
        std::chrono::year_month_day expiration{};
        double                      time_to_expiry{0.0};
        double                      strike{0.0};
        ore::core::OptionType       type{ore::core::OptionType::Call};
        ore::analytics::Moneyness   moneyness_convention{
            ore::analytics::Moneyness::LogSimple};
        double                      moneyness{0.0};
        double                      implied_volatility{0.0};
    };

    /**
     * @struct TermStructureRow
     * @brief One row of `term_structure.csv` (one expiration on one date).
     *
     * `atm_iv` is `std::nullopt` when `build_term_structure` produced
     * `NaN` for that expiration (no bracketing calibrated strikes).
     * Empty CSV field → pandas `NaN`.
     */
    struct TermStructureRow {
        std::chrono::year_month_day date{};
        std::chrono::year_month_day expiration{};
        double                      time_to_expiry{0.0};
        std::optional<double>       atm_iv{};
    };

    /**
     * @struct SurfaceRow
     * @brief One row of `surface.csv` — one (expiration, strike) grid
     *        cell on one date. Long-format so pandas can pivot back
     *        to the 2D grid when needed.
     */
    struct SurfaceRow {
        std::chrono::year_month_day date{};
        std::chrono::year_month_day expiration{};
        double                      time_to_expiry{0.0};
        double                      strike{0.0};
        std::optional<double>       implied_volatility{};
    };

    /**
     * @struct SkewRow
     * @brief One row of `skew.csv` — the standard 25-delta metrics
     *        for one expiration on one date.
     */
    struct SkewRow {
        std::chrono::year_month_day date{};
        std::chrono::year_month_day expiration{};
        double                      time_to_expiry{0.0};
        std::optional<double>       atm_iv{};
        std::optional<double>       call_25delta_iv{};
        std::optional<double>       put_25delta_iv{};
        std::optional<double>       risk_reversal{};
        std::optional<double>       butterfly{};
    };

    /**
     * @struct Statistics
     * @brief Aggregate counters populated in `end()`.
     */
    struct Statistics {
        /** Trading days seen (calls to `process`). */
        std::size_t days_processed{0};
        /** Total contracts across every day (calibrated + skipped +
         *  failed). Equals the sum of chain sizes visited. */
        std::size_t total_contracts{0};
        /** Contracts the calibrator solved successfully. */
        std::size_t total_calibrated{0};
        /** Contracts the calibrator skipped upfront (any
         *  `SkipReason != None`). */
        std::size_t total_skipped{0};
        /** Contracts where the solver ran but did not converge. */
        std::size_t total_failed_solves{0};
        /** Contracts contributing to the provider-vs-computed IV
         *  comparison (both IVs present). */
        std::size_t total_iv_comparisons{0};
        /** Convergence rate averaged over the days that had at
         *  least one solver attempt. Zero if none. */
        double      mean_convergence_rate{0.0};
        /** Mean of `|provider_iv - computed_iv|` across
         *  `total_iv_comparisons`. Zero if none. */
        double      mean_absolute_iv_error{0.0};
    };

    // -------------------------------------------------------------------
    // Introspection accessors
    // -------------------------------------------------------------------

    [[nodiscard]] const std::vector<CalibrationRow>&    calibration_rows()    const noexcept { return calibration_rows_; }
    [[nodiscard]] const std::vector<SmileRow>&          smile_rows()          const noexcept { return smile_rows_; }
    [[nodiscard]] const std::vector<TermStructureRow>&  term_structure_rows() const noexcept { return term_rows_; }
    [[nodiscard]] const std::vector<SurfaceRow>&        surface_rows()        const noexcept { return surface_rows_; }
    [[nodiscard]] const std::vector<SkewRow>&           skew_rows()           const noexcept { return skew_rows_; }
    [[nodiscard]] const Statistics&                     stats()               const noexcept { return stats_; }
    [[nodiscard]] const Config&                         config()              const noexcept { return config_; }

private:
    Config                          config_{};
    Statistics                      stats_{};

    // Accumulated per-day rows. Under parallel execution each worker
    // clone has its own set; `merge()` splices them into the primary.
    std::vector<CalibrationRow>     calibration_rows_{};
    std::vector<SmileRow>           smile_rows_{};
    std::vector<TermStructureRow>   term_rows_{};
    std::vector<SurfaceRow>         surface_rows_{};
    std::vector<SkewRow>            skew_rows_{};

    // Running aggregates rebuilt from the row vectors in `end()`; these
    // fields survive across `process()` calls only for parallel merges.
    std::size_t total_contracts_{0};
    std::size_t total_skipped_{0};
    std::size_t total_failed_solves_{0};
    std::size_t total_iv_comparisons_{0};
    double      sum_abs_iv_error_{0.0};
    double      sum_convergence_rate_{0.0};
    std::size_t days_with_attempts_{0};

    /** Guards the per-day merge into `*_rows_` and the running
     *  counters when a `clone()` is executing on a worker thread. */
    mutable std::mutex mutex_{};
};

} // namespace ore::research
