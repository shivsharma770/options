/**
 * @file iv_validation_study.hpp
 * @brief Built-in study: recover implied volatility from market
 *        midpoints and compare against the provider's IV.
 *
 * ### Purpose
 *
 * For every option in every trading day, the study
 *
 *   1. Computes the market midpoint `mid = (bid + ask) / 2`.
 *   2. Runs `ImpliedVolatilitySolver` to recover `sigma` such that
 *      `BlackScholes(sigma) == mid`.
 *   3. Compares the recovered `sigma` against the provider's
 *      published IV (`Quote::implied_volatility`).
 *
 * The study writes a per-contract CSV and computes aggregate error
 * metrics (mean, median, RMSE, worst, convergence rate) at the end.
 *
 * ### Interpretation
 *
 * "IV validation" here does not mean "the provider is right and we
 * check ourselves against it" — the provider is a black-box
 * whose IV computation is neither documented nor guaranteed
 * consistent with Black-Scholes. The study measures the *distance*
 * between the two IVs, which is the diagnostic quantity a
 * researcher actually cares about: large systematic gaps flag
 * regime differences (dividend handling, rate assumptions, American
 * exercise, ...); small gaps confirm the pipeline is intact.
 *
 * ### Filtering rules
 *
 * A contract is skipped (and counted in `skipped_contracts`) when:
 *
 *   - `bid <= 0` or `ask <= 0` (no market).
 *   - `bid > ask` (crossed quote).
 *   - `Quote::implied_volatility` is missing (nothing to compare against).
 *   - The option is expired at the snapshot date.
 *   - `strike <= 0` (defensive).
 *
 * Every other contract is fed to the solver. Non-convergent solves
 * are still emitted as CSV rows, with `converged = 0` and the
 * partial residual — that visibility is more useful than silently
 * dropping them.
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

#include <ore/core/types.hpp>
#include <ore/pricing/implied_volatility_solver.hpp>
#include <ore/research/research_study.hpp>

namespace ore::research {

/**
 * @class IVValidationStudy
 * @brief Recover IV from mid-price and diff against provider IV.
 */
class IVValidationStudy final : public ResearchStudy {
public:
    /**
     * @brief Study configuration.
     *
     * Defaults are chosen to match the shape of typical equity-index
     * research; override for other asset classes.
     */
    struct Config {
        /** IV solver configuration. Defaults are pragmatic for
         *  SPY-scale options. */
        ore::pricing::ImpliedVolatilitySolver::Config solver{};

        /** Output CSV filename (relative to
         *  `ResearchContext::output_dir`). */
        std::string output_filename{"iv_validation.csv"};

        /** If `true`, only European contracts are attempted (the IV
         *  solver only supports European). American contracts count
         *  as `skipped`. */
        bool european_only{true};
    };

    IVValidationStudy() = default;
    explicit IVValidationStudy(Config config) : config_(std::move(config)) {}

    /** @copydoc ResearchStudy::name */
    [[nodiscard]] std::string_view name() const noexcept override {
        return "IVValidationStudy";
    }

    void begin(const ResearchContext& ctx) override;
    void process(const ResearchContext& ctx) override;
    void end(const ResearchContext& ctx, ResearchReport& report) override;

    /** @copydoc ResearchStudy::clone */
    [[nodiscard]] std::unique_ptr<ResearchStudy> clone() const override;

    /** @copydoc ResearchStudy::merge */
    void merge(const ResearchStudy& other) override;

    /**
     * @struct Row
     * @brief One CSV row of the study output.
     *
     * Exposed so tests can inspect the accumulated results directly
     * (rather than round-tripping through the file). Studies that
     * only care about the summary can call `stats()` and ignore
     * `rows()`.
     */
    struct Row {
        std::chrono::year_month_day date{};
        std::chrono::year_month_day expiration{};
        double                      strike{0.0};
        ore::core::OptionType       type{ore::core::OptionType::Call};
        double                      mid_price{0.0};
        std::optional<double>       provider_iv{};
        std::optional<double>       computed_iv{};
        std::optional<double>       absolute_error{};
        std::optional<double>       relative_error{};
        std::size_t                 iterations{0};
        ore::pricing::ImpliedVolatilityMethod method{
            ore::pricing::ImpliedVolatilityMethod::Newton};
        bool                        converged{false};
    };

    /**
     * @struct Statistics
     * @brief Aggregate error statistics filled in `end()` and
     *        available afterwards.
     */
    struct Statistics {
        std::size_t solves_attempted{0};
        std::size_t solves_converged{0};
        double      mean_absolute_error{0.0};
        double      median_absolute_error{0.0};
        double      rmse{0.0};
        double      worst_error{0.0};
        /** Convergence rate = converged / attempted, or `0.0` for an
         *  empty run. Distinct from
         *  `solves_converged / (attempted-newton-only)` so the
         *  metric stays meaningful when bisection is in play. */
        double      convergence_rate{0.0};
    };

    /** @brief Accumulated per-contract rows. */
    [[nodiscard]] const std::vector<Row>& rows() const noexcept { return rows_; }
    /** @brief Aggregate statistics. Populated by `end()`. */
    [[nodiscard]] const Statistics& stats() const noexcept { return stats_; }
    /** @brief The active configuration. */
    [[nodiscard]] const Config& config() const noexcept { return config_; }

private:
    Config              config_{};
    std::vector<Row>    rows_{};
    Statistics          stats_{};
    std::size_t         skipped_{0};

    /** Guards `rows_` and `skipped_` when a `clone()` is running in
     *  a worker thread. Only the clone touches its own state so
     *  contention is nil in practice, but the mutex makes the
     *  contract explicit and lets the study be re-used in scenarios
     *  we haven't anticipated. */
    mutable std::mutex  mutex_{};
};

} // namespace ore::research
