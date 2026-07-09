/**
 * @file pricing_validation_study.hpp
 * @brief Built-in study: verify the round-trip
 *        Market Price → IV Solver → Black-Scholes → Market Price
 *        to numerical precision.
 *
 * ### What the study checks
 *
 * For every viable contract:
 *
 *   1. Take the market midpoint `M = (bid + ask) / 2`.
 *   2. Recover `sigma` such that `BS(sigma) = M` via
 *      `ImpliedVolatilitySolver`.
 *   3. Reprice: `M' = BS(sigma)`.
 *   4. Residual `M' - M` should be at most the solver tolerance
 *      (`ImpliedVolatilitySolver::Config::tolerance`, default 1e-10).
 *
 * A large residual reveals that the pricing engine and the IV
 * solver disagree on the same problem — the kind of internal
 * inconsistency that can silently corrupt every downstream
 * analytic. Running the check daily across thousands of contracts
 * is a cheap regression test for the numerical layer.
 *
 * ### Metrics reported
 *
 *   - Mean absolute residual.
 *   - RMSE.
 *   - Max absolute residual (with the option identity that produced it).
 *   - Fraction of contracts within `residual_tolerance`.
 *
 * ### Not what the study checks
 *
 * The residual measures self-consistency, not model correctness. A
 * contract where BS is a bad model (deep-ITM American, dividend
 * discontinuity, ...) can still round-trip perfectly: the round-trip
 * only says "BS with sigma = X reproduces price X". The residual
 * validates the solver's convergence, not the appropriateness of
 * the model.
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
 * @class PricingValidationStudy
 * @brief Round-trip market → IV → BS → market and record the
 *        residual.
 */
class PricingValidationStudy final : public ResearchStudy {
public:
    struct Config {
        ore::pricing::ImpliedVolatilitySolver::Config solver{};
        std::string output_filename{"pricing_validation.csv"};

        /**
         * Residual threshold: contracts whose `|price - mid|` is
         * below this are counted as "within-tolerance" in the
         * summary. The default is intentionally loose (1e-8) so a
         * mid-quote quantised to `0.01` (typical US-listed) can
         * still land within tolerance even when the IV solver
         * itself converges to `1e-10`.
         */
        double residual_tolerance{1e-8};

        /**
         * Skip contracts whose provider IV is missing. In principle
         * the round-trip test does not need the provider IV — the
         * study runs entirely against BS. In practice, restricting
         * to rows where the provider published an IV yields a
         * fairer sample (contracts the vendor deemed liquid enough
         * to price). Set to `false` to include every contract with
         * a valid bid/ask.
         */
        bool require_provider_iv{true};
    };

    PricingValidationStudy() = default;
    explicit PricingValidationStudy(Config config) : config_(std::move(config)) {}

    [[nodiscard]] std::string_view name() const noexcept override {
        return "PricingValidationStudy";
    }

    void begin(const ResearchContext& ctx) override;
    void process(const ResearchContext& ctx) override;
    void end(const ResearchContext& ctx, ResearchReport& report) override;

    [[nodiscard]] std::unique_ptr<ResearchStudy> clone() const override;
    void merge(const ResearchStudy& other) override;

    struct Row {
        std::chrono::year_month_day date{};
        std::chrono::year_month_day expiration{};
        double                      strike{0.0};
        ore::core::OptionType       type{ore::core::OptionType::Call};
        double                      mid_price{0.0};
        std::optional<double>       recovered_iv{};
        std::optional<double>       repriced{};
        std::optional<double>       residual{};
        bool                        converged{false};
    };

    struct Statistics {
        std::size_t total{0};
        std::size_t within_tolerance{0};
        double      mean_abs_residual{0.0};
        double      rmse_residual{0.0};
        double      worst_abs_residual{0.0};
        double      fraction_within_tolerance{0.0};
    };

    [[nodiscard]] const std::vector<Row>& rows()   const noexcept { return rows_; }
    [[nodiscard]] const Statistics&       stats()  const noexcept { return stats_; }
    [[nodiscard]] const Config&           config() const noexcept { return config_; }

private:
    Config             config_{};
    std::vector<Row>   rows_{};
    Statistics         stats_{};
    std::size_t        skipped_{0};

    mutable std::mutex mutex_{};
};

} // namespace ore::research
