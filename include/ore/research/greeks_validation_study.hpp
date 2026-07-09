/**
 * @file greeks_validation_study.hpp
 * @brief Built-in study: compare Black-Scholes Greeks against the
 *        provider-reported Greeks that come with the EOD feed.
 *
 * ### Pipeline
 *
 * For each option contract with a complete set of provider Greeks
 * (delta, gamma, theta, vega, rho) and a positive provider IV:
 *
 *   1. Call `BlackScholesEngine::price` using the provider's IV.
 *   2. Convert BS Greeks into the vendor's unit convention:
 *        - vega:  BS is per unit-vol, vendor is per-1%   → x 0.01
 *        - theta: BS is per year,      vendor is per-day → x (1/365)
 *   3. Record both sets side by side and the signed / absolute error.
 *
 * ### Purpose
 *
 * The study exercises the pricing engine against thousands of
 * contracts every trading day and confirms that
 *
 *   *given* the same IV as the vendor, our Black-Scholes derivatives
 *   agree with the vendor's derivatives to numerical precision.
 *
 * Systematic disagreement flags a unit-convention mismatch, a
 * dividend-treatment difference, or an off-by-one in day count —
 * things that only show up in aggregate over a very large sample.
 *
 * ### Aggregate metrics
 *
 * The study reports MAE, RMSE, and 50/95/99th-percentile absolute
 * errors for every Greek. That triplet answers three separate
 * questions:
 *
 *   - MAE: typical difference size.
 *   - RMSE: whether a few large outliers dominate.
 *   - Percentiles: distributional shape, robust to outliers.
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

#include <ore/core/provider_greeks.hpp>
#include <ore/core/types.hpp>
#include <ore/research/research_study.hpp>

namespace ore::research {

/**
 * @class GreeksValidationStudy
 * @brief Compares BS Greeks (converted to vendor units) against
 *        provider Greeks.
 */
class GreeksValidationStudy final : public ResearchStudy {
public:
    /**
     * @brief Study configuration.
     */
    struct Config {
        std::string output_filename{"greeks_validation.csv"};

        /**
         * Multiplier applied to BS vega to match the vendor's unit.
         * OptionsDX archives publish vega per-1% (a common desk
         * convention). BS reports vega per-unit-vol; a value of 0.01
         * converts.
         */
        double vega_scale{0.01};

        /**
         * Multiplier applied to BS theta to match the vendor's unit.
         * OptionsDX archives publish theta per calendar day; BS
         * reports theta per year. `1 / 365` converts.
         */
        double theta_scale{1.0 / 365.0};

        /**
         * Skip rows whose provider IV is missing or non-positive.
         * BS cannot compute Greeks without a volatility input, and
         * substituting a fallback would defeat the study's purpose.
         */
        bool require_provider_iv{true};

        /**
         * Skip rows whose provider Greeks are not all populated.
         * When `false`, the row is still emitted but only the
         * populated Greeks are compared; the missing ones survive
         * in the CSV as empty fields (pandas NaN).
         */
        bool require_all_provider_greeks{false};
    };

    GreeksValidationStudy() = default;
    explicit GreeksValidationStudy(Config config) : config_(std::move(config)) {}

    [[nodiscard]] std::string_view name() const noexcept override {
        return "GreeksValidationStudy";
    }

    void begin(const ResearchContext& ctx) override;
    void process(const ResearchContext& ctx) override;
    void end(const ResearchContext& ctx, ResearchReport& report) override;

    [[nodiscard]] std::unique_ptr<ResearchStudy> clone() const override;
    void merge(const ResearchStudy& other) override;

    /**
     * @struct Row
     * @brief One CSV row of the study output.
     */
    struct Row {
        std::chrono::year_month_day date{};
        std::chrono::year_month_day expiration{};
        double                      strike{0.0};
        ore::core::OptionType       type{ore::core::OptionType::Call};
        double                      provider_iv{0.0};

        std::optional<double> provider_delta{}, computed_delta{}, delta_error{};
        std::optional<double> provider_gamma{}, computed_gamma{}, gamma_error{};
        std::optional<double> provider_theta{}, computed_theta{}, theta_error{};
        std::optional<double> provider_vega{},  computed_vega{},  vega_error{};
        std::optional<double> provider_rho{},   computed_rho{},   rho_error{};
    };

    /**
     * @struct Statistics
     * @brief Per-Greek aggregate error statistics.
     */
    struct GreekStats {
        std::size_t comparisons{0};
        double      mae{0.0};
        double      rmse{0.0};
        double      p50{0.0};
        double      p95{0.0};
        double      p99{0.0};
        double      worst{0.0};
    };

    struct Statistics {
        GreekStats delta{};
        GreekStats gamma{};
        GreekStats theta{};
        GreekStats vega{};
        GreekStats rho{};
    };

    [[nodiscard]] const std::vector<Row>& rows()  const noexcept { return rows_; }
    [[nodiscard]] const Statistics&       stats() const noexcept { return stats_; }
    [[nodiscard]] const Config&           config() const noexcept { return config_; }

private:
    Config             config_{};
    std::vector<Row>   rows_{};
    Statistics         stats_{};
    std::size_t        skipped_{0};

    mutable std::mutex mutex_{};
};

} // namespace ore::research
