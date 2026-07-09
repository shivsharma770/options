/**
 * @file black_scholes_engine.hpp
 * @brief Analytical Black-Scholes-Merton pricer for European vanilla options.
 *
 * ### Model assumptions
 * - **European exercise only.** American options are rejected at the boundary
 *   with `std::invalid_argument`.
 * - **Lognormal underlying.** \f$ dS/S = (\mu - q)\, dt + \sigma\, dW \f$ with
 *   `q` continuously-compounded dividend yield.
 * - **Constant volatility, constant risk-free rate, constant dividend
 *   yield** over the option's life.
 * - **Frictionless markets, continuous trading, no arbitrage.** No taxes,
 *   transaction costs, bid-ask spreads, borrow costs, or capital controls.
 * - **Continuously-compounded rates** on ACT/365 fixed day count. The
 *   `(Option, MarketSnapshot)` overload derives `T` from
 *   `market.valuation_date` and `option.expiration` using
 *   `(days_between / 365)`.
 *
 * ### Formulas implemented
 * Let \f$ S \f$ = spot, \f$ K \f$ = strike, \f$ r \f$ = risk-free rate,
 * \f$ q \f$ = dividend yield, \f$ \sigma \f$ = volatility, \f$ T \f$ =
 * time-to-expiry in years.
 * \f{align*}{
 *   d_1 &= \frac{\ln(S/K) + (r - q + \sigma^2 / 2)\, T}{\sigma \sqrt{T}} \\
 *   d_2 &= d_1 - \sigma \sqrt{T} \\
 *   C &= S e^{-qT} N(d_1) - K e^{-rT} N(d_2) \\
 *   P &= K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)
 * \f}
 * Greek formulas and unit conventions are documented in `greeks.hpp`. See
 * `docs/BLACK_SCHOLES_VALIDATION.md` for the reference benchmarks the
 * implementation is tested against.
 *
 * ### Reference
 * - Black, F. and Scholes, M. (1973), "The Pricing of Options and
 *   Corporate Liabilities", *Journal of Political Economy* 81(3), 637-654.
 * - Merton, R. (1973), "Theory of Rational Option Pricing", *Bell Journal
 *   of Economics and Management Science* 4(1), 141-183. (Adds the
 *   continuous dividend yield.)
 * - Hull, J. (2018), *Options, Futures, and Other Derivatives*, 10th Ed.,
 *   Chapters 15 and 17.
 */
#pragma once

#include <string_view>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/types.hpp>
#include <ore/pricing/pricing_engine.hpp>
#include <ore/pricing/pricing_result.hpp>

namespace ore::pricing {

/**
 * @class BlackScholesEngine
 * @brief Closed-form European-option pricer (Black-Scholes-Merton).
 *
 * Stateless — a single instance can be reused across many pricing calls,
 * on different options, from multiple threads. Preferred over free
 * functions because it plugs into the `PricingEngine` virtual interface
 * that future engines will also implement.
 */
class BlackScholesEngine final : public PricingEngine {
public:
    /**
     * @brief Raw scalar inputs the analytic formulas actually consume.
     *
     * Exposed as a nested struct so callers who don't want to build a
     * full `Option` + `MarketSnapshot` pair can price directly (research
     * notebooks, finite-difference test fixtures, calibrators that
     * iterate over many parameter tuples). The `(Option, MarketSnapshot)`
     * overload delegates to the `(Inputs)` overload after deriving
     * `time_to_expiry`.
     */
    struct Inputs {
        double spot{0.0};              ///< S, > 0.
        double strike{0.0};            ///< K, > 0.
        double rate{0.0};              ///< r, continuously compounded (decimal).
        double dividend_yield{0.0};    ///< q, continuously compounded (decimal).
        double volatility{0.0};        ///< sigma, annualized (decimal). >= 0.
        double time_to_expiry{0.0};    ///< T in years (ACT/365F when derived from dates). >= 0.
        ore::core::OptionType type{ore::core::OptionType::Call};
    };

    BlackScholesEngine() = default;

    /**
     * @copydoc PricingEngine::price
     *
     * Derives `T` from `market.valuation_date` and `option.expiration`
     * using ACT/365 fixed. Throws `std::invalid_argument` if the option
     * is American; delegates to the `Inputs`-overload for all other
     * validation.
     */
    [[nodiscard]] PricingResult price(
        const ore::core::Option& option,
        const ore::core::MarketSnapshot& market) const override;

    /**
     * @brief Price directly from raw scalar inputs.
     *
     * @param inputs  Model inputs. See `Inputs` for units and constraints.
     *
     * @return Populated `PricingResult` — price, all five Greeks in the
     *         units documented in `greeks.hpp`, and `engine_name =
     *         "BlackScholes"`. Optional diagnostics (`iterations`,
     *         `standard_error`) are left empty because BS is
     *         deterministic and closed-form.
     *
     * @throws std::invalid_argument  If `spot <= 0`, `strike <= 0`,
     *         `volatility < 0`, `time_to_expiry < 0`, or if any input is
     *         non-finite.
     */
    [[nodiscard]] PricingResult price(const Inputs& inputs) const;

    /** @copydoc PricingEngine::name */
    [[nodiscard]] std::string_view name() const noexcept override {
        return "BlackScholes";
    }
};

} // namespace ore::pricing
