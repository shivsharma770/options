/**
 * @file market_snapshot.hpp
 * @brief Market-wide inputs used by pricing engines.
 *
 * Rate, yield & volatility conventions (applied consistently across the
 * project):
 *
 * - `risk_free_rate` — **continuously compounded**, expressed as a decimal
 *   (0.05, not 5.0), quoted on an **ACT/365F** basis (matching the year
 *   fraction used by pricing engines).
 * - `dividend_yield` — same conventions: continuous, decimal, ACT/365F.
 * - `volatility`     — annualized standard deviation of log-returns,
 *   decimal (0.20 = 20%). Constant-parameter models (Black-Scholes) read
 *   this directly; when we later introduce term-structures or vol
 *   surfaces the scalar here becomes the *fallback* used when no surface
 *   is supplied.
 *
 * These conventions are *by declaration*: pricing engines assume them, and
 * the market-data layer is responsible for converting any externally sourced
 * rates into this convention.
 */
#pragma once

#include <chrono>

namespace ore::core {

/**
 * Market state on a particular valuation date.
 */
struct MarketSnapshot {
    double spot{0.0};                              ///< Spot price of the underlying.
    double risk_free_rate{0.0};                    ///< Continuous risk-free rate (decimal, ACT/365F).
    double dividend_yield{0.0};                    ///< Continuous dividend yield (decimal, ACT/365F).
    double volatility{0.0};                        ///< Annualized volatility of log-returns (decimal).
    std::chrono::year_month_day valuation_date{};  ///< Observation date; pricing engines combine this with `Option::expiration` to derive time-to-maturity.

    friend bool operator==(const MarketSnapshot&, const MarketSnapshot&) = default;
};

} // namespace ore::core
