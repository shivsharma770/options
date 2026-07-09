/**
 * @file pricing_engine.hpp
 * @brief Abstract interface every concrete pricing model implements.
 *
 * Design notes
 * ------------
 * - The engine takes an `Option` (contract) and a `MarketSnapshot` (market
 *   state). Neither embeds the other. Time-to-maturity is derived inside
 *   the concrete engine from `option.expiration` and
 *   `market.valuation_date`.
 * - Engines are stateless (or hold only *configuration*, never market
 *   data). The `price` method is `const`. This makes engines trivially
 *   thread-safe.
 * - Protected special member functions on the base prevent object slicing
 *   while still allowing derived classes to be copyable/movable — a
 *   standard modern-C++ idiom for polymorphic base classes.
 */
#pragma once

#include <string_view>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/pricing/pricing_result.hpp>

namespace ore::pricing {

/**
 * Abstract base for all pricing models (Black-Scholes, binomial, MC, FD, ...).
 */
class PricingEngine {
public:
    virtual ~PricingEngine() = default;

    /**
     * Compute the price of `option` given market state `market`.
     * Concrete engines document their required assumptions (e.g. European
     * exercise, constant volatility, no dividends beyond `dividend_yield`,
     * etc.).
     */
    [[nodiscard]] virtual PricingResult price(
        const ore::core::Option& option,
        const ore::core::MarketSnapshot& market) const = 0;

    /**
     * Short human-readable name of the engine, useful for logging and
     * model-comparison analytics. Should be stable across runs.
     */
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

protected:
    PricingEngine() = default;
    PricingEngine(const PricingEngine&) = default;
    PricingEngine(PricingEngine&&) noexcept = default;
    PricingEngine& operator=(const PricingEngine&) = default;
    PricingEngine& operator=(PricingEngine&&) noexcept = default;
};

} // namespace ore::pricing
