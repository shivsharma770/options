/**
 * @file option_chain.hpp
 * @brief The fundamental unit of market data consumed by the engine.
 *
 * An `OptionChain` bundles the three pieces of information the pricing and
 * analytics layers need together:
 *   - `Underlying`    — which asset the options are written on.
 *   - `MarketSnapshot` — market state at the observation moment (spot,
 *                       rates, dividend yield, valuation date).
 *   - `std::vector<OptionMarketSnapshot>` — every listed contract at the
 *                       snapshot (calls and puts combined, all expirations).
 *
 * The class holds its data by value and is immutable after construction:
 * accessors return `const` references but there are no setters. This makes
 * the chain trivially safe to pass around and share across threads.
 */
#pragma once

#include <utility>
#include <vector>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option_market_snapshot.hpp>
#include <ore/core/underlying.hpp>

namespace ore::core {

/**
 * Full option chain: one underlying, one market snapshot, many contracts.
 */
class OptionChain {
public:
    OptionChain(Underlying underlying,
                MarketSnapshot market,
                std::vector<OptionMarketSnapshot> options)
        : underlying_(std::move(underlying)),
          market_(std::move(market)),
          options_(std::move(options)) {}

    [[nodiscard]] const Underlying&                          underlying() const noexcept { return underlying_; }
    [[nodiscard]] const MarketSnapshot&                      market()     const noexcept { return market_; }
    [[nodiscard]] const std::vector<OptionMarketSnapshot>&   options()    const noexcept { return options_; }

    [[nodiscard]] std::size_t size()  const noexcept { return options_.size(); }
    [[nodiscard]] bool        empty() const noexcept { return options_.empty(); }

    // Range-based-for support directly on the chain, so callers can write
    // `for (const auto& snap : chain) { ... }` in addition to iterating
    // `chain.options()` explicitly.
    [[nodiscard]] auto begin() const noexcept { return options_.begin(); }
    [[nodiscard]] auto end()   const noexcept { return options_.end();   }

private:
    Underlying                          underlying_;
    MarketSnapshot                      market_;
    std::vector<OptionMarketSnapshot>   options_;
};

} // namespace ore::core
