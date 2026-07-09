/**
 * @file historical_snapshot.hpp
 * @brief One trading day's observation of a ticker — the unit of
 *        historical research in this library.
 *
 * A `HistoricalSnapshot` bundles:
 *   - the calendar date on which the data was observed,
 *   - the full `OptionChain` observed on that date (underlying + market
 *     state + all contracts).
 *
 * The class is a thin wrapper around `OptionChain` that adds a first-
 * class `date` field. The date is functionally the same as
 * `chain().market().valuation_date`, but exposing it explicitly makes
 * chronological ordering and lookup natural (`snapshot.date() <
 * other.date()`, `if (snapshot.date() == target)`, etc.) without
 * requiring callers to reach into the market snapshot.
 *
 * ### Immutability
 *
 * Held by value, no setters. Threads may share a `HistoricalSnapshot`
 * freely. This mirrors the design of `OptionChain` and every other
 * core data type in the project.
 *
 * ### Ownership of the option chain
 *
 * `OptionChain` is not copyable-cheap (it carries a
 * `std::vector<OptionMarketSnapshot>` which itself owns each contract's
 * underlying). We move the chain into the snapshot at construction time
 * and expose only const-references thereafter. `HistoricalDataset`
 * stores snapshots by value in a `std::vector`, so moves and reserves
 * are used judiciously — see `HistoricalDataset` for that story.
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <utility>
#include <vector>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option_chain.hpp>
#include <ore/core/option_market_snapshot.hpp>
#include <ore/core/underlying.hpp>

namespace ore::marketdata {

/**
 * @class HistoricalSnapshot
 * @brief One day of historical option-chain data for a single ticker.
 */
class HistoricalSnapshot {
public:
    /**
     * Construct from a date and a fully-loaded `OptionChain`. The
     * chain is moved into the snapshot.
     *
     * @param date  Trading date the chain was observed on. Callers are
     *              expected to pass the same value as
     *              `chain.market().valuation_date` — this class does
     *              not enforce the equality (fixtures with intentional
     *              mismatches show up as an integrity diagnostic in
     *              `HistoricalDataset::stats()` rather than as a hard
     *              construction failure).
     * @param chain The observed option chain, moved.
     */
    HistoricalSnapshot(std::chrono::year_month_day date,
                       ore::core::OptionChain      chain)
        : date_(date), chain_(std::move(chain)) {}

    /** Calendar date of the observation. */
    [[nodiscard]] std::chrono::year_month_day date() const noexcept { return date_; }

    /** Underlying identity (ticker, exchange, asset type). */
    [[nodiscard]] const ore::core::Underlying& underlying() const noexcept {
        return chain_.underlying();
    }

    /** Market-wide state at the observation moment. */
    [[nodiscard]] const ore::core::MarketSnapshot& market() const noexcept {
        return chain_.market();
    }

    /** Per-contract data (calls and puts, all expirations). */
    [[nodiscard]] const std::vector<ore::core::OptionMarketSnapshot>&
    options() const noexcept {
        return chain_.options();
    }

    /** The full underlying `OptionChain`, for callers that already
     *  work in terms of chains. */
    [[nodiscard]] const ore::core::OptionChain& chain() const noexcept { return chain_; }

    /** Number of contracts (calls + puts) in this snapshot. */
    [[nodiscard]] std::size_t size()  const noexcept { return chain_.size();  }
    /** `true` if the snapshot has no listed contracts. */
    [[nodiscard]] bool        empty() const noexcept { return chain_.empty(); }

    /** Range-based-for over the individual `OptionMarketSnapshot`s. */
    [[nodiscard]] auto begin() const noexcept { return chain_.begin(); }
    [[nodiscard]] auto end()   const noexcept { return chain_.end();   }

private:
    std::chrono::year_month_day date_{};
    ore::core::OptionChain      chain_;
};

} // namespace ore::marketdata
