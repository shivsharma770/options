/**
 * @file underlying.hpp
 * @brief The underlying asset a derivative contract is written on.
 *
 * `Underlying` deliberately carries only *identity* fields (symbol, exchange,
 * asset type). Live market data (spot, historical prices, dividend yield)
 * lives in `MarketSnapshot`, not here. This mirrors the same
 * contract-vs-market split already in place for `Option`.
 *
 * Ownership model: `Option` holds an `Underlying` **by value**. Short tickers
 * fit in `std::string`'s small-string optimisation, so an option chain of
 * hundreds of strikes for a single ticker costs a few kilobytes of
 * duplication — a non-issue at this project's scale. If profiling later
 * shows the duplication matters we can switch call sites over to
 * `std::shared_ptr<const Underlying>` without changing the type itself.
 */
#pragma once

#include <string>

namespace ore::core {

/**
 * Broad classification of an underlying asset. Extend as new instrument
 * types become relevant.
 */
enum class AssetType {
    Equity, ///< Common stock.
    Index,  ///< Market index (e.g., SPX, NDX).
    ETF,    ///< Exchange-traded fund.
    Future, ///< Futures contract.
    FX,     ///< Foreign exchange pair.
    Other   ///< Unclassified.
};

/**
 * Identifier of an underlying asset. All fields are optional in the sense
 * that they may be empty / defaulted; the market-data layer is responsible
 * for populating whatever it knows.
 */
struct Underlying {
    std::string symbol{};                       ///< Ticker symbol, e.g., "AAPL".
    std::string exchange{};                     ///< Listing exchange, e.g., "NASDAQ"; empty if unknown.
    AssetType   asset_type{AssetType::Equity};  ///< Broad asset class.

    friend bool operator==(const Underlying&, const Underlying&) = default;
};

} // namespace ore::core
