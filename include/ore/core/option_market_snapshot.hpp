/**
 * @file option_market_snapshot.hpp
 * @brief Pairing of a contract with its per-instrument market quote plus
 *        a provider-assigned identifier.
 *
 * Note that `OptionMarketSnapshot` does *not* carry the market-wide
 * `MarketSnapshot` (spot, rates, yield). That is passed alongside a
 * collection of `OptionMarketSnapshot` values via `OptionChain`.
 *
 * Composition is by value on purpose:
 * - `Option` and `Quote` are small and cheap to copy.
 * - Reference/pointer members would introduce lifetime bugs for zero gain.
 *
 * `contract_symbol` lives here (rather than on `Option`) because it is a
 * provider-assigned identifier for a specific listed instrument, not a
 * contract term. Two options with identical strike/expiration/type must
 * compare equal regardless of provider labelling.
 */
#pragma once

#include <string>

#include <ore/core/option.hpp>
#include <ore/core/quote.hpp>

namespace ore::core {

/**
 * A single option contract paired with its most recent market quote and
 * its provider-assigned identifier.
 */
struct OptionMarketSnapshot {
    Option      option{};          ///< The contract terms.
    Quote       quote{};           ///< The market's quote for that contract.
    std::string contract_symbol{}; ///< Provider-assigned identifier, e.g. "SPY260808C00470000". Empty if unknown.

    friend bool operator==(const OptionMarketSnapshot&, const OptionMarketSnapshot&) = default;
};

} // namespace ore::core
