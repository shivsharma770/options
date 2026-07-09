/**
 * @file quote.hpp
 * @brief Point-in-time market quote for a single instrument.
 *
 * Design notes
 * ------------
 * - A `Quote` describes *what the market is saying about a specific
 *   instrument at a specific time*. It has no knowledge of the contract
 *   itself; that is paired externally via `OptionMarketSnapshot`.
 * - Volume and open-interest are integer-valued in the market but are
 *   stored as `double` for interoperability with parsers that read them as
 *   floating-point (some CSV/Parquet feeds emit `NaN` for missing values).
 *   The tradeoff is intentional; if we later need bit-exact integer
 *   semantics we introduce a distinct type.
 */
#pragma once

#include <chrono>
#include <cmath>
#include <optional>

#include <ore/core/provider_greeks.hpp>

namespace ore::core {

/**
 * Bid/ask/last/size snapshot from a market data feed.
 */
struct Quote {
    double bid{0.0};                                 ///< Best bid price.
    double ask{0.0};                                 ///< Best ask price.
    double last{0.0};                                ///< Last traded price.
    double volume{0.0};                              ///< Traded volume for the session (contracts).
    double open_interest{0.0};                       ///< Open interest at the timestamp.
    /**
     * Provider-reported implied volatility, when published. Yahoo and most
     * commercial vendors publish this alongside the quote; some feeds
     * (deep OTM contracts, thin markets) omit it. `std::nullopt` means
     * "not published" — a genuinely-zero provider IV is stored as
     * `std::optional{0.0}` so the two cases stay unambiguously distinct
     * (a distinction pandas / CSV consumers rely on when computing
     * summary statistics).
     *
     * This field is *provider data*. The analytics module re-derives
     * implied volatility from `bid`/`ask` via the pricing engine and may
     * disagree with the provider — see `OptionChainCalibrator`.
     */
    std::optional<double> implied_volatility{};

    /**
     * Provider-reported first-order sensitivities, when published.
     * Populated for vendor feeds that include Greeks alongside quotes
     * (e.g. OptionsDX / Delta-Neutral end-of-day archives). Left as
     * `std::nullopt` for feeds that do not publish Greeks
     * (Yahoo Finance).
     *
     * Individual Greeks inside `ProviderGreeks` are themselves
     * optional — a vendor may publish some but not others for a given
     * contract. Research studies that compare provider Greeks against
     * computed values are expected to check `has_value()` per Greek
     * before consuming.
     *
     * Vendor unit conventions may differ from `ore::pricing::Greeks`
     * (which is per-year theta / per-unit-vol vega). See
     * `provider_greeks.hpp` for details.
     */
    std::optional<ProviderGreeks> provider_greeks{};

    std::chrono::system_clock::time_point timestamp{}; ///< When the quote was observed.

    /**
     * Convenience mid-price `(bid + ask) / 2`. Callers wanting robustness
     * against NaN or non-positive quotes must validate the fields
     * themselves — see the `bidask_finite()` / `is_crossed()` /
     * `is_unquoted()` predicates below, which are the pieces every
     * consumer previously re-implemented locally.
     */
    [[nodiscard]] constexpr double mid() const noexcept {
        return 0.5 * (bid + ask);
    }

    /**
     * True iff both `bid` and `ask` are finite (not NaN, not Inf).
     * Non-`constexpr` because `std::isfinite` is not constexpr in
     * C++20; the check itself is a couple of ALU ops.
     */
    [[nodiscard]] bool bidask_finite() const noexcept {
        return std::isfinite(bid) && std::isfinite(ask);
    }

    /**
     * True iff `bid == 0 && ask == 0`. Yahoo / OCC convention for a
     * listed contract with no market on it (deep-OTM strikes are
     * typically unquoted). Distinct from "both fields missing" —
     * missing fields would already have failed `bidask_finite()`.
     */
    [[nodiscard]] constexpr bool is_unquoted() const noexcept {
        return bid == 0.0 && ask == 0.0;
    }

    /**
     * True iff `bid > ask` — the classical crossed-market condition.
     * Meaningful only for finite quotes; callers that need to
     * distinguish "crossed and positive" from "crossed with a zero
     * side" should combine with `is_unquoted()` or explicit sign
     * checks.
     */
    [[nodiscard]] constexpr bool is_crossed() const noexcept {
        return bid > ask;
    }

    friend bool operator==(const Quote&, const Quote&) = default;
};

} // namespace ore::core
