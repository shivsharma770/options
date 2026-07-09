/**
 * @file option.hpp
 * @brief The `Option` value type — immutable contract information only.
 *
 * Design notes
 * ------------
 * - Contains only *contract* fields (underlying, strike, expiration date,
 *   payoff type, exercise style). Nothing about market state (spot, rates,
 *   volatility) lives here — that belongs in `MarketSnapshot`.
 * - Time-to-expiration is deliberately **not** stored. It is derived by
 *   pricing engines from `MarketSnapshot::valuation_date` so the same
 *   `Option` value can be priced at multiple valuation dates without
 *   mutation.
 * - The underlying is held by value (see `underlying.hpp` for ownership
 *   rationale).
 * - Plain aggregate value type: copyable, movable, defaultable, comparable.
 *   No inheritance. No `const` members (they would break assignability and
 *   give us nothing over simply not providing setters).
 */
#pragma once

#include <chrono>

#include <ore/core/types.hpp>
#include <ore/core/underlying.hpp>

namespace ore::core {

/**
 * A vanilla option contract.
 */
struct Option {
    Underlying underlying{};                            ///< Underlying asset the option is written on.
    double strike{0.0};                                 ///< Strike price, same currency as the underlying.
    std::chrono::year_month_day expiration{};           ///< Calendar expiration date (no time-of-day).
    OptionType type{OptionType::Call};                  ///< Call or put.
    ExerciseStyle exercise{ExerciseStyle::European};    ///< European (default) or American.

    /**
     * Structural equality. `year_month_day` and enum comparisons are
     * well-defined; floating-point strike comparison is exact-bit — callers
     * are responsible for any tolerance-based comparison.
     */
    friend bool operator==(const Option&, const Option&) = default;
};

} // namespace ore::core
