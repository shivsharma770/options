/**
 * @file types.hpp
 * @brief Small, shared enums for the core module.
 *
 * These types describe *contract properties* of an option — they never
 * depend on market state or on any pricing model. Kept in a single header so
 * callers only need one include for the common enums.
 */
#pragma once

#include <string_view>

namespace ore::core {

/**
 * Payoff type of a vanilla option.
 */
enum class OptionType {
    Call, ///< Right to buy the underlying at the strike.
    Put   ///< Right to sell the underlying at the strike.
};

/**
 * Exercise style of an option. Placed on the contract because it is a
 * property of the *instrument*, not of any pricing model. Every future
 * pricing engine (Black-Scholes, binomial, MC, FD) reads this to decide
 * how to price the contract.
 */
enum class ExerciseStyle {
    European, ///< Exercisable only at expiration.
    American  ///< Exercisable at any time up to and including expiration.
};

/** Human-readable name of an `OptionType`, useful for logs and diagnostics. */
[[nodiscard]] constexpr std::string_view to_string(OptionType t) noexcept {
    switch (t) {
        case OptionType::Call: return "Call";
        case OptionType::Put:  return "Put";
    }
    return "Unknown";
}

/** Human-readable name of an `ExerciseStyle`. */
[[nodiscard]] constexpr std::string_view to_string(ExerciseStyle s) noexcept {
    switch (s) {
        case ExerciseStyle::European: return "European";
        case ExerciseStyle::American: return "American";
    }
    return "Unknown";
}

} // namespace ore::core
