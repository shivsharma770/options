/**
 * @file constants.hpp
 * @brief Mathematical constants for the numerics module.
 *
 * Only constants that C++20's `<numbers>` does not already provide, and
 * that are (or will be) reused across multiple numerics/finance
 * components. Prefer `std::numbers::pi_v<double>`, `std::numbers::sqrt2_v`,
 * `std::numbers::e_v` directly from the standard library for those.
 *
 * All values are `inline constexpr double` — they participate in constant
 * evaluation and do not violate the ODR when included in multiple TUs.
 * Digits are given to more than double precision so the compiler's
 * rounding produces the correctly-rounded `double` value.
 */
#pragma once

namespace ore::numerics::constants {

/** @brief \f$ 2\pi \f$ — used by any Gaussian density and by uniform-angle sampling. */
inline constexpr double two_pi = 6.283185307179586476925286766559005768;

/** @brief \f$ \sqrt{2\pi} \f$ — used by the standard-normal PDF. */
inline constexpr double sqrt_two_pi = 2.506628274631000502415765284811045253;

/** @brief \f$ \frac{1}{\sqrt{2\pi}} \f$ — used by the standard-normal PDF (avoids a division). */
inline constexpr double inv_sqrt_two_pi = 0.398942280401432677939946059934381868;

} // namespace ore::numerics::constants
