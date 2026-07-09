/**
 * @file comparison.hpp
 * @brief Floating-point comparison helpers for tests and convergence checks.
 *
 * Scope is deliberately narrow: three small `inline` free functions that
 * cover the comparison patterns we actually need. Not included and not
 * planned: ULPs-based comparison, componentwise container comparison,
 * SIMD helpers. Add them when we have a concrete second use case.
 *
 * These are `noexcept` and treat NaN correctly (NaN compared against
 * anything, including NaN, is "not equal" and produces NaN errors).
 */
#pragma once

#include <algorithm>
#include <cmath>

namespace ore::numerics {

/**
 * @brief \f$ |a - b| \f$.
 *
 * @note Returns NaN if either argument is NaN.
 */
[[nodiscard]] inline double absolute_error(double a, double b) noexcept {
    return std::abs(a - b);
}

/**
 * @brief \f$ \frac{|a - b|}{\max(|a|, |b|)} \f$, with the zero-divisor case
 *        defined so that `relative_error(0, 0) == 0`.
 *
 * The scaling denominator is `max(|a|, |b|)` rather than `|a|` or `|b|`
 * alone so the metric is symmetric in `a` and `b`. When both operands are
 * exactly zero we return `0.0` — otherwise the natural definition would
 * be 0/0, which is undefined.
 *
 * @note Returns NaN if either argument is NaN.
 */
[[nodiscard]] inline double relative_error(double a, double b) noexcept {
    const double denom = std::max(std::abs(a), std::abs(b));
    if (denom == 0.0) return 0.0;
    return std::abs(a - b) / denom;
}

/**
 * @brief Robust "close enough" test combining absolute and relative
 *        tolerances.
 *
 * Returns `true` iff
 * \f[ |a - b| \le \text{abs\_tol} + \text{rel\_tol} \cdot \max(|a|, |b|). \f]
 *
 * The combined form handles both edge cases gracefully:
 * - When `a` and `b` are close to zero, `abs_tol` dominates (relative
 *   error is meaningless for tiny numbers).
 * - When `a` and `b` are large, `rel_tol * max(|a|, |b|)` dominates
 *   (absolute error is uninformative when the magnitude is large).
 *
 * @param a        First value.
 * @param b        Second value.
 * @param abs_tol  Absolute tolerance. Must be `>= 0`.
 * @param rel_tol  Relative tolerance. Must be `>= 0`.
 *
 * @return `true` iff the values agree within the combined tolerance.
 *         Returns `false` if either input is non-finite (NaN or +/-inf).
 */
[[nodiscard]] inline bool approximately_equal(
    double a,
    double b,
    double abs_tol,
    double rel_tol) noexcept
{
    // Reject any non-finite operand. NaN already compares false to
    // everything, but an infinite operand must be rejected explicitly:
    // otherwise the combined bound below degenerates to `inf <= inf`
    // (both `diff` and `scale` become +inf), which is `true` and would
    // wrongly report e.g. `approximately_equal(inf, 1.0, ...)`.
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    const double diff = std::abs(a - b);
    const double scale = std::max(std::abs(a), std::abs(b));
    return diff <= abs_tol + rel_tol * scale;
}

} // namespace ore::numerics
