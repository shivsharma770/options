/**
 * @file normal_distribution.hpp
 * @brief Standard normal distribution — PDF, CDF, and inverse CDF.
 *
 * Provides the three evaluations that every option-pricing / risk /
 * Monte-Carlo path eventually needs:
 *   - \f$ \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \f$          (PDF)
 *   - \f$ \Phi(x) = \int_{-\infty}^{x} \phi(t)\, dt \f$           (CDF)
 *   - \f$ \Phi^{-1}(p) \f$ such that \f$ \Phi(\Phi^{-1}(p)) = p \f$ (inverse CDF / quantile)
 *
 * See the .cpp for the numerical rationale — in particular why the CDF is
 * expressed via `std::erfc` rather than `1 - std::erf`, and which
 * approximation the inverse CDF uses (Wichura AS241).
 *
 * Static-function class rather than free functions: keeps the
 * `NormalDistribution::cdf(x)` call-site pattern the project uses, and
 * gives us a single, discoverable home for further named distribution
 * functions (samplers, log-density, tail expectations) later without
 * polluting the namespace.
 */
#pragma once

namespace ore::numerics {

/**
 * @class NormalDistribution
 * @brief Standard normal distribution \f$ \mathcal{N}(0, 1) \f$.
 *
 * ### Assumptions
 * All functions take an unrestricted `double`. Non-finite input propagates
 * to non-finite output; the caller is responsible for validating input if
 * they need that.
 *
 * ### Accuracy
 * Backed by the C++ standard library's `std::exp` and `std::erfc`. On
 * MSVC, libstdc++, and libc++ these are accurate to within ~1 ULP over
 * the entire representable range. That is significantly better than any
 * hand-rolled polynomial approximation and is exactly what production
 * quant-finance libraries (QuantLib, Boost.Math) rely on.
 *
 * ### Complexity
 * O(1) — a small, fixed number of transcendental evaluations per call.
 */
class NormalDistribution {
public:
    NormalDistribution() = delete;

    /**
     * @brief Probability density \f$ \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \f$.
     *
     * @param x  Point at which to evaluate the density.
     * @return \f$ \phi(x) \f$, which is always non-negative and finite for
     *         finite `x`. For very large `|x|` the result underflows to
     *         `0.0`, which is the correct IEEE-754 behaviour.
     * @note   Symmetric: `pdf(x) == pdf(-x)` (bitwise, since it depends
     *         only on `x*x`).
     */
    [[nodiscard]] static double pdf(double x) noexcept;

    /**
     * @brief Cumulative distribution \f$ \Phi(x) = P(Z \le x) \f$ for
     *        \f$ Z \sim \mathcal{N}(0, 1) \f$.
     *
     * Computed as `0.5 * std::erfc(-x / sqrt(2))`. The `erfc` form is used
     * (rather than `0.5 * (1 + erf(x/sqrt(2)))`) to avoid catastrophic
     * cancellation in the tail: for large positive `x`, `erf(x/sqrt(2))`
     * is close to 1, and computing `1 - erf(...)` loses most of the
     * significant digits. `erfc` computes the complementary error function
     * directly, preserving accuracy in the tail — the region deep-ITM /
     * deep-OTM option pricing lives in.
     *
     * @param x  Point at which to evaluate the CDF.
     * @return \f$ \Phi(x) \in [0, 1] \f$. Monotone non-decreasing.
     * @note   Symmetry: `cdf(-x) + cdf(x) == 1.0` up to rounding.
     */
    [[nodiscard]] static double cdf(double x) noexcept;

    /**
     * @brief Quantile / inverse CDF: the value `x` such that
     *        `cdf(x) == p`.
     *
     * Implemented with Wichura's AS241 algorithm (Applied Statistics vol.
     * 37, no. 3, 1988): a piecewise rational approximation in three
     * regions (central, moderate tail, extreme tail). Accuracy is ~1 ULP
     * over the full representable range — the same algorithm R's `qnorm`
     * uses. See the .cpp for coefficients and the region boundaries.
     *
     * ### Complexity
     * O(1) — a fixed number of multiply-adds plus one `sqrt` and one
     * `log` in the tail branches.
     *
     * ### Primary uses
     * - Monte-Carlo path generation via the inverse-CDF transform of
     *   uniform variates (\f$ Z = \Phi^{-1}(U) \f$, \f$ U \sim
     *   \text{Uniform}(0, 1) \f$).
     * - Confidence-interval endpoints (e.g. `inverse_cdf(0.975) ≈ 1.96`).
     * - Bracket generation for implied-volatility root finding.
     *
     * @param p  Probability in the open interval `(0, 1)`.
     * @return   The quantile \f$ \Phi^{-1}(p) \f$.
     *           - `p == 0.0` → \f$ -\infty \f$
     *           - `p == 1.0` → \f$ +\infty \f$
     *           - `p < 0`, `p > 1`, or NaN → NaN
     * @note     Symmetry: `inverse_cdf(1 - p) == -inverse_cdf(p)` up to
     *           rounding.
     */
    [[nodiscard]] static double inverse_cdf(double p) noexcept;
};

} // namespace ore::numerics
