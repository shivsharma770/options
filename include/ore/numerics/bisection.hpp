/**
 * @file bisection.hpp
 * @brief Bisection scalar root finder.
 *
 * ### Algorithm
 * Given a continuous \f$ f : [a, b] \to \mathbb{R} \f$ with
 * \f$ f(a) \cdot f(b) < 0 \f$ (i.e. a sign change on the bracket), repeatedly
 * evaluate \f$ f \f$ at the midpoint and shrink the bracket to the sub-
 * interval that still brackets a sign change. The interval width halves
 * every iteration, so after \f$ n \f$ iterations the root is localised to
 * within \f$ (b - a) / 2^n \f$.
 *
 * ### Convergence
 * - **Linear** — one correct bit per iteration. Guaranteed to converge
 *   for any continuous function on a valid bracket. Much slower than
 *   Newton-Raphson when both are applicable, but has no failure modes
 *   beyond "the bracket isn't valid" or "the function returned NaN".
 * - Preferred over Newton when you have no good initial guess but you
 *   can bracket the root (e.g. implied-volatility solves on
 *   \f$ \sigma \in [10^{-4}, 5] \f$).
 *
 * ### Complexity
 * O(log_2((b - a) / ε)) function evaluations for absolute tolerance ε on
 * the root, or the residual-tolerance equivalent, whichever fires first.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include <ore/numerics/solver_result.hpp>

namespace ore::numerics {

/**
 * @class BisectionSolver
 * @brief Configurable bisection scalar solver.
 *
 * The solver is stateless — a single instance can be reused across many
 * calls, on different callables, from multiple threads.
 */
class BisectionSolver {
public:
    /**
     * @brief Configuration knobs for a solve.
     *
     * All fields are validated on entry to `solve()`; invalid config
     * throws `std::invalid_argument` before the user function is called.
     */
    struct Config {
        /** Absolute residual tolerance: converged iff
         *  \f$ |f(\text{midpoint})| < \text{tolerance} \f$. Must be `> 0`. */
        double      tolerance{1e-10};

        /** Hard iteration cap. Must be `> 0`. Bisection needs roughly
         *  \f$ \log_2((b - a) / \text{tolerance}) \f$ iterations to
         *  narrow the bracket to `tolerance`; the default of 200 is
         *  effectively "never" for any reasonable input. */
        std::size_t max_iterations{200};
    };

    /** Default-configured solver. */
    BisectionSolver() = default;

    /** Construct with a specific configuration. */
    explicit BisectionSolver(Config config) : config_(config) {}

    /** Access the immutable configuration. */
    [[nodiscard]] const Config& config() const noexcept { return config_; }

    /**
     * @brief Bisect for a root of `f` in `[a, b]`.
     *
     * @tparam Fn  Callable `double(double)`.
     *
     * @param f  Function whose root is sought.
     * @param a  One endpoint of the bracket.
     * @param b  The other endpoint. `a` and `b` are automatically ordered;
     *           the algorithm assumes `f(a) * f(b) < 0` and returns
     *           `SolverStatus::InvalidBracket` if that fails.
     *
     * @return A populated `SolverResult`. `.converged()` is `true` iff
     *         `.status == SolverStatus::Converged`.
     *
     * @throws std::invalid_argument if `a` or `b` is non-finite, if
     *         `a == b`, if the configured `tolerance <= 0`, or if
     *         `max_iterations == 0`.
     */
    template <typename Fn>
    [[nodiscard]] SolverResult solve(Fn&& f, double a, double b) const {
        validate_config();
        if (!std::isfinite(a) || !std::isfinite(b)) {
            throw std::invalid_argument("BisectionSolver::solve: bracket endpoints must be finite");
        }
        if (a == b) {
            throw std::invalid_argument("BisectionSolver::solve: bracket endpoints must differ");
        }
        if (a > b) std::swap(a, b);

        double fa = f(a);
        double fb = f(b);
        if (!std::isfinite(fa) || !std::isfinite(fb)) {
            return {0.5 * (a + b), 0, SolverStatus::NonFiniteEvaluation, 0.0};
        }

        // Cheap zeros-at-endpoint short-circuit — no reason to bisect.
        if (std::abs(fa) < config_.tolerance) return {a, 0, SolverStatus::Converged, std::abs(fa)};
        if (std::abs(fb) < config_.tolerance) return {b, 0, SolverStatus::Converged, std::abs(fb)};

        // Reject same-sign endpoints — bisection cannot guarantee a root.
        if ((fa > 0.0) == (fb > 0.0)) {
            return {0.5 * (a + b), 0, SolverStatus::InvalidBracket, 0.0};
        }

        double mid = 0.5 * (a + b);
        double fm  = 0.0;

        for (std::size_t i = 0; i < config_.max_iterations; ++i) {
            mid = 0.5 * (a + b);
            fm  = f(mid);
            if (!std::isfinite(fm)) {
                return {mid, i, SolverStatus::NonFiniteEvaluation, std::abs(fm)};
            }
            if (std::abs(fm) < config_.tolerance) {
                return {mid, i + 1, SolverStatus::Converged, std::abs(fm)};
            }
            // If the bracket collapses below what `double` can distinguish,
            // treat it as converged at floating-point precision. Prevents
            // a needless run to `max_iterations` for functions whose
            // absolute-value floor is above `tolerance`. Two collapse
            // conditions: `mid == a || mid == b` catches exact adjacency
            // (a symmetric bracket around the root); the width floor also
            // catches the case where the sign change sits *on* an endpoint
            // (e.g. a step function bracketed as [a, 0]), where the
            // midpoint keeps halving toward the fixed endpoint without ever
            // coinciding with it until subnormal underflow ~1074 iterations
            // later.
            const double width_floor =
                std::numeric_limits<double>::epsilon()
                * std::max({std::abs(a), std::abs(b), 1.0});
            if (mid == a || mid == b || (b - a) <= width_floor) {
                return {mid, i + 1, SolverStatus::Converged, std::abs(fm)};
            }

            if ((fm > 0.0) == (fa > 0.0)) {
                a  = mid;
                fa = fm;
            } else {
                b  = mid;
                fb = fm;
            }
        }

        return {mid, config_.max_iterations, SolverStatus::MaxIterationsReached, std::abs(fm)};
    }

private:
    void validate_config() const {
        if (!(config_.tolerance > 0.0)) {
            throw std::invalid_argument("BisectionSolver: tolerance must be > 0");
        }
        if (config_.max_iterations == 0) {
            throw std::invalid_argument("BisectionSolver: max_iterations must be > 0");
        }
    }

    Config config_{};
};

} // namespace ore::numerics
