/**
 * @file newton_raphson.hpp
 * @brief Newton-Raphson (secant-with-analytic-derivative) scalar root finder.
 *
 * ### Algorithm
 * Given a differentiable \f$ f : \mathbb{R} \to \mathbb{R} \f$ with
 * analytic derivative \f$ f' \f$ and an initial guess \f$ x_0 \f$, iterate
 * \f[
 *   x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
 * \f]
 * until \f$ |f(x_n)| < \text{tolerance} \f$ or a stopping condition fires.
 *
 * ### Assumptions
 * - `f` and `fprime` are finite in a neighbourhood of the true root.
 * - `fprime` does not vanish near the root — otherwise iteration diverges
 *   or amplifies rounding error. The solver detects `|f'(x)| < min_derivative`
 *   and reports `SolverStatus::DerivativeTooSmall`.
 * - `x0` is close enough to the true root for the quadratic-convergence
 *   basin. For arbitrary starting points use `BisectionSolver` instead
 *   (linear but globally robust).
 *
 * ### Convergence
 * - **Quadratic** near a simple root: the number of correct digits
 *   roughly doubles each iteration, so `max_iterations = 100` is
 *   effectively "never" for well-posed problems.
 * - Sensitive to a bad initial guess. Divergence typically shows up as
 *   `NonFiniteEvaluation` or `MaxIterationsReached`.
 *
 * ### Complexity
 * O(log log(1/ε)) function+derivative evaluations for a well-posed
 * problem — one `f` and one `f'` evaluation per iteration.
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <ore/numerics/solver_result.hpp>

namespace ore::numerics {

/**
 * @class NewtonRaphsonSolver
 * @brief Configurable Newton-Raphson scalar solver.
 *
 * The solver is stateless — a single instance can be reused across many
 * calls, on different callables, from multiple threads.
 */
class NewtonRaphsonSolver {
public:
    /**
     * @brief Configuration knobs for a solve.
     *
     * All fields are validated on entry to `solve()`; invalid config
     * throws `std::invalid_argument` before the user function is called.
     */
    struct Config {
        /** Absolute residual tolerance: converged iff \f$ |f(x)| < \text{tolerance} \f$. Must be `> 0`. */
        double      tolerance{1e-10};

        /** Hard iteration cap. Must be `> 0`. */
        std::size_t max_iterations{100};

        /** Absolute floor on \f$ |f'(x)| \f$. If violated the solver
         *  returns `DerivativeTooSmall`; the step \f$ f/f' \f$ would
         *  otherwise blow up or lose all precision. */
        double      min_derivative{1e-14};
    };

    /** Default-configured solver. */
    NewtonRaphsonSolver() = default;

    /** Construct with a specific configuration. */
    explicit NewtonRaphsonSolver(Config config) : config_(config) {}

    /** Access the immutable configuration. */
    [[nodiscard]] const Config& config() const noexcept { return config_; }

    /**
     * @brief Run Newton-Raphson from `x0`.
     *
     * @tparam Fn  Callable `double(double)` — the function whose root is sought.
     * @tparam Fp  Callable `double(double)` — its analytic derivative.
     *
     * @param f       Function whose root is sought.
     * @param fprime  Derivative \f$ f'(x) \f$.
     * @param x0      Initial guess.
     *
     * @return A populated `SolverResult`. `.converged()` is `true` iff
     *         `.status == SolverStatus::Converged`.
     *
     * @throws std::invalid_argument if `x0` is non-finite, or if the
     *         configured `tolerance <= 0`, or if `max_iterations == 0`.
     */
    template <typename Fn, typename Fp>
    [[nodiscard]] SolverResult solve(Fn&& f, Fp&& fprime, double x0) const {
        validate_config();
        if (!std::isfinite(x0)) {
            throw std::invalid_argument("NewtonRaphsonSolver::solve: x0 must be finite");
        }

        double x  = x0;
        double fx = f(x);
        if (!std::isfinite(fx)) {
            return {x, 0, SolverStatus::NonFiniteEvaluation, std::abs(fx)};
        }

        for (std::size_t i = 0; i < config_.max_iterations; ++i) {
            if (std::abs(fx) < config_.tolerance) {
                return {x, i, SolverStatus::Converged, std::abs(fx)};
            }

            const double fpx = fprime(x);
            if (!std::isfinite(fpx)) {
                return {x, i, SolverStatus::NonFiniteEvaluation, std::abs(fx)};
            }
            if (std::abs(fpx) < config_.min_derivative) {
                return {x, i, SolverStatus::DerivativeTooSmall, std::abs(fx)};
            }

            x  = x - fx / fpx;
            fx = f(x);
            if (!std::isfinite(fx)) {
                return {x, i + 1, SolverStatus::NonFiniteEvaluation, std::abs(fx)};
            }
        }

        return {x, config_.max_iterations, SolverStatus::MaxIterationsReached, std::abs(fx)};
    }

private:
    void validate_config() const {
        if (!(config_.tolerance > 0.0)) {
            throw std::invalid_argument("NewtonRaphsonSolver: tolerance must be > 0");
        }
        if (config_.max_iterations == 0) {
            throw std::invalid_argument("NewtonRaphsonSolver: max_iterations must be > 0");
        }
        if (!(config_.min_derivative >= 0.0)) {
            throw std::invalid_argument("NewtonRaphsonSolver: min_derivative must be >= 0");
        }
    }

    Config config_{};
};

} // namespace ore::numerics
