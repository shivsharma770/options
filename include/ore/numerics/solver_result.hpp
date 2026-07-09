/**
 * @file solver_result.hpp
 * @brief Common return type for scalar root-finding algorithms.
 *
 * Both `NewtonRaphsonSolver` and `BisectionSolver` return a `SolverResult`.
 * The design rationale:
 *
 * - **Non-convergence is not an exception.** A solver hitting
 *   `max_iterations` without meeting the tolerance is a legitimate
 *   outcome the caller may want to react to (retry with a different
 *   initial guess, widen the bracket, tighten tolerance, ...). Forcing
 *   the caller into a try/catch for that would be heavy and hide intent.
 * - **The result carries diagnostics.** `iterations` and `residual` let
 *   the caller decide whether an "almost-converged" result is good
 *   enough, and reproducibly compare algorithm performance.
 * - **Preconditions still throw.** Non-finite initial guesses, non-
 *   positive tolerance, zero max_iterations — these are programmer
 *   errors and are surfaced via `std::invalid_argument` from the solver
 *   itself. See each solver's documentation.
 */
#pragma once

#include <cstddef>
#include <string_view>

namespace ore::numerics {

/**
 * @brief Classification of a scalar solver's termination state.
 *
 * `Converged` is the only "success" outcome; every other value indicates
 * a specific reason the algorithm stopped without finding a root to the
 * requested tolerance.
 */
enum class SolverStatus {
    /** The algorithm produced a value whose residual meets the tolerance. */
    Converged,

    /** The algorithm exhausted `max_iterations` without meeting tolerance. */
    MaxIterationsReached,

    /** Bisection only: `f(a)` and `f(b)` have the same sign, so no root
     *  can be guaranteed inside `[a, b]`. */
    InvalidBracket,

    /** Newton only: `f'(x)` fell below `min_derivative` — the step would
     *  either diverge or lose all precision. */
    DerivativeTooSmall,

    /** The user's function returned `NaN` or `Inf` during iteration. */
    NonFiniteEvaluation,
};

/**
 * @brief Result of a single call to a scalar solver.
 *
 * @note `root` is the best estimate the algorithm produced regardless of
 *       status — it is well-defined even for `MaxIterationsReached` and
 *       friends. `residual` is the absolute value of the user function at
 *       `root` at termination.
 */
struct SolverResult {
    double        root{0.0};                              ///< Best estimate at termination.
    std::size_t   iterations{0};                          ///< Number of iterations consumed.
    SolverStatus  status{SolverStatus::MaxIterationsReached}; ///< How the algorithm terminated.
    double        residual{0.0};                          ///< \f$ |f(\text{root})| \f$ at termination.

    /** Convenience predicate: `true` iff `status == SolverStatus::Converged`. */
    [[nodiscard]] constexpr bool converged() const noexcept {
        return status == SolverStatus::Converged;
    }
};

/**
 * @brief Human-readable name of a `SolverStatus`, for logging and test
 *        assertions.
 */
[[nodiscard]] constexpr std::string_view to_string(SolverStatus s) noexcept {
    switch (s) {
        case SolverStatus::Converged:            return "Converged";
        case SolverStatus::MaxIterationsReached: return "MaxIterationsReached";
        case SolverStatus::InvalidBracket:       return "InvalidBracket";
        case SolverStatus::DerivativeTooSmall:   return "DerivativeTooSmall";
        case SolverStatus::NonFiniteEvaluation:  return "NonFiniteEvaluation";
    }
    return "Unknown";
}

} // namespace ore::numerics
