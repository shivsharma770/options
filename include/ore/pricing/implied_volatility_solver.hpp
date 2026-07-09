/**
 * @file implied_volatility_solver.hpp
 * @brief Recover the volatility implied by an observed European-option price.
 *
 * ### Why implied volatility exists
 * Volatility is the **only** Black-Scholes input that cannot be observed
 * directly in the market. Spot, strike, rate, dividend yield, and time
 * to expiration are all quoted or contract-defined. What the market
 * quotes is not the volatility itself but the option's *price*, and the
 * volatility that would make the closed-form model reproduce that price
 * (the "implied volatility") is treated as a summary statistic of the
 * option — one number per contract that traders, risk desks, and vol-
 * surface fitters use as their common language.
 *
 * ### Why there is no closed-form solution
 * The Black-Scholes call price is
 * \f[
 *   C(\sigma) = S e^{-qT} \Phi(d_1(\sigma)) - K e^{-rT} \Phi(d_2(\sigma))
 * \f]
 * with \f$ d_1 \f$ a non-elementary function of \f$ \sigma \f$ (via
 * `log(S/K)` and `sigma * sqrt(T)`) and \f$ \Phi \f$ the standard normal
 * CDF (itself transcendental — no elementary inverse). Composing them
 * makes \f$ C(\sigma) \f$ a transcendental function whose inverse
 * cannot be written in elementary form. See Manaster & Koehler (1982),
 * "The Calculation of Implied Variances from the Black-Scholes Model:
 * A Note", *Journal of Finance* 37(1), for a formal treatment.
 *
 * ### Why Newton-Raphson uses analytical Vega
 * `BlackScholesEngine::price(...)` returns Vega as part of the same
 * evaluation that produces the price — the two share `d_1`, `phi(d_1)`,
 * `exp(-qT)`, and `sqrt(T)`. Using the analytical Vega gives full
 * double precision at zero extra cost, and lets Newton keep its
 * quadratic convergence: numerical finite differences would introduce
 * O(h^2) truncation error and force Newton down to a slower regime.
 *
 * ### Why we need a fallback method
 * Newton is *fast* (5-7 iterations typical to 1e-10 residual) but not
 * globally robust:
 *   - **Vega collapses** for deep-ITM/OTM options (`phi(d_1) -> 0`),
 *     making `f / f'` explode.
 *   - **Poor initial guesses** can send early iterates into negative
 *     or absurdly-large regions.
 *   - **Prices near arbitrage bounds** yield tiny residuals; Newton
 *     can chatter around the answer.
 * Bisection, by contrast, is unconditionally convergent on a bracket
 * that contains a sign change, and BS is monotone in `sigma` so a
 * `[sigma_low, sigma_high]` bracket is always available. Bisection is
 * slower per-iteration (linear convergence, ~1 bit per iteration) but
 * only runs when Newton has already given up — so its cost is bounded.
 */
#pragma once

#include <cstddef>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/numerics/solver_result.hpp>
#include <ore/pricing/black_scholes_engine.hpp>

namespace ore::pricing {

/**
 * @brief Which numerical method actually produced a converged implied vol.
 *
 * Diagnostic only — callers who only need the numerical answer can ignore
 * this. The value is meaningful even when the solve did not converge:
 * `Newton` means Newton was the last method attempted, `Bisection` means
 * bisection was invoked (either as the fallback or, if Newton failed
 * fast, as the only method).
 */
enum class ImpliedVolatilityMethod {
    /** The Newton-Raphson primary solver converged (or was the only method run). */
    Newton,
    /** Newton failed or bailed; bisection was invoked. */
    Bisection,
};

/**
 * @brief Result of a single implied-volatility solve.
 *
 * Field-compatible with `ore::numerics::SolverResult` (all names and
 * meanings match) so callers who ignore `method` do not need to change
 * anything. `method` adds one bit of diagnostic information: whether the
 * value came out of Newton-Raphson or the bisection fallback. Useful for
 * whole-chain analytics that want to report how often the primary method
 * suffices.
 */
struct ImpliedVolatilityResult {
    /** Best sigma estimate at termination (best root regardless of status). */
    double root{0.0};

    /** Iterations consumed by the *winning* method (Newton, or bisection if used). */
    std::size_t iterations{0};

    /** How the algorithm terminated. See `ore::numerics::SolverStatus`. */
    ore::numerics::SolverStatus status{ore::numerics::SolverStatus::MaxIterationsReached};

    /** \f$ |BS(\text{root}) - \text{market\_price}| \f$ at termination. */
    double residual{0.0};

    /** Which numerical method produced this result. */
    ImpliedVolatilityMethod method{ImpliedVolatilityMethod::Newton};

    /** Convenience predicate — `true` iff `status == Converged`. */
    [[nodiscard]] constexpr bool converged() const noexcept {
        return status == ore::numerics::SolverStatus::Converged;
    }
};

/**
 * @class ImpliedVolatilitySolver
 * @brief Recovers the volatility implied by an observed option price.
 *
 * Stateless — a single instance can be reused across many pricing calls
 * from multiple threads. The solver internally instantiates a
 * `BlackScholesEngine` on each `solve()` call, so no engine reference
 * needs to be supplied.
 */
class ImpliedVolatilitySolver {
public:
    /**
     * @brief Configuration knobs for a solve.
     *
     * Default-constructed `Config{}` produces sensible behaviour for
     * SPY-style equity options. All fields have well-documented
     * defaults; see individual member docs.
     */
    struct Config {
        /**
         * Absolute price-residual tolerance: converged iff
         * `|BS(sigma) - market_price| < tolerance`. This is a *price*
         * tolerance (in the option's currency), not a `sigma` tolerance.
         * Must be `> 0`.
         */
        double tolerance{1e-10};

        /**
         * Hard iteration cap for Newton-Raphson. Bisection has its own
         * cap (see `max_bisection_iterations`). Must be `> 0`.
         */
        std::size_t max_iterations{100};

        /**
         * When Newton fails (non-convergence, degenerate derivative, or
         * an out-of-range converged value), re-run the same problem
         * with `BisectionSolver` over `[min_volatility, max_volatility]`.
         * Bisection is globally convergent on a valid bracket so this
         * dramatically improves robustness at ~30 iterations of cost.
         */
        bool use_bisection_fallback{true};

        /**
         * Initial `sigma` for Newton. `0` means "use the Brenner-
         * Subrahmanyam heuristic" (see the .cpp for the formula).
         * Positive values override the heuristic — useful when the
         * caller has a good prior (e.g. yesterday's IV).
         */
        double initial_guess{0.0};

        /**
         * Lower end of the bisection bracket. Must be `> 0`. `1e-6`
         * corresponds to essentially zero vol — `BS(1e-6)` equals the
         * discounted intrinsic (in forward space) to full precision.
         */
        double min_volatility{1e-6};

        /**
         * Upper end of the bisection bracket. 500% annualized vol is
         * well above any liquid-equity regime; crypto or single-name
         * blowups can approach it. Adjust for regimes with higher
         * expected vol.
         */
        double max_volatility{5.0};

        /**
         * Bisection iteration cap. `log2((max - min) / tolerance)` gives
         * the required count; ~30 is enough for any realistic setting.
         */
        std::size_t max_bisection_iterations{100};
    };

    ImpliedVolatilitySolver() = default;

    /** Construct with a specific configuration. */
    explicit ImpliedVolatilitySolver(Config config) : config_(config) {}

    /** Access the immutable configuration. */
    [[nodiscard]] const Config& config() const noexcept { return config_; }

    /**
     * @brief Solve for the volatility that reproduces `market_price` under
     *        `inputs`.
     *
     * @param inputs        Black-Scholes inputs *excluding* `volatility`
     *                      (which is what we're solving for). If a value
     *                      is supplied it is ignored.
     * @param market_price  Observed option price. Must satisfy
     *                      `lower_bound <= market_price <= upper_bound`
     *                      (see arbitrage bounds in the class docstring).
     *
     * @return Populated `ImpliedVolatilityResult`. `.root` is the implied
     *         volatility on success. `.iterations` reflects the *winning*
     *         method (Newton if Newton converged; bisection if
     *         bisection took over). `.method` records which method
     *         produced the value. `.status` follows the numerics module
     *         convention.
     *
     * @throws std::invalid_argument if `market_price` is non-finite,
     *         negative, or above the arbitrage upper bound; if inputs
     *         violate `BlackScholesEngine`'s preconditions
     *         (`spot <= 0`, `strike <= 0`, `T < 0`, non-finite rates);
     *         if `config().tolerance <= 0`, `config().max_iterations == 0`,
     *         or bracket bounds are non-positive with bisection enabled.
     */
    [[nodiscard]] ImpliedVolatilityResult solve(
        const BlackScholesEngine::Inputs& inputs,
        double market_price) const;

    /**
     * @brief Convenience overload that takes an `Option` + `MarketSnapshot`.
     *
     * Derives `time_to_expiry` from `market.valuation_date` and
     * `option.expiration` (ACT/365F). `market.volatility` is ignored —
     * we're finding it. Throws `std::invalid_argument` for American
     * exercise.
     */
    [[nodiscard]] ImpliedVolatilityResult solve(
        const ore::core::Option& option,
        const ore::core::MarketSnapshot& market,
        double market_price) const;

private:
    Config config_{};
};

} // namespace ore::pricing
