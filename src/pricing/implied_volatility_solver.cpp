#include <ore/pricing/implied_volatility_solver.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

#include <ore/numerics/bisection.hpp>
#include <ore/numerics/newton_raphson.hpp>

namespace ore::pricing {

namespace {

using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionType;
using ore::numerics::BisectionSolver;
using ore::numerics::NewtonRaphsonSolver;
using ore::numerics::SolverResult;
using ore::numerics::SolverStatus;

/**
 * Discounted intrinsic value in forward space — the theoretical lower
 * arbitrage bound on a European option's price. At `sigma == 0` the
 * Black-Scholes closed form collapses to exactly this quantity, so
 * `market_price < lower_bound` is impossible under no-arbitrage
 * assumptions.
 *
 * Call:  max(0, S*e^{-qT} - K*e^{-rT})
 * Put:   max(0, K*e^{-rT} - S*e^{-qT})
 */
double compute_lower_bound(const BlackScholesEngine::Inputs& in) noexcept {
    const double disc_S = in.spot   * std::exp(-in.dividend_yield * in.time_to_expiry);
    const double disc_K = in.strike * std::exp(-in.rate           * in.time_to_expiry);
    if (in.type == OptionType::Call) {
        return std::max(0.0, disc_S - disc_K);
    }
    return std::max(0.0, disc_K - disc_S);
}

/**
 * Theoretical upper arbitrage bound on a European option's price. As
 * `sigma -> infinity`:
 *   - the call approaches `S*e^{-qT}` (asymptote from below), and
 *   - the put approaches `K*e^{-rT}` (also from below).
 * A market price exceeding these bounds is a hard arbitrage violation.
 */
double compute_upper_bound(const BlackScholesEngine::Inputs& in) noexcept {
    if (in.type == OptionType::Call) {
        return in.spot   * std::exp(-in.dividend_yield * in.time_to_expiry);
    }
    return in.strike * std::exp(-in.rate * in.time_to_expiry);
}

/**
 * Initial `sigma` guess for Newton-Raphson.
 *
 * Uses the Brenner-Subrahmanyam (1988) at-the-money-forward approximation
 *   sigma_0 ~= sqrt(2*pi / T) * C / S
 * which is *exact* to first order for an ATM-forward call and remains
 * good (within ~30% of the truth) for a wide range of moneyness. See
 * Brenner, M. and Subrahmanyam, M. (1988), "A Simple Formula to Compute
 * the Implied Standard Deviation", *Financial Analysts Journal* 44(5),
 * 80-83.
 *
 * For puts we convert the market price to an equivalent call price via
 * put-call parity
 *   C_equiv = P + S*e^{-qT} - K*e^{-rT}
 * before applying the formula, keeping the heuristic unified for both
 * option types.
 *
 * The result is clamped to `[0.05, 3.0]`:
 *   - The lower clamp prevents starting Newton from a value where Vega
 *     is numerically tiny (deep-OTM regime).
 *   - The upper clamp prevents starting outside any realistic vol range.
 * Non-finite results (from `T <= 0` or a negative `C_equiv`) fall back
 * to the default of 20% — the historical median annualized vol for the
 * broad US equity market. Newton's quadratic convergence means that even
 * this fallback typically reaches 1e-10 residual within 6-8 iterations.
 */
double compute_initial_guess(
    const BlackScholesEngine::Inputs& in,
    double market_price,
    const ImpliedVolatilitySolver::Config& config) noexcept
{
    if (config.initial_guess > 0.0) {
        return config.initial_guess;
    }

    constexpr double kDefault = 0.20;
    constexpr double kLo      = 0.05;
    constexpr double kHi      = 3.0;

    if (!(in.time_to_expiry > 0.0)) return kDefault;

    double call_equiv = market_price;
    if (in.type == OptionType::Put) {
        const double disc_S = in.spot   * std::exp(-in.dividend_yield * in.time_to_expiry);
        const double disc_K = in.strike * std::exp(-in.rate           * in.time_to_expiry);
        call_equiv = market_price + disc_S - disc_K;
    }
    if (!(call_equiv > 0.0)) return kDefault;

    const double sigma_0 = std::sqrt(2.0 * std::numbers::pi_v<double> / in.time_to_expiry)
                         * call_equiv / in.spot;

    if (!std::isfinite(sigma_0)) return kDefault;
    return std::clamp(sigma_0, kLo, kHi);
}

/**
 * Validate the *inputs* struct's fields ahead of dispatching to the
 * engine. `BlackScholesEngine::price(Inputs)` performs its own
 * validation but it does not check `time_to_expiry > 0`: the deterministic
 * `T == 0` branch would produce the intrinsic and any implied-vol solve
 * against a zero-time price is degenerate (and useless).
 */
void validate_inputs(const BlackScholesEngine::Inputs& in) {
    if (!(in.time_to_expiry > 0.0)) {
        throw std::invalid_argument(
            "ImpliedVolatilitySolver: time_to_expiry must be > 0 "
            "(zero-time options have no meaningful implied volatility)");
    }
}

void validate_config(const ImpliedVolatilitySolver::Config& c) {
    if (!(c.tolerance > 0.0)) {
        throw std::invalid_argument(
            "ImpliedVolatilitySolver: tolerance must be > 0");
    }
    if (c.max_iterations == 0) {
        throw std::invalid_argument(
            "ImpliedVolatilitySolver: max_iterations must be > 0");
    }
    if (c.use_bisection_fallback) {
        if (!(c.min_volatility > 0.0) || !(c.max_volatility > c.min_volatility)) {
            throw std::invalid_argument(
                "ImpliedVolatilitySolver: require "
                "0 < min_volatility < max_volatility for bisection");
        }
        if (c.max_bisection_iterations == 0) {
            throw std::invalid_argument(
                "ImpliedVolatilitySolver: max_bisection_iterations must be > 0");
        }
    }
}

/**
 * Was Newton's `SolverResult` a usable answer? Newton is considered to
 * have succeeded iff it converged to a positive, finite `sigma`. Because
 * Black-Scholes is strictly monotone in `sigma`, any converged root is
 * *the* root — we do not need a range check against the bisection
 * bracket, and imposing one would spuriously reject valid high-vol
 * solutions (e.g. crypto or single-name blowup regimes).
 */
bool newton_is_acceptable(const SolverResult& r) noexcept {
    if (r.status != SolverStatus::Converged)     return false;
    if (!std::isfinite(r.root) || r.root <= 0.0) return false;
    return true;
}

/**
 * Lift a low-level `SolverResult` into the pricing-side result type,
 * tagging it with the numerical method that produced it. Field-by-field
 * copy — the two structs share every name.
 */
ImpliedVolatilityResult tag(const SolverResult& r, ImpliedVolatilityMethod method) noexcept {
    return ImpliedVolatilityResult{
        .root       = r.root,
        .iterations = r.iterations,
        .status     = r.status,
        .residual   = r.residual,
        .method     = method,
    };
}

}  // namespace

ImpliedVolatilityResult ImpliedVolatilitySolver::solve(
    const BlackScholesEngine::Inputs& inputs,
    double market_price) const
{
    validate_config(config_);
    validate_inputs(inputs);

    if (!std::isfinite(market_price) || market_price < 0.0) {
        throw std::invalid_argument(
            "ImpliedVolatilitySolver: market_price must be finite and >= 0");
    }

    const double lower_bound = compute_lower_bound(inputs);
    const double upper_bound = compute_upper_bound(inputs);

    if (market_price > upper_bound) {
        throw std::invalid_argument(
            "ImpliedVolatilitySolver: market_price exceeds theoretical "
            "arbitrage upper bound (S*e^{-qT} for calls, K*e^{-rT} for puts)");
    }
    // At-or-below the lower bound is the sigma == 0 limit of the model.
    // Return it as a converged trivial solution rather than throwing —
    // this is a valid (if boring) implied vol, and callers hitting it
    // often want to know how far below the bound they are (in the
    // residual field). `Newton` is a slight misuse of `method` here (we
    // did not actually iterate), but the caller convention is "if
    // convergence happened without invoking bisection, method == Newton".
    if (market_price <= lower_bound) {
        return ImpliedVolatilityResult{
            .root       = 0.0,
            .iterations = 0,
            .status     = SolverStatus::Converged,
            .residual   = lower_bound - market_price,
            .method     = ImpliedVolatilityMethod::Newton,
        };
    }

    // ---- Set up the objective and its derivative ---------------------------
    //
    // The Newton objective is
    //     f(sigma) = BlackScholes(sigma) - market_price
    // and Newton uses the *analytical* Vega (df/dsigma) returned by the
    // engine as `PricingResult::greeks.vega`. Numerical finite differences
    // would introduce O(h^2) truncation error and slow the algorithm below
    // its quadratic-convergence regime — see the header for full rationale.
    //
    // Efficiency invariant: each Newton iteration must trigger *exactly one*
    // BlackScholes evaluation, even though it consults both `f` and `fprime`
    // for the same `sigma`. We achieve this with a tiny closure-owned cache
    // keyed on the last-evaluated `sigma`; the second callable (whichever
    // Newton hits second) is a cache hit and does no engine work.
    // ------------------------------------------------------------------------
    BlackScholesEngine engine;
    auto pricing_inputs = inputs;

    double cached_sigma = std::numeric_limits<double>::quiet_NaN();
    double cached_price = 0.0;
    double cached_vega  = 0.0;

    auto refresh = [&](double sigma) {
        // NaN != NaN, so the first call always misses.
        if (sigma == cached_sigma) return;
        pricing_inputs.volatility = sigma;
        const auto result = engine.price(pricing_inputs);
        cached_sigma = sigma;
        cached_price = result.price;
        cached_vega  = result.greeks.vega;
    };

    // Newton is allowed to try any sigma the arithmetic produces — including
    // negative ones after an aggressive step. Returning NaN there causes
    // `NewtonRaphsonSolver` to bail with `NonFiniteEvaluation`, which we
    // then treat as a fallback trigger. This is preferable to clamping,
    // which would silently distort the derivative in Newton's eyes.
    auto f = [&](double sigma) -> double {
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        refresh(sigma);
        return cached_price - market_price;
    };
    auto fprime = [&](double sigma) -> double {
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        refresh(sigma);
        return cached_vega;
    };

    // ---- Attempt Newton-Raphson --------------------------------------------
    const NewtonRaphsonSolver newton(NewtonRaphsonSolver::Config{
        .tolerance      = config_.tolerance,
        .max_iterations = config_.max_iterations,
        // Vega collapses toward 0 for deep-ITM/OTM options; if it drops
        // below `min_derivative` Newton would divide by a near-zero and
        // ricochet arbitrarily. `1e-8` is well below any realistic-regime
        // Vega but well above `min_derivative`'s default of `1e-14`, so
        // Newton hands the problem to bisection *before* producing a
        // catastrophic step.
        .min_derivative = 1e-8,
    });

    const double sigma_0 = compute_initial_guess(inputs, market_price, config_);
    const auto newton_result = newton.solve(f, fprime, sigma_0);

    if (newton_is_acceptable(newton_result)) {
        return tag(newton_result, ImpliedVolatilityMethod::Newton);
    }

    // ---- Fallback to bisection ---------------------------------------------
    //
    // We fall back on: MaxIterationsReached, DerivativeTooSmall,
    // NonFiniteEvaluation, or a "Converged" result that landed outside the
    // valid sigma range. All of these signal that Newton is unreliable on
    // this problem; bisection is unconditionally convergent on a bracket
    // that contains a sign change, and BS is monotone in sigma, so the
    // bracket [min_volatility, max_volatility] is guaranteed valid provided
    // `lower_bound < market_price < upper_bound` — which we validated above.
    //
    // Bracket choice (default 1e-6 to 5.0):
    //   * 1e-6 corresponds to essentially zero vol; BS(1e-6) = lower_bound
    //     to full precision, so f(1e-6) has the same sign as
    //     (lower_bound - market_price) < 0.
    //   * 5.0 is 500% annualized vol — well above any liquid regime.
    //     BS(5.0) is close to upper_bound, and for market_price strictly
    //     less than upper_bound f(5.0) > 0.
    // Sign change is therefore guaranteed and bisection needs
    // log2(5e6 / tolerance) ~ 22-40 iterations to converge.
    if (!config_.use_bisection_fallback) {
        return tag(newton_result, ImpliedVolatilityMethod::Newton);
    }

    // Reset the cache — Newton may have left it in a state that no longer
    // reflects a useful last evaluation. The first bisection call will
    // repopulate it.
    cached_sigma = std::numeric_limits<double>::quiet_NaN();

    const BisectionSolver bisection(BisectionSolver::Config{
        .tolerance      = config_.tolerance,
        .max_iterations = config_.max_bisection_iterations,
    });

    return tag(bisection.solve(f, config_.min_volatility, config_.max_volatility),
               ImpliedVolatilityMethod::Bisection);
}

ImpliedVolatilityResult ImpliedVolatilitySolver::solve(
    const Option& option,
    const MarketSnapshot& market,
    double market_price) const
{
    if (option.exercise != ore::core::ExerciseStyle::European) {
        throw std::invalid_argument(
            "ImpliedVolatilitySolver only supports European exercise; "
            "received an American option.");
    }

    // ACT/365F — matches `BlackScholesEngine::price(Option, MarketSnapshot)`
    // and the day-count convention documented on `MarketSnapshot`. Callers
    // wanting a different day count should compute `T` themselves and use
    // the `Inputs` overload.
    const auto val_days = std::chrono::sys_days{market.valuation_date};
    const auto exp_days = std::chrono::sys_days{option.expiration};
    const auto days     = (exp_days - val_days).count();
    const double T      = static_cast<double>(days) / 365.0;

    return solve(BlackScholesEngine::Inputs{
        .spot           = market.spot,
        .strike         = option.strike,
        .rate           = market.risk_free_rate,
        .dividend_yield = market.dividend_yield,
        // Deliberately zeroed — we're *solving* for volatility, so any
        // value stored on the snapshot is ignored. Overwritten on every
        // Newton/bisection iteration inside the primary overload.
        .volatility     = 0.0,
        .time_to_expiry = T,
        .type           = option.type,
    }, market_price);
}

}  // namespace ore::pricing
