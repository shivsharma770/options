#include <ore/pricing/black_scholes_engine.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <ore/numerics/normal_distribution.hpp>

namespace ore::pricing {

namespace {

using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionType;
using ore::numerics::NormalDistribution;

/**
 * Intermediate quantities computed exactly once per pricing call and
 * reused by every Greek. See the .hpp for the formulas.
 */
struct Internals {
    double sqrt_T;   // sqrt(T)
    double d1;       // (ln(S/K) + (r - q + sigma^2 / 2) * T) / (sigma * sqrt(T))
    double d2;       // d1 - sigma * sqrt(T)
    double n_d1;     // N(d1)
    double n_d2;     // N(d2)
    double phi_d1;   // phi(d1) — density, same for calls and puts
    double df_r;     // e^{-r * T}, discount factor on the risk-free rate
    double df_q;     // e^{-q * T}, discount factor on the dividend yield
};

/**
 * Validate an `Inputs` struct. Preconditions violations produce
 * `std::invalid_argument`; boundary cases (`T == 0`, `sigma == 0`) are
 * *valid* and dispatched to the deterministic-limit code below.
 */
void validate(const BlackScholesEngine::Inputs& in) {
    if (!std::isfinite(in.spot) || in.spot <= 0.0) {
        throw std::invalid_argument(
            "BlackScholesEngine: spot must be finite and > 0");
    }
    if (!std::isfinite(in.strike) || in.strike <= 0.0) {
        throw std::invalid_argument(
            "BlackScholesEngine: strike must be finite and > 0");
    }
    if (!std::isfinite(in.rate)) {
        throw std::invalid_argument(
            "BlackScholesEngine: rate must be finite");
    }
    if (!std::isfinite(in.dividend_yield)) {
        throw std::invalid_argument(
            "BlackScholesEngine: dividend_yield must be finite");
    }
    if (!std::isfinite(in.volatility) || in.volatility < 0.0) {
        throw std::invalid_argument(
            "BlackScholesEngine: volatility must be finite and >= 0");
    }
    if (!std::isfinite(in.time_to_expiry) || in.time_to_expiry < 0.0) {
        throw std::invalid_argument(
            "BlackScholesEngine: time_to_expiry must be finite and >= 0");
    }
}

/**
 * Compute all reusable intermediates once. Called only when both
 * `sigma > 0` and `T > 0`, so `sigma * sqrt(T) > 0` and the formula is
 * well-defined.
 */
Internals compute_internals(const BlackScholesEngine::Inputs& in) {
    Internals I{};
    I.sqrt_T = std::sqrt(in.time_to_expiry);

    const double sigma_sqrt_T = in.volatility * I.sqrt_T;
    const double drift_term   = (in.rate - in.dividend_yield
                                 + 0.5 * in.volatility * in.volatility)
                                * in.time_to_expiry;

    I.d1     = (std::log(in.spot / in.strike) + drift_term) / sigma_sqrt_T;
    I.d2     = I.d1 - sigma_sqrt_T;
    I.n_d1   = NormalDistribution::cdf(I.d1);
    I.n_d2   = NormalDistribution::cdf(I.d2);
    I.phi_d1 = NormalDistribution::pdf(I.d1);
    I.df_r   = std::exp(-in.rate * in.time_to_expiry);
    I.df_q   = std::exp(-in.dividend_yield * in.time_to_expiry);
    return I;
}

/**
 * Deterministic-limit result — used when volatility or time-to-expiry is
 * exactly zero. The option's payoff becomes a known function of the
 * (already-known) forward price, so its value collapses to a discounted
 * intrinsic and all Greeks except delta are zero.
 *
 * At T = 0 the forward equals spot and discount factors are 1, so both
 * boundary cases fall out of the same formula.
 */
PricingResult deterministic_result(const BlackScholesEngine::Inputs& in) {
    const double df_r    = std::exp(-in.rate * in.time_to_expiry);
    const double df_q    = std::exp(-in.dividend_yield * in.time_to_expiry);
    const double forward = in.spot * std::exp((in.rate - in.dividend_yield)
                                              * in.time_to_expiry);

    PricingResult r;
    r.engine_name = "BlackScholes";

    if (in.type == OptionType::Call) {
        r.price          = df_r * std::max(forward - in.strike, 0.0);
        r.greeks.delta   = (forward > in.strike) ? df_q : 0.0;
    } else {
        r.price          = df_r * std::max(in.strike - forward, 0.0);
        r.greeks.delta   = (forward < in.strike) ? -df_q : 0.0;
    }
    // gamma, vega, theta, rho are all identically 0 in the deterministic
    // limit — see docs/BLACK_SCHOLES_VALIDATION.md for the derivation.
    return r;
}

/**
 * The regular (non-degenerate) case: `sigma > 0` and `T > 0`.
 */
PricingResult general_result(const BlackScholesEngine::Inputs& in,
                             const Internals& I) {
    // These two show up in every put formula; compute once.
    const double n_minus_d1 = 1.0 - I.n_d1;
    const double n_minus_d2 = 1.0 - I.n_d2;

    // The "gamma * volatility" piece of theta is identical for calls and
    // puts. See Hull 10th Ed. eq. 19.4.
    const double time_decay_term =
        -in.spot * I.df_q * I.phi_d1 * in.volatility / (2.0 * I.sqrt_T);

    PricingResult r;
    r.engine_name = "BlackScholes";

    // Gamma and vega are identical between calls and puts.
    r.greeks.gamma =
        I.df_q * I.phi_d1 / (in.spot * in.volatility * I.sqrt_T);
    r.greeks.vega  = in.spot * I.df_q * I.phi_d1 * I.sqrt_T;

    if (in.type == OptionType::Call) {
        r.price        =  in.spot   * I.df_q * I.n_d1
                        - in.strike * I.df_r * I.n_d2;
        r.greeks.delta =  I.df_q * I.n_d1;
        r.greeks.theta =  time_decay_term
                        - in.rate           * in.strike * I.df_r * I.n_d2
                        + in.dividend_yield * in.spot   * I.df_q * I.n_d1;
        r.greeks.rho   =  in.strike * in.time_to_expiry * I.df_r * I.n_d2;
    } else {
        r.price        =  in.strike * I.df_r * n_minus_d2
                        - in.spot   * I.df_q * n_minus_d1;
        r.greeks.delta = -I.df_q * n_minus_d1;
        r.greeks.theta =  time_decay_term
                        + in.rate           * in.strike * I.df_r * n_minus_d2
                        - in.dividend_yield * in.spot   * I.df_q * n_minus_d1;
        r.greeks.rho   = -in.strike * in.time_to_expiry * I.df_r * n_minus_d2;
    }

    return r;
}

} // unnamed namespace

PricingResult BlackScholesEngine::price(const Inputs& inputs) const {
    validate(inputs);
    if (inputs.time_to_expiry == 0.0 || inputs.volatility == 0.0) {
        return deterministic_result(inputs);
    }
    return general_result(inputs, compute_internals(inputs));
}

PricingResult BlackScholesEngine::price(
    const Option& option,
    const MarketSnapshot& market) const
{
    if (option.exercise != ore::core::ExerciseStyle::European) {
        throw std::invalid_argument(
            "BlackScholesEngine only supports European exercise; received an "
            "American option.");
    }

    // ACT/365F — the same convention MarketSnapshot documents for the
    // rate and yield. If the caller's data uses a different day count
    // they should compute T themselves and use the Inputs overload.
    const auto val_days = std::chrono::sys_days{market.valuation_date};
    const auto exp_days = std::chrono::sys_days{option.expiration};
    const auto days     = (exp_days - val_days).count();
    const double T      = static_cast<double>(days) / 365.0;

    return price(Inputs{
        .spot           = market.spot,
        .strike         = option.strike,
        .rate           = market.risk_free_rate,
        .dividend_yield = market.dividend_yield,
        .volatility     = market.volatility,
        .time_to_expiry = T,
        .type           = option.type,
    });
}

} // namespace ore::pricing
