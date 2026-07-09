#include <ore/pricing/binomial_tree_engine.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ore::pricing {

namespace {

using ore::core::ExerciseStyle;
using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionType;

// ---------------------------------------------------------------------------
// Input validation.
//
// Boundary cases (T == 0, sigma == 0) are *valid* and dispatched to the
// deterministic-limit code path — they are not preconditions.
// ---------------------------------------------------------------------------

void validate(const BinomialTreeEngine::Inputs& in) {
    if (!std::isfinite(in.spot) || in.spot <= 0.0) {
        throw std::invalid_argument(
            "BinomialTreeEngine: spot must be finite and > 0");
    }
    if (!std::isfinite(in.strike) || in.strike <= 0.0) {
        throw std::invalid_argument(
            "BinomialTreeEngine: strike must be finite and > 0");
    }
    if (!std::isfinite(in.rate)) {
        throw std::invalid_argument(
            "BinomialTreeEngine: rate must be finite");
    }
    if (!std::isfinite(in.dividend_yield)) {
        throw std::invalid_argument(
            "BinomialTreeEngine: dividend_yield must be finite");
    }
    if (!std::isfinite(in.volatility) || in.volatility < 0.0) {
        throw std::invalid_argument(
            "BinomialTreeEngine: volatility must be finite and >= 0");
    }
    if (!std::isfinite(in.time_to_expiry) || in.time_to_expiry < 0.0) {
        throw std::invalid_argument(
            "BinomialTreeEngine: time_to_expiry must be finite and >= 0");
    }
}

// ---------------------------------------------------------------------------
// Vanilla payoff. Shared between calls and puts to avoid two nearly-
// identical induction loops.
// ---------------------------------------------------------------------------

constexpr double payoff(double spot, double strike, OptionType type) noexcept {
    return type == OptionType::Call
        ? std::max(spot - strike, 0.0)
        : std::max(strike - spot, 0.0);
}

// ---------------------------------------------------------------------------
// Deterministic-limit: sigma == 0 or T == 0. Same reasoning as
// `BlackScholesEngine::deterministic_result` — the forward is known, the
// option's value collapses to a discounted intrinsic, and only delta is
// non-zero (± the dividend-yield discount factor).
// ---------------------------------------------------------------------------

double deterministic_price(const BinomialTreeEngine::Inputs& in) noexcept {
    const double df_r    = std::exp(-in.rate * in.time_to_expiry);
    const double forward = in.spot * std::exp((in.rate - in.dividend_yield)
                                               * in.time_to_expiry);
    const double intrinsic_at_expiry = payoff(forward, in.strike, in.type);

    // For American: an option that never gets stochastic still gets to
    // exercise now. Intrinsic at t=0 is payoff(spot, strike).
    if (in.exercise == ExerciseStyle::American) {
        const double intrinsic_now = payoff(in.spot, in.strike, in.type);
        return std::max(df_r * intrinsic_at_expiry, intrinsic_now);
    }
    return df_r * intrinsic_at_expiry;
}

// ---------------------------------------------------------------------------
// CRR core: one full backward induction. Called for the primary price
// and once per Greek bump.
//
// Memory: one `std::vector<double>` of size N+1 that gets overwritten
//         in place ("rolling vector"). No full lattice is stored.
// Time:   the outer loop runs N times, the k-th iteration has O(k)
//         inner work, so total is O(N(N+1)/2) = O(N²).
//
// We precompute `u_pow[0..N]` and `d_pow[0..N]` once so the intermediate
// spot at any lattice node reduces to two multiplications:
//     S_{i,k} = S_0 · u^i · d^{k-i} = spot · u_pow[i] · d_pow[k-i]
// This is O(N) extra memory but removes N² `std::pow` calls, which
// dominate runtime otherwise.
// ---------------------------------------------------------------------------

double crr_price(const BinomialTreeEngine::Inputs& in, std::size_t steps) {
    // Steps == 0 makes the model ill-defined (dt = T/0). Enforce N >= 1.
    if (steps == 0) {
        throw std::invalid_argument(
            "BinomialTreeEngine: steps must be >= 1");
    }
    // Boundary case: sigma == 0 or T == 0 — no tree required.
    if (in.volatility == 0.0 || in.time_to_expiry == 0.0) {
        return deterministic_price(in);
    }

    const double dt   = in.time_to_expiry / static_cast<double>(steps);
    const double u    = std::exp(in.volatility * std::sqrt(dt));
    const double d    = 1.0 / u;
    // Continuously compounded drift over one step.
    const double growth = std::exp((in.rate - in.dividend_yield) * dt);
    const double p    = (growth - d) / (u - d);
    const double q    = 1.0 - p;
    const double df   = std::exp(-in.rate * dt);

    // With finite N and large sigma·sqrt(dt), CRR can produce p outside
    // [0, 1]. This does not violate no-arbitrage of the discrete model
    // per se — the tree is still a well-defined finite calculation —
    // but it does mean the underlying continuous-time interpretation is
    // strained. We warn (in the diagnostic sense) by returning NaN
    // rather than a nonsense number. Callers should either shrink dt
    // (more steps) or use a different tree parameterisation.
    if (p < 0.0 || p > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Precompute u_pow[k] = u^k and d_pow[k] = d^k for k in [0, N].
    // These give O(1) spot lookup at any node without repeated std::pow.
    std::vector<double> u_pow(steps + 1);
    std::vector<double> d_pow(steps + 1);
    u_pow[0] = d_pow[0] = 1.0;
    for (std::size_t k = 1; k <= steps; ++k) {
        u_pow[k] = u_pow[k - 1] * u;
        d_pow[k] = d_pow[k - 1] * d;
    }

    // Terminal payoffs. values[i] corresponds to node (i up-moves, N-i
    // down-moves), so S_{i,N} = spot · u^i · d^{N-i}.
    std::vector<double> values(steps + 1);
    for (std::size_t i = 0; i <= steps; ++i) {
        const double S_iN = in.spot * u_pow[i] * d_pow[steps - i];
        values[i] = payoff(S_iN, in.strike, in.type);
    }

    // Backward induction. At the start of iteration k, `values[0..k]`
    // holds discounted risk-neutral expectations at time-step k; after
    // the loop k is decremented to k-1 and values[0..k-1] are the new
    // node values at step k-1.
    const bool is_american = (in.exercise == ExerciseStyle::American);
    for (std::size_t k = steps; k > 0; --k) {
        const std::size_t next = k - 1;   // time step we're solving for
        for (std::size_t i = 0; i <= next; ++i) {
            // Risk-neutral one-step expectation, discounted.
            const double continuation = df * (p * values[i + 1] + q * values[i]);
            if (is_american) {
                const double S_here = in.spot * u_pow[i] * d_pow[next - i];
                const double intrinsic = payoff(S_here, in.strike, in.type);
                values[i] = std::max(continuation, intrinsic);
            } else {
                values[i] = continuation;
            }
        }
    }
    return values[0];
}

// ---------------------------------------------------------------------------
// Numerical Greeks — bump-and-revalue.
//
// Central differences are used wherever both sides of the bump are
// physically valid. Theta uses a forward difference toward expiration
// so it does not fail near T = 0.
//
// Bump-size philosophy:
//   * Truncation error of central diff is O(h²), of forward is O(h).
//   * Tree noise (the "sawtooth" oscillation around Black-Scholes) is
//     O(1/N) in the price. Bumps must be *large enough* that the true
//     derivative signal exceeds this noise floor.
//   * The chosen values are the trader-desk conventions:
//       Δ, Γ  : 1 % of spot
//       ν     : 10 bp of vol   (0.001 in decimal)
//       ρ     : 1 bp of rate   (0.0001 in decimal)
//       Θ     : 1 calendar day (1/365 year)
//   These are unrelated to the *unit* the Greek is reported in (see
//   `greeks.hpp`); they only affect the numerical accuracy of the
//   estimate, not its scaling.
// ---------------------------------------------------------------------------

constexpr double kDeltaBumpRelative = 0.01;    // 1 % of spot
constexpr double kVegaBump          = 1.0e-3;  // 10 bp of vol
constexpr double kRhoBump           = 1.0e-4;  // 1 bp of rate
constexpr double kThetaBump         = 1.0 / 365.0;

Greeks compute_greeks(
    const BinomialTreeEngine::Inputs& in,
    std::size_t steps,
    double center_price)
{
    // A local helper: repeat the CRR price with one field perturbed.
    // Copying the struct is cheap (7 doubles + 2 enums).
    const auto revalue = [&](auto&& mutator) {
        auto perturbed = in;
        mutator(perturbed);
        return crr_price(perturbed, steps);
    };

    // ---- Spot bumps (shared by Delta and Gamma) ----
    const double h_S    = kDeltaBumpRelative * in.spot;
    const double V_S_up = revalue([&](auto& p) { p.spot = in.spot + h_S; });
    const double V_S_dn = revalue([&](auto& p) { p.spot = in.spot - h_S; });

    Greeks g{};
    g.delta = (V_S_up - V_S_dn) / (2.0 * h_S);
    g.gamma = (V_S_up - 2.0 * center_price + V_S_dn) / (h_S * h_S);

    // ---- Vega (central diff in volatility) ----
    // If sigma - h would go negative we clamp at 0, becoming an
    // asymmetric bump. The denominator uses the *actual* gap
    // `sigma_up - sigma_dn`, not `2h`, so the estimator remains
    // unbiased for the derivative at the midpoint of the [sigma_dn,
    // sigma_up] interval.
    const double h_sigma  = kVegaBump;
    const double sigma_up = in.volatility + h_sigma;
    const double sigma_dn = std::max(in.volatility - h_sigma, 0.0);
    const double denom_sigma = sigma_up - sigma_dn;
    if (denom_sigma > 0.0) {
        const double V_v_up = revalue([&](auto& p) { p.volatility = sigma_up; });
        const double V_v_dn = revalue([&](auto& p) { p.volatility = sigma_dn; });
        g.vega = (V_v_up - V_v_dn) / denom_sigma;
    }
    // else h_sigma == 0 (unreachable given constexpr bump); leave vega at 0.

    // ---- Rho (central diff in rate) ----
    const double h_r   = kRhoBump;
    const double V_r_up = revalue([&](auto& p) { p.rate = in.rate + h_r; });
    const double V_r_dn = revalue([&](auto& p) { p.rate = in.rate - h_r; });
    g.rho = (V_r_up - V_r_dn) / (2.0 * h_r);

    // ---- Theta (forward diff toward expiration) ----
    // Θ = dV/dt where t is calendar time = -dV/dT.
    // Estimate dV/dT ≈ (V(T) - V(T - h)) / h, then negate. This is
    // valid at every T > 0; we clamp T - h at 0 for the T < h case.
    const double h_T   = kThetaBump;
    const double T_dn  = std::max(in.time_to_expiry - h_T, 0.0);
    const double actual_h_T = in.time_to_expiry - T_dn;  // 0 if T == 0
    if (actual_h_T > 0.0) {
        const double V_T_dn = revalue([&](auto& p) { p.time_to_expiry = T_dn; });
        g.theta = -(center_price - V_T_dn) / actual_h_T;
    }
    // else T == 0 exactly: theta is undefined; leave at 0 for the same
    // convention as the BS deterministic limit.

    return g;
}

// ---------------------------------------------------------------------------
// Engine-name builder — updates the returned string_view when Config
// changes.
// ---------------------------------------------------------------------------

std::string make_engine_name(std::size_t steps) {
    // 32 characters is enough for step counts up to a trillion; larger
    // would be exotic and probably a bug.
    char buf[48];
    const int n = std::snprintf(buf, sizeof(buf), "Binomial(CRR, N=%zu)", steps);
    return std::string(buf, (n > 0) ? static_cast<std::size_t>(n) : 0);
}

} // unnamed namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

BinomialTreeEngine::BinomialTreeEngine()
    : config_{}, name_(make_engine_name(config_.steps)) {}

BinomialTreeEngine::BinomialTreeEngine(Config config)
    : config_(config), name_(make_engine_name(config.steps)) {}

PricingResult BinomialTreeEngine::price(const Inputs& inputs) const {
    validate(inputs);
    if (config_.steps == 0) {
        throw std::invalid_argument(
            "BinomialTreeEngine: Config::steps must be >= 1");
    }

    PricingResult r;
    r.engine_name    = name_;
    r.iterations     = config_.steps;
    r.standard_error = std::nullopt;

    r.price = crr_price(inputs, config_.steps);

    if (config_.compute_greeks) {
        r.greeks = compute_greeks(inputs, config_.steps, r.price);
    }
    return r;
}

PricingResult BinomialTreeEngine::price(
    const Option& option,
    const MarketSnapshot& market) const
{
    // ACT/365F — same convention as MarketSnapshot documents.
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
        .exercise       = option.exercise,
    });
}

} // namespace ore::pricing
