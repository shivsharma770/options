#include <ore/pricing/monte_carlo_engine.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <ore/numerics/random_number_generator.hpp>

namespace ore::pricing {

namespace {

using ore::core::ExerciseStyle;
using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionType;
using ore::numerics::MersenneTwisterNormalGenerator;
using ore::numerics::NormalGenerator;

// -----------------------------------------------------------------------------
// Input validation. Same shape as BlackScholesEngine and
// BinomialTreeEngine so error messages read consistently.
// -----------------------------------------------------------------------------

void validate(const MonteCarloEngine::Inputs& in) {
    if (!std::isfinite(in.spot) || in.spot <= 0.0) {
        throw std::invalid_argument(
            "MonteCarloEngine: spot must be finite and > 0");
    }
    if (!std::isfinite(in.strike) || in.strike <= 0.0) {
        throw std::invalid_argument(
            "MonteCarloEngine: strike must be finite and > 0");
    }
    if (!std::isfinite(in.rate)) {
        throw std::invalid_argument(
            "MonteCarloEngine: rate must be finite");
    }
    if (!std::isfinite(in.dividend_yield)) {
        throw std::invalid_argument(
            "MonteCarloEngine: dividend_yield must be finite");
    }
    if (!std::isfinite(in.volatility) || in.volatility < 0.0) {
        throw std::invalid_argument(
            "MonteCarloEngine: volatility must be finite and >= 0");
    }
    if (!std::isfinite(in.time_to_expiry) || in.time_to_expiry < 0.0) {
        throw std::invalid_argument(
            "MonteCarloEngine: time_to_expiry must be finite and >= 0");
    }
}

// -----------------------------------------------------------------------------
// Payoff. Shared between calls and puts.
// -----------------------------------------------------------------------------

constexpr double payoff(double spot, double strike, OptionType type) noexcept {
    return type == OptionType::Call
        ? std::max(spot - strike, 0.0)
        : std::max(strike - spot, 0.0);
}

// -----------------------------------------------------------------------------
// Deterministic-limit result for sigma == 0 or T == 0. Matches
// BlackScholesEngine::deterministic_result — the price is the discounted
// intrinsic at the (deterministic) forward, and all Greeks are 0.
// -----------------------------------------------------------------------------

double deterministic_price(const MonteCarloEngine::Inputs& in) noexcept {
    const double df_r    = std::exp(-in.rate * in.time_to_expiry);
    const double forward = in.spot * std::exp((in.rate - in.dividend_yield)
                                               * in.time_to_expiry);
    return df_r * payoff(forward, in.strike, in.type);
}

// -----------------------------------------------------------------------------
// One Monte Carlo *sampling* pass. Runs Welford in-place; the caller
// owns the state.
//
// Returns `(mean, sample_variance, count)`. `count` is the number of
// Welford samples processed (= config.paths, may be 0 in the edge
// case config.paths == 0 which the caller must guard against).
//
// Welford's update rule (single-pass, numerically stable):
//     mean += (x - mean) / n
//     M2   += (x - mean_old) * (x - mean_new)
//     sample_variance = M2 / (n - 1)      // Bessel-corrected
//
// We track (mean, M2, n) rather than accumulating x and x². The
// naïve "sum of squares" formulation loses relative precision
// catastrophically for prices with high mean-to-variance ratio; see
// Welford (1962) and Chan-Golub-LeVeque (1983). For MC prices with
// millions of samples this can be a 6-digit accuracy improvement.
// -----------------------------------------------------------------------------

struct MonteCarloStats {
    double      mean{0.0};
    double      sample_variance{0.0};
    std::size_t count{0};
};

MonteCarloStats simulate(
    const MonteCarloEngine::Inputs& in,
    const MonteCarloEngine::Config& config,
    NormalGenerator& rng)
{
    const double T          = in.time_to_expiry;
    const double sqrt_T     = std::sqrt(T);
    const double drift      = (in.rate - in.dividend_yield
                                - 0.5 * in.volatility * in.volatility) * T;
    const double vol_sqrtT  = in.volatility * sqrt_T;
    const double df_r       = std::exp(-in.rate * T);

    double mean = 0.0;
    double M2   = 0.0;
    std::size_t n = 0;

    // The payoff is applied to the *discounted* final spot price so
    // that the running Welford accumulator tracks the discounted mean
    // directly — no bias adjustment at the end.
    const auto sample_at = [&](double z) noexcept {
        const double s_T = in.spot * std::exp(drift + vol_sqrtT * z);
        return df_r * payoff(s_T, in.strike, in.type);
    };

    const auto welford_update = [&](double x) noexcept {
        ++n;
        const double delta   = x - mean;
        mean += delta / static_cast<double>(n);
        const double delta_new = x - mean;   // uses the *updated* mean
        M2   += delta * delta_new;
    };

    if (config.antithetic_variates) {
        for (std::size_t i = 0; i < config.paths; ++i) {
            const double z    = rng.next();
            const double v_up = sample_at( z);
            const double v_dn = sample_at(-z);
            welford_update(0.5 * (v_up + v_dn));
        }
    } else {
        for (std::size_t i = 0; i < config.paths; ++i) {
            welford_update(sample_at(rng.next()));
        }
    }

    // Bessel-corrected sample variance. For n <= 1 it is undefined; we
    // report 0 (the caller will then produce a nonsense CI but the
    // point estimate is still well-defined).
    const double sample_variance = (n >= 2)
        ? M2 / static_cast<double>(n - 1)
        : 0.0;

    return MonteCarloStats{
        .mean = mean,
        .sample_variance = sample_variance,
        .count = n,
    };
}

// -----------------------------------------------------------------------------
// Numerical Greeks — bump-and-revalue with common random numbers.
//
// CRN reduces the variance of a bump-and-revalue Greek estimate by
// several orders of magnitude compared to running each bump with an
// *independent* seed: the noise in V(x + h) and V(x - h) is highly
// correlated when the same seed drives both, so their difference
// isolates the true derivative signal.
//
// Bump-size choices match `BinomialTreeEngine`:
//   Δ, Γ  : 1 % of spot          → central diff in S
//   ν     : 10 bp of vol         → central diff in σ
//   ρ     : 1 bp of rate         → central diff in r
//   Θ     : 1 calendar day       → forward diff toward expiration
//
// Cost: 8 additional Monte Carlo runs of `paths` samples each.
// Consider halving `paths` in Greeks-heavy workflows if that latency
// matters (CRN keeps the relative accuracy per path high).
// -----------------------------------------------------------------------------

constexpr double kDeltaBumpRelative = 0.01;
constexpr double kVegaBump          = 1.0e-3;
constexpr double kRhoBump           = 1.0e-4;
constexpr double kThetaBump         = 1.0 / 365.0;

Greeks compute_greeks(
    const MonteCarloEngine::Inputs& in,
    const MonteCarloEngine::Config& config,
    double center_price)
{
    // Local helper: run one bumped MC with a fresh generator seeded from
    // `config.seed` — this is what implements common random numbers.
    const auto revalue = [&](auto&& mutator) {
        auto perturbed = in;
        mutator(perturbed);
        MersenneTwisterNormalGenerator rng(config.seed);
        return simulate(perturbed, config, rng).mean;
    };

    // ---- Spot bumps (shared by Delta and Gamma) ----
    const double h_S    = kDeltaBumpRelative * in.spot;
    const double V_S_up = revalue([&](auto& p) { p.spot = in.spot + h_S; });
    const double V_S_dn = revalue([&](auto& p) { p.spot = in.spot - h_S; });

    Greeks g{};
    g.delta = (V_S_up - V_S_dn) / (2.0 * h_S);
    g.gamma = (V_S_up - 2.0 * center_price + V_S_dn) / (h_S * h_S);

    // ---- Vega (central diff in volatility) ----
    // Clamp σ_dn at 0; the denominator uses the actual (σ_up − σ_dn)
    // span so the estimator remains unbiased for the midpoint slope
    // even when the bump is one-sided at zero vol.
    const double h_sigma = kVegaBump;
    const double sigma_up = in.volatility + h_sigma;
    const double sigma_dn = std::max(in.volatility - h_sigma, 0.0);
    const double denom_sigma = sigma_up - sigma_dn;
    if (denom_sigma > 0.0) {
        const double V_v_up = revalue([&](auto& p) { p.volatility = sigma_up; });
        const double V_v_dn = revalue([&](auto& p) { p.volatility = sigma_dn; });
        g.vega = (V_v_up - V_v_dn) / denom_sigma;
    }

    // ---- Rho (central diff in rate) ----
    const double h_r    = kRhoBump;
    const double V_r_up = revalue([&](auto& p) { p.rate = in.rate + h_r; });
    const double V_r_dn = revalue([&](auto& p) { p.rate = in.rate - h_r; });
    g.rho = (V_r_up - V_r_dn) / (2.0 * h_r);

    // ---- Theta (forward diff toward expiration) ----
    // Θ = dV/dt where t is calendar time = -dV/dT.
    const double h_T  = kThetaBump;
    const double T_dn = std::max(in.time_to_expiry - h_T, 0.0);
    const double actual_h_T = in.time_to_expiry - T_dn;
    if (actual_h_T > 0.0) {
        const double V_T_dn = revalue([&](auto& p) { p.time_to_expiry = T_dn; });
        g.theta = -(center_price - V_T_dn) / actual_h_T;
    }

    return g;
}

// -----------------------------------------------------------------------------
// Engine-name builder — updated whenever the config changes.
// -----------------------------------------------------------------------------

std::string make_engine_name(const MonteCarloEngine::Config& c) {
    char buf[96];
    const int n = std::snprintf(
        buf, sizeof(buf),
        "MonteCarlo(paths=%zu, seed=%llu, antithetic=%s)",
        c.paths,
        static_cast<unsigned long long>(c.seed),
        c.antithetic_variates ? "true" : "false");
    return std::string(buf, (n > 0) ? static_cast<std::size_t>(n) : 0);
}

} // unnamed namespace

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

MonteCarloEngine::MonteCarloEngine()
    : config_{}, name_(make_engine_name(config_)) {}

MonteCarloEngine::MonteCarloEngine(Config config)
    : config_(config), name_(make_engine_name(config)) {}

PricingResult MonteCarloEngine::price(const Inputs& inputs) const {
    validate(inputs);
    if (config_.paths == 0) {
        throw std::invalid_argument(
            "MonteCarloEngine: Config::paths must be >= 1");
    }

    PricingResult r;
    r.engine_name = name_;

    // Boundary case: sigma == 0 or T == 0. The problem is deterministic,
    // so no simulation is needed and no confidence interval is
    // meaningful.
    if (inputs.volatility == 0.0 || inputs.time_to_expiry == 0.0) {
        r.price          = deterministic_price(inputs);
        r.iterations     = 0;
        r.standard_error = std::nullopt;
        r.confidence_interval_95 = std::nullopt;
        return r;
    }

    // Central simulation with a freshly seeded generator. Constructing
    // the RNG here (not in the ctor) makes `price()` reproducible
    // regardless of intervening usage.
    MersenneTwisterNormalGenerator rng(config_.seed);
    const auto stats = simulate(inputs, config_, rng);

    r.price      = stats.mean;
    r.iterations = stats.count;

    // Standard error is the sample stddev divided by sqrt(N).
    if (stats.count >= 2 && stats.sample_variance >= 0.0) {
        const double se = std::sqrt(stats.sample_variance
                                    / static_cast<double>(stats.count));
        r.standard_error = se;
        r.confidence_interval_95 = std::pair<double, double>{
            stats.mean - kZ95 * se,
            stats.mean + kZ95 * se,
        };
    }

    if (config_.compute_greeks) {
        r.greeks = compute_greeks(inputs, config_, stats.mean);
    }
    return r;
}

PricingResult MonteCarloEngine::price(
    const Option& option,
    const MarketSnapshot& market) const
{
    if (option.exercise != ExerciseStyle::European) {
        throw std::invalid_argument(
            "MonteCarloEngine only supports European exercise; received an "
            "American option.");
    }
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
