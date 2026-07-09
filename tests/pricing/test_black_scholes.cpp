/**
 * @file test_black_scholes.cpp
 * @brief Correctness tests for `ore::pricing::BlackScholesEngine`.
 *
 * See `docs/BLACK_SCHOLES_VALIDATION.md` for the source citations backing
 * every pinned numeric value. Analytical identities (parity, gamma
 * equality, ...) are verified without external reference data because
 * they hold exactly for any correct implementation.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/types.hpp>
#include <ore/numerics/comparison.hpp>
#include <ore/pricing/black_scholes_engine.hpp>

using ore::core::ExerciseStyle;
using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionType;
using ore::core::Underlying;
using ore::numerics::approximately_equal;
using ore::pricing::BlackScholesEngine;
using ore::pricing::PricingResult;

// -----------------------------------------------------------------------------
// Helper: build canonical Inputs with per-test overrides via a lambda that
// takes the Inputs by reference. Keeps individual tests short and readable.
// -----------------------------------------------------------------------------
namespace {

BlackScholesEngine::Inputs canonical_call() {
    return {
        .spot           = 100.0,
        .strike         = 100.0,
        .rate           = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .time_to_expiry = 1.0,
        .type           = OptionType::Call,
    };
}

BlackScholesEngine::Inputs canonical_put() {
    auto in = canonical_call();
    in.type = OptionType::Put;
    return in;
}

} // namespace

// =============================================================================
// PRICING — pinned reference values
// =============================================================================

// Reference: Hull, "Options, Futures, and Other Derivatives", 10th Ed.,
// Example 15.6. Hull publishes the call price to 3 decimals.
TEST(BlackScholesPricingTest, HullExample_15_6_Call) {
    BlackScholesEngine engine;
    BlackScholesEngine::Inputs in{
        .spot           = 42.0,
        .strike         = 40.0,
        .rate           = 0.10,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .time_to_expiry = 0.5,
        .type           = OptionType::Call,
    };
    const auto r = engine.price(in);
    EXPECT_NEAR(r.price, 4.7594, 5e-3);
    EXPECT_EQ(r.engine_name, "BlackScholes");
    EXPECT_FALSE(r.iterations.has_value());
    EXPECT_FALSE(r.standard_error.has_value());
}

// Reference: hand-computation to 12 digits from the Black-Scholes closed
// form using scipy.stats.norm.cdf(0.35) = 0.6368306511756191 and
// scipy.stats.norm.cdf(0.15) = 0.5596176923702426. Cross-referenced
// against QuantLib 1.31's AnalyticEuropeanEngine.
TEST(BlackScholesPricingTest, CanonicalATM_Call) {
    BlackScholesEngine engine;
    const auto r = engine.price(canonical_call());
    // Widely-published value across QuantLib/RQuantLib/macroption.com to
    // 5 significant figures. Pinned to 1e-4 so tests are not fragile to
    // the exact last-bit of my reference-value hand computation.
    EXPECT_NEAR(r.price, 10.4506, 1e-4);
}

TEST(BlackScholesPricingTest, CanonicalATM_Put) {
    BlackScholesEngine engine;
    const auto r = engine.price(canonical_put());
    EXPECT_NEAR(r.price, 5.5735, 1e-4);
}

TEST(BlackScholesPricingTest, ITM_CallHasHigherIntrinsic) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.spot = 120.0;
    const auto r = engine.price(in);
    // Deep ITM: price should exceed intrinsic (S - K*e^-rT).
    EXPECT_GT(r.price, 120.0 - 100.0 * std::exp(-0.05));
    // But be below spot (a call is always worth less than the underlying).
    EXPECT_LT(r.price, 120.0);
}

TEST(BlackScholesPricingTest, OTM_CallStaysPositive) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.spot = 80.0;
    const auto r = engine.price(in);
    // Deep OTM but not zero — time value is always positive.
    EXPECT_GT(r.price, 0.0);
    EXPECT_LT(r.price, 5.0);
}

TEST(BlackScholesPricingTest, ITM_PutHasHigherIntrinsic) {
    BlackScholesEngine engine;
    auto in = canonical_put();
    in.spot = 80.0;
    const auto r = engine.price(in);
    EXPECT_GT(r.price, 100.0 * std::exp(-0.05) - 80.0);
    EXPECT_LT(r.price, 100.0);
}

TEST(BlackScholesPricingTest, OTM_PutStaysPositive) {
    BlackScholesEngine engine;
    auto in = canonical_put();
    in.spot = 120.0;
    const auto r = engine.price(in);
    EXPECT_GT(r.price, 0.0);
    EXPECT_LT(r.price, 5.0);
}

// =============================================================================
// GREEKS — pinned reference values (canonical case)
// =============================================================================

TEST(BlackScholesGreeksTest, CanonicalCall_AllFive) {
    BlackScholesEngine engine;
    const auto r = engine.price(canonical_call());
    // References: computed from the closed form using
    //   Phi(0.35) = 0.6368306511756191
    //   phi(0.35) = 0.3752403469169692
    //   Phi(0.15) = 0.5596176923702426
    //   exp(-0.05) = 0.9512294245007140
    // Delta, gamma, and vega involve only a single evaluation of N or phi
    // so are effectively exact; theta and rho involve products of three
    // high-precision constants and are pinned to fewer digits.
    EXPECT_NEAR(r.greeks.delta,  0.6368306511756191, 1e-10);
    EXPECT_NEAR(r.greeks.gamma,  0.0187620173458485, 1e-10);
    EXPECT_NEAR(r.greeks.vega,  37.5240346916969,   1e-8);
    EXPECT_NEAR(r.greeks.theta, -6.41403,           1e-4);
    EXPECT_NEAR(r.greeks.rho,   53.23248,           1e-4);
}

TEST(BlackScholesGreeksTest, CanonicalPut_AllFive) {
    BlackScholesEngine engine;
    const auto r = engine.price(canonical_put());
    // For q = 0: Delta_put = Delta_call - 1;
    // Theta_put = Theta_call + r*K*e^-rT; Rho_put = Rho_call - K*T*e^-rT.
    // Gamma and Vega are identical between call and put.
    EXPECT_NEAR(r.greeks.delta, -0.3631693488243809, 1e-10);
    EXPECT_NEAR(r.greeks.gamma,  0.0187620173458485, 1e-10);
    EXPECT_NEAR(r.greeks.vega,  37.5240346916969,   1e-8);
    EXPECT_NEAR(r.greeks.theta, -1.65790,           1e-4);
    EXPECT_NEAR(r.greeks.rho,  -41.89046,           1e-4);
}

// =============================================================================
// ANALYTIC IDENTITIES — hold to machine precision for any correct BS impl
// =============================================================================

/**
 * Put-call parity (with continuous dividend):
 *   C - P = S * exp(-qT) - K * exp(-rT)
 * Tested across a grid of parameters — any violation of parity is a
 * model-independent bug indicator.
 */
TEST(BlackScholesIdentitiesTest, PutCallParity) {
    BlackScholesEngine engine;
    for (double S : {80.0, 100.0, 120.0}) {
        for (double K : {80.0, 100.0, 120.0}) {
            for (double r : {0.0, 0.05, 0.10}) {
                for (double q : {0.0, 0.03}) {
                    for (double sigma : {0.10, 0.30}) {
                        for (double T : {0.25, 1.0, 2.0}) {
                            BlackScholesEngine::Inputs base{
                                .spot           = S,
                                .strike         = K,
                                .rate           = r,
                                .dividend_yield = q,
                                .volatility     = sigma,
                                .time_to_expiry = T,
                                .type           = OptionType::Call,
                            };
                            const double C = engine.price(base).price;
                            base.type = OptionType::Put;
                            const double P = engine.price(base).price;

                            const double lhs = C - P;
                            const double rhs = S * std::exp(-q * T)
                                             - K * std::exp(-r * T);
                            EXPECT_TRUE(approximately_equal(lhs, rhs, 1e-12, 1e-12))
                                << "parity violation: S=" << S << " K=" << K
                                << " r=" << r << " q=" << q
                                << " sigma=" << sigma << " T=" << T
                                << "; C-P=" << lhs << " expected " << rhs;
                        }
                    }
                }
            }
        }
    }
}

/**
 * Delta identity: `Delta_call - Delta_put = exp(-qT)`.
 */
TEST(BlackScholesIdentitiesTest, CallPutDeltaRelationship) {
    BlackScholesEngine engine;
    for (double q : {0.0, 0.02, 0.05}) {
        for (double T : {0.25, 1.0}) {
            auto in = canonical_call();
            in.dividend_yield = q;
            in.time_to_expiry = T;
            const double dc = engine.price(in).greeks.delta;
            in.type = OptionType::Put;
            const double dp = engine.price(in).greeks.delta;

            EXPECT_TRUE(approximately_equal(dc - dp, std::exp(-q * T),
                                             1e-14, 1e-13))
                << "delta identity violated at q=" << q << " T=" << T;
        }
    }
}

/**
 * Gamma equality: `Gamma_call == Gamma_put`.
 */
TEST(BlackScholesIdentitiesTest, GammaEquality) {
    BlackScholesEngine engine;
    for (double S : {80.0, 100.0, 120.0}) {
        auto in = canonical_call();
        in.spot = S;
        const double gc = engine.price(in).greeks.gamma;
        in.type = OptionType::Put;
        const double gp = engine.price(in).greeks.gamma;
        EXPECT_DOUBLE_EQ(gc, gp);
    }
}

/**
 * Vega equality: `Vega_call == Vega_put`.
 */
TEST(BlackScholesIdentitiesTest, VegaEquality) {
    BlackScholesEngine engine;
    for (double sigma : {0.10, 0.20, 0.50}) {
        auto in = canonical_call();
        in.volatility = sigma;
        const double vc = engine.price(in).greeks.vega;
        in.type = OptionType::Put;
        const double vp = engine.price(in).greeks.vega;
        EXPECT_DOUBLE_EQ(vc, vp);
    }
}

/**
 * Rho identity: `Rho_call - Rho_put = K * T * exp(-rT)`.
 */
TEST(BlackScholesIdentitiesTest, RhoIdentity) {
    BlackScholesEngine engine;
    for (double r : {0.0, 0.03, 0.08}) {
        auto in = canonical_call();
        in.rate = r;
        const double rc = engine.price(in).greeks.rho;
        in.type = OptionType::Put;
        const double rp = engine.price(in).greeks.rho;
        const double expected = in.strike * in.time_to_expiry
                              * std::exp(-r * in.time_to_expiry);
        EXPECT_TRUE(approximately_equal(rc - rp, expected, 1e-10, 1e-12))
            << "rho identity violated at r=" << r;
    }
}

// =============================================================================
// FINITE-DIFFERENCE VALIDATION OF GREEKS
// =============================================================================
// Compare each analytical Greek against a central finite-difference estimate.
// This is one of the strongest correctness checks: it exercises the entire
// pricing pipeline (formula + N + phi + exp + log) and only agrees with the
// analytic Greek if every piece is correct.

namespace {

// Central difference for a single-input bump. `bump` mutates a copy of `in`.
template <typename BumpFn>
double central_diff(const BlackScholesEngine& engine,
                    BlackScholesEngine::Inputs in,
                    BumpFn&& bump,
                    double h)
{
    auto up = in;   bump(up,    h);
    auto dn = in;   bump(dn,   -h);
    return (engine.price(up).price - engine.price(dn).price) / (2.0 * h);
}

} // namespace

TEST(BlackScholesFiniteDiffTest, DeltaMatchesAnalytical) {
    BlackScholesEngine engine;
    for (double S : {80.0, 100.0, 120.0}) {
        for (OptionType type : {OptionType::Call, OptionType::Put}) {
            auto in = canonical_call();
            in.spot = S;
            in.type = type;
            const double analytical = engine.price(in).greeks.delta;
            const double h = 1e-3 * S;
            const double numerical = central_diff(engine, in,
                [](auto& x, double dh) { x.spot += dh; }, h);
            EXPECT_TRUE(approximately_equal(analytical, numerical, 1e-6, 1e-5))
                << "delta mismatch at S=" << S << " type=" << to_string(type)
                << ": analytical=" << analytical << " fd=" << numerical;
        }
    }
}

TEST(BlackScholesFiniteDiffTest, GammaMatchesAnalytical) {
    BlackScholesEngine engine;
    for (double S : {80.0, 100.0, 120.0}) {
        auto in = canonical_call();
        in.spot = S;
        const double analytical = engine.price(in).greeks.gamma;
        const double h = 1e-2 * S;  // larger h for second-order FD to
                                     // balance truncation vs rounding
        auto up = in; up.spot += h;
        auto dn = in; dn.spot -= h;
        const double numerical =
            (engine.price(up).price - 2.0 * engine.price(in).price
             + engine.price(dn).price) / (h * h);
        EXPECT_TRUE(approximately_equal(analytical, numerical, 1e-6, 1e-4))
            << "gamma mismatch at S=" << S
            << ": analytical=" << analytical << " fd=" << numerical;
    }
}

TEST(BlackScholesFiniteDiffTest, VegaMatchesAnalytical) {
    BlackScholesEngine engine;
    for (double sigma : {0.10, 0.20, 0.40}) {
        for (OptionType type : {OptionType::Call, OptionType::Put}) {
            auto in = canonical_call();
            in.volatility = sigma;
            in.type = type;
            const double analytical = engine.price(in).greeks.vega;
            const double numerical = central_diff(engine, in,
                [](auto& x, double dh) { x.volatility += dh; }, 1e-4);
            EXPECT_TRUE(approximately_equal(analytical, numerical, 1e-4, 1e-5))
                << "vega mismatch at sigma=" << sigma
                << " type=" << to_string(type);
        }
    }
}

TEST(BlackScholesFiniteDiffTest, ThetaMatchesAnalytical) {
    BlackScholesEngine engine;
    // theta = dV/dt with t = calendar time. Since T = T_expiry - t,
    // theta = -dV/dT. Our FD bumps T, so we negate the result.
    for (double T : {0.25, 1.0, 2.0}) {
        for (OptionType type : {OptionType::Call, OptionType::Put}) {
            auto in = canonical_call();
            in.time_to_expiry = T;
            in.type = type;
            const double analytical = engine.price(in).greeks.theta;
            const double dV_dT = central_diff(engine, in,
                [](auto& x, double dh) { x.time_to_expiry += dh; }, 1e-4);
            const double numerical = -dV_dT;
            EXPECT_TRUE(approximately_equal(analytical, numerical, 1e-4, 1e-4))
                << "theta mismatch at T=" << T
                << " type=" << to_string(type);
        }
    }
}

TEST(BlackScholesFiniteDiffTest, RhoMatchesAnalytical) {
    BlackScholesEngine engine;
    for (double r : {0.0, 0.05, 0.10}) {
        for (OptionType type : {OptionType::Call, OptionType::Put}) {
            auto in = canonical_call();
            in.rate = r;
            in.type = type;
            const double analytical = engine.price(in).greeks.rho;
            const double numerical = central_diff(engine, in,
                [](auto& x, double dh) { x.rate += dh; }, 1e-5);
            EXPECT_TRUE(approximately_equal(analytical, numerical, 1e-4, 1e-5))
                << "rho mismatch at r=" << r
                << " type=" << to_string(type);
        }
    }
}

// =============================================================================
// EDGE CASES
// =============================================================================

TEST(BlackScholesEdgeCaseTest, TimeToExpiryZero_Call) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.time_to_expiry = 0.0;
    // ITM at expiry -> intrinsic S - K, delta = 1, other Greeks = 0.
    in.spot = 120.0;
    const auto r1 = engine.price(in);
    EXPECT_DOUBLE_EQ(r1.price, 20.0);
    EXPECT_DOUBLE_EQ(r1.greeks.delta, 1.0);
    EXPECT_DOUBLE_EQ(r1.greeks.gamma, 0.0);
    EXPECT_DOUBLE_EQ(r1.greeks.vega, 0.0);
    EXPECT_DOUBLE_EQ(r1.greeks.theta, 0.0);
    EXPECT_DOUBLE_EQ(r1.greeks.rho, 0.0);
    // OTM at expiry -> price 0, delta 0.
    in.spot = 80.0;
    const auto r2 = engine.price(in);
    EXPECT_DOUBLE_EQ(r2.price, 0.0);
    EXPECT_DOUBLE_EQ(r2.greeks.delta, 0.0);
}

TEST(BlackScholesEdgeCaseTest, TimeToExpiryZero_Put) {
    BlackScholesEngine engine;
    auto in = canonical_put();
    in.time_to_expiry = 0.0;
    in.spot = 80.0;
    const auto r1 = engine.price(in);
    EXPECT_DOUBLE_EQ(r1.price, 20.0);
    EXPECT_DOUBLE_EQ(r1.greeks.delta, -1.0);
    in.spot = 120.0;
    const auto r2 = engine.price(in);
    EXPECT_DOUBLE_EQ(r2.price, 0.0);
    EXPECT_DOUBLE_EQ(r2.greeks.delta, 0.0);
}

TEST(BlackScholesEdgeCaseTest, VolatilityZero) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.volatility = 0.0;
    // Forward: 100 * exp(0.05) = 105.127... > K = 100, so ITM in forward space.
    const auto r = engine.price(in);
    const double forward = in.spot * std::exp(in.rate * in.time_to_expiry);
    const double expected = std::exp(-in.rate * in.time_to_expiry)
                          * (forward - in.strike);
    EXPECT_NEAR(r.price, expected, 1e-13);
    EXPECT_NEAR(r.greeks.delta, std::exp(0.0), 1e-13);  // e^{-qT} = 1
    EXPECT_DOUBLE_EQ(r.greeks.gamma, 0.0);
    EXPECT_DOUBLE_EQ(r.greeks.vega, 0.0);
    EXPECT_DOUBLE_EQ(r.greeks.theta, 0.0);
    EXPECT_DOUBLE_EQ(r.greeks.rho, 0.0);
}

TEST(BlackScholesEdgeCaseTest, VeryLowVolatilityStable) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.volatility = 1e-6;
    // Formula should give roughly the deterministic-limit value.
    const auto r = engine.price(in);
    EXPECT_TRUE(std::isfinite(r.price));
    EXPECT_TRUE(std::isfinite(r.greeks.gamma));
    // Delta should be very close to e^{-qT} because we are far into
    // "sigma near zero, forward > strike" territory.
    EXPECT_NEAR(r.greeks.delta, 1.0, 1e-4);
}

TEST(BlackScholesEdgeCaseTest, VeryHighVolatilityStable) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.volatility = 5.0;   // 500% annualized — extreme but valid
    const auto r = engine.price(in);
    EXPECT_TRUE(std::isfinite(r.price));
    EXPECT_GE(r.price, 0.0);
    EXPECT_LE(r.price, in.spot);
    // As sigma -> infinity, call price -> S (intuition: any strike is
    // near-guaranteed to be exceeded).
    EXPECT_NEAR(r.price, in.spot, 5.0);
}

TEST(BlackScholesEdgeCaseTest, NearExpirationStable) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.time_to_expiry = 1.0 / 365.0;  // 1 day
    const auto r = engine.price(in);
    EXPECT_TRUE(std::isfinite(r.price));
    EXPECT_TRUE(std::isfinite(r.greeks.gamma));
    EXPECT_GE(r.price, 0.0);
}

TEST(BlackScholesEdgeCaseTest, LongExpirationStable) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.time_to_expiry = 30.0;  // 30-year LEAPs
    const auto r = engine.price(in);
    EXPECT_TRUE(std::isfinite(r.price));
    EXPECT_LT(r.price, in.spot);
}

TEST(BlackScholesEdgeCaseTest, ZeroInterestRate) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.rate = 0.0;
    const auto r = engine.price(in);
    // Rho analytical = K*T*exp(-rT)*N(d2). With r=0, e^-rT=1.
    EXPECT_GT(r.price, 0.0);
    EXPECT_TRUE(std::isfinite(r.greeks.rho));
}

TEST(BlackScholesEdgeCaseTest, NegativeInterestRate) {
    // Post-2015 negative-rate regimes: BS is well-defined for r < 0.
    // Verify prices stay reasonable and put-call parity still holds.
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.rate = -0.02;
    const double C = engine.price(in).price;
    in.type = OptionType::Put;
    const double P = engine.price(in).price;
    const double expected = in.spot * std::exp(-in.dividend_yield
                                               * in.time_to_expiry)
                          - in.strike * std::exp(-in.rate * in.time_to_expiry);
    EXPECT_TRUE(approximately_equal(C - P, expected, 1e-12, 1e-12));
}

TEST(BlackScholesEdgeCaseTest, DeepITMCall_ApproachesStockMinusDiscountedStrike) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.spot = 500.0;  // deep, deep ITM
    const auto r = engine.price(in);
    // Deep ITM call ~ S*e^{-qT} - K*e^{-rT}, and delta ~ e^{-qT}.
    const double lower_bound = in.spot * std::exp(-in.dividend_yield
                                                  * in.time_to_expiry)
                             - in.strike * std::exp(-in.rate
                                                    * in.time_to_expiry);
    EXPECT_NEAR(r.price, lower_bound, 1e-6);
    EXPECT_NEAR(r.greeks.delta, std::exp(-in.dividend_yield
                                          * in.time_to_expiry), 1e-6);
}

TEST(BlackScholesEdgeCaseTest, DeepOTMPut_ApproachesZero) {
    BlackScholesEngine engine;
    auto in = canonical_put();
    in.spot = 500.0;
    const auto r = engine.price(in);
    EXPECT_NEAR(r.price, 0.0, 1e-6);
    EXPECT_NEAR(r.greeks.delta, 0.0, 1e-6);
}

// =============================================================================
// INVALID INPUTS — precondition violations must throw
// =============================================================================

TEST(BlackScholesInvalidInputTest, RejectsNonPositiveSpot) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.spot = 0.0;
    EXPECT_THROW(engine.price(in), std::invalid_argument);
    in.spot = -1.0;
    EXPECT_THROW(engine.price(in), std::invalid_argument);
}

TEST(BlackScholesInvalidInputTest, RejectsNonPositiveStrike) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.strike = 0.0;
    EXPECT_THROW(engine.price(in), std::invalid_argument);
    in.strike = -1.0;
    EXPECT_THROW(engine.price(in), std::invalid_argument);
}

TEST(BlackScholesInvalidInputTest, RejectsNegativeVolatility) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.volatility = -0.01;
    EXPECT_THROW(engine.price(in), std::invalid_argument);
}

TEST(BlackScholesInvalidInputTest, RejectsNegativeTimeToExpiry) {
    BlackScholesEngine engine;
    auto in = canonical_call();
    in.time_to_expiry = -0.001;
    EXPECT_THROW(engine.price(in), std::invalid_argument);
}

TEST(BlackScholesInvalidInputTest, RejectsNonFiniteInputs) {
    BlackScholesEngine engine;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // Each mutator has a distinct closure type, so they cannot share a
    // braced-init-list directly (no common type to deduce). Wrapping in
    // std::function homogenises the element type.
    using Mutator = std::function<void(BlackScholesEngine::Inputs&)>;
    for (const Mutator& set : {
             Mutator{[&](auto& in) { in.spot = nan; }},
             Mutator{[&](auto& in) { in.strike = nan; }},
             Mutator{[&](auto& in) { in.rate = nan; }},
             Mutator{[&](auto& in) { in.dividend_yield = nan; }},
             Mutator{[&](auto& in) { in.volatility = nan; }},
             Mutator{[&](auto& in) { in.time_to_expiry = nan; }},
         }) {
        auto in = canonical_call();
        set(in);
        EXPECT_THROW(engine.price(in), std::invalid_argument);
    }
}

TEST(BlackScholesInvalidInputTest, RejectsAmericanExercise) {
    BlackScholesEngine engine;
    Option opt{
        .underlying = Underlying{},
        .strike     = 100.0,
        .expiration = std::chrono::year_month_day{
            std::chrono::year{2027}, std::chrono::month{7},
            std::chrono::day{8}},
        .type       = OptionType::Call,
        .exercise   = ExerciseStyle::American,
    };
    MarketSnapshot m{
        .spot           = 100.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .valuation_date = std::chrono::year_month_day{
            std::chrono::year{2026}, std::chrono::month{7},
            std::chrono::day{8}},
    };
    EXPECT_THROW(engine.price(opt, m), std::invalid_argument);
}

// =============================================================================
// DATE-BASED OVERLOAD
// =============================================================================

TEST(BlackScholesDateInterfaceTest, MatchesRawOverloadOn1YearHorizon) {
    BlackScholesEngine engine;

    Option opt{
        .underlying = Underlying{},
        .strike     = 100.0,
        .expiration = std::chrono::year_month_day{
            std::chrono::year{2027}, std::chrono::month{7},
            std::chrono::day{8}},
        .type       = OptionType::Call,
        .exercise   = ExerciseStyle::European,
    };
    MarketSnapshot m{
        .spot           = 100.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .valuation_date = std::chrono::year_month_day{
            std::chrono::year{2026}, std::chrono::month{7},
            std::chrono::day{8}},
    };
    const auto date_result = engine.price(opt, m);

    // 2026-07-08 to 2027-07-08 is 365 calendar days -> T = 1.0 exactly.
    const auto raw_result = engine.price(canonical_call());
    EXPECT_DOUBLE_EQ(date_result.price, raw_result.price);
    EXPECT_DOUBLE_EQ(date_result.greeks.delta, raw_result.greeks.delta);
}

TEST(BlackScholesDateInterfaceTest, ExpiredOptionThrows) {
    // Expiration strictly before valuation date -> T < 0.
    BlackScholesEngine engine;
    Option opt{
        .underlying = Underlying{},
        .strike     = 100.0,
        .expiration = std::chrono::year_month_day{
            std::chrono::year{2025}, std::chrono::month{7},
            std::chrono::day{8}},
        .type       = OptionType::Call,
        .exercise   = ExerciseStyle::European,
    };
    MarketSnapshot m{
        .spot           = 100.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .valuation_date = std::chrono::year_month_day{
            std::chrono::year{2026}, std::chrono::month{7},
            std::chrono::day{8}},
    };
    EXPECT_THROW(engine.price(opt, m), std::invalid_argument);
}

TEST(BlackScholesDateInterfaceTest, ExpirationEqualsValuationIsZeroTime) {
    // Same-day pricing -> T = 0 exactly -> intrinsic path.
    BlackScholesEngine engine;
    Option opt{
        .underlying = Underlying{},
        .strike     = 100.0,
        .expiration = std::chrono::year_month_day{
            std::chrono::year{2026}, std::chrono::month{7},
            std::chrono::day{8}},
        .type       = OptionType::Call,
        .exercise   = ExerciseStyle::European,
    };
    MarketSnapshot m{
        .spot           = 120.0,  // ITM
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .valuation_date = std::chrono::year_month_day{
            std::chrono::year{2026}, std::chrono::month{7},
            std::chrono::day{8}},
    };
    const auto r = engine.price(opt, m);
    EXPECT_DOUBLE_EQ(r.price, 20.0);  // intrinsic (S - K)
}

// =============================================================================
// ENGINE IDENTITY
// =============================================================================

TEST(BlackScholesEngineIdentityTest, NameIsStable) {
    BlackScholesEngine engine;
    EXPECT_EQ(engine.name(), "BlackScholes");
}

TEST(BlackScholesEngineIdentityTest, PopulatesEngineNameInResult) {
    BlackScholesEngine engine;
    const auto r = engine.price(canonical_call());
    EXPECT_EQ(r.engine_name, "BlackScholes");
}

TEST(BlackScholesEngineIdentityTest, StochasticDiagnosticsLeftEmpty) {
    BlackScholesEngine engine;
    const auto r = engine.price(canonical_call());
    EXPECT_FALSE(r.iterations.has_value());
    EXPECT_FALSE(r.standard_error.has_value());
}
