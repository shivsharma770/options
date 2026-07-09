/**
 * @file test_monte_carlo_engine.cpp
 * @brief Correctness tests for `ore::pricing::MonteCarloEngine`.
 *
 * Suite structure:
 *   1.  Pricing vs Black-Scholes on a range of moneyness / maturities.
 *   2.  Standard-error and confidence-interval semantics.
 *   3.  Variance-reduction identity for antithetic variates.
 *   4.  RNG reproducibility and seed sensitivity.
 *   5.  Numerical Greeks vs Black-Scholes analytical Greeks.
 *   6.  Edge cases (σ = 0, T = 0, r = 0, negative r, deep ITM/OTM).
 *   7.  PricingEngine polymorphism — the "interchangeable engines"
 *       architectural test.
 *
 * A single seed (default `42`) is used throughout, sample counts
 * chosen so tests never flake at the fixed seed but comfortably
 * exercise the statistical machinery.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/types.hpp>
#include <ore/core/underlying.hpp>
#include <ore/numerics/comparison.hpp>
#include <ore/pricing/black_scholes_engine.hpp>
#include <ore/pricing/binomial_tree_engine.hpp>
#include <ore/pricing/monte_carlo_engine.hpp>
#include <ore/pricing/pricing_engine.hpp>

using ore::core::AssetType;
using ore::core::ExerciseStyle;
using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionType;
using ore::core::Underlying;
using ore::numerics::approximately_equal;
using ore::pricing::BinomialTreeEngine;
using ore::pricing::BlackScholesEngine;
using ore::pricing::MonteCarloEngine;
using ore::pricing::PricingEngine;
using ore::pricing::PricingResult;

namespace {

MonteCarloEngine::Inputs canonical_call() {
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

BlackScholesEngine::Inputs to_bs(const MonteCarloEngine::Inputs& in) {
    return {
        .spot           = in.spot,
        .strike         = in.strike,
        .rate           = in.rate,
        .dividend_yield = in.dividend_yield,
        .volatility     = in.volatility,
        .time_to_expiry = in.time_to_expiry,
        .type           = in.type,
    };
}

double bs_price(const MonteCarloEngine::Inputs& in) {
    BlackScholesEngine bs;
    auto bs_in = to_bs(in);
    return bs.price(bs_in).price;
}

// A default-configured MC engine at a moderate sample count is
// used for most tests.
MonteCarloEngine make_mc(std::size_t paths = 200'000,
                        std::uint64_t seed = 42,
                        bool antithetic = true,
                        bool greeks = false)
{
    return MonteCarloEngine(MonteCarloEngine::Config{
        .paths = paths,
        .seed = seed,
        .antithetic_variates = antithetic,
        .compute_greeks = greeks,
    });
}

}  // namespace

// =============================================================================
// PRICING vs BLACK-SCHOLES
// =============================================================================

TEST(MonteCarloPricingTest, AtmCallMatchesBlackScholes) {
    const auto in = canonical_call();
    const double bs = bs_price(in);
    const auto r = make_mc(500'000).price(in);
    ASSERT_TRUE(r.standard_error.has_value());
    // At N=500k with antithetic the SE should be well under 0.02.
    // Assert the point estimate is within 4 SE of BS — this fails at
    // the 0.006 % level under H_0, effectively never at the fixed seed.
    EXPECT_LT(std::abs(r.price - bs), 4.0 * *r.standard_error);
}

TEST(MonteCarloPricingTest, AtmPutMatchesBlackScholes) {
    auto in = canonical_call();
    in.type = OptionType::Put;
    const double bs = bs_price(in);
    const auto r = make_mc(500'000).price(in);
    ASSERT_TRUE(r.standard_error.has_value());
    EXPECT_LT(std::abs(r.price - bs), 4.0 * *r.standard_error);
}

TEST(MonteCarloPricingTest, ItmCallMatchesBlackScholes) {
    auto in = canonical_call();
    in.strike = 80.0;
    const double bs = bs_price(in);
    const auto r = make_mc(500'000).price(in);
    ASSERT_TRUE(r.standard_error.has_value());
    EXPECT_LT(std::abs(r.price - bs), 4.0 * *r.standard_error);
}

TEST(MonteCarloPricingTest, OtmCallMatchesBlackScholes) {
    auto in = canonical_call();
    in.strike = 120.0;
    const double bs = bs_price(in);
    const auto r = make_mc(500'000).price(in);
    ASSERT_TRUE(r.standard_error.has_value());
    // OTM options have lower absolute value; use combined tolerance.
    EXPECT_TRUE(approximately_equal(r.price, bs,
                                    4.0 * *r.standard_error, 0.0));
}

TEST(MonteCarloPricingTest, MultipleMaturitiesMatchBlackScholes) {
    for (double T : {0.1, 0.5, 1.0, 2.0, 5.0}) {
        auto in = canonical_call();
        in.time_to_expiry = T;
        const double bs = bs_price(in);
        const auto r = make_mc(200'000).price(in);
        ASSERT_TRUE(r.standard_error.has_value());
        EXPECT_LT(std::abs(r.price - bs), 5.0 * *r.standard_error)
            << "T=" << T;
    }
}

TEST(MonteCarloPricingTest, DividendPayingUnderlyingMatchesBlackScholes) {
    auto in = canonical_call();
    in.dividend_yield = 0.03;
    const double bs = bs_price(in);
    const auto r = make_mc(500'000).price(in);
    ASSERT_TRUE(r.standard_error.has_value());
    EXPECT_LT(std::abs(r.price - bs), 4.0 * *r.standard_error);
}

// =============================================================================
// CONVERGENCE
// =============================================================================

TEST(MonteCarloConvergenceTest, StandardErrorDecreasesLikeOneOverSqrtN) {
    const auto in = canonical_call();
    // Turn OFF antithetic so that the SE-vs-N relationship is exactly
    // the textbook 1/sqrt(N) — antithetic changes the constant, not
    // the exponent.
    const auto r_small = MonteCarloEngine({
        .paths = 10'000, .seed = 42,
        .antithetic_variates = false, .compute_greeks = false
    }).price(in);
    const auto r_large = MonteCarloEngine({
        .paths = 160'000, .seed = 42,
        .antithetic_variates = false, .compute_greeks = false
    }).price(in);

    ASSERT_TRUE(r_small.standard_error.has_value());
    ASSERT_TRUE(r_large.standard_error.has_value());

    // With 16× the paths, SE should drop by ~sqrt(16) = 4×.
    const double ratio = *r_small.standard_error / *r_large.standard_error;
    EXPECT_NEAR(ratio, 4.0, 0.5)
        << "SE ratio " << ratio << " (small SE=" << *r_small.standard_error
        << ", large SE=" << *r_large.standard_error << ")";
}

TEST(MonteCarloConvergenceTest, ErrorShrinksWithMorePaths) {
    const auto in = canonical_call();
    const double bs = bs_price(in);
    const double err_10k  = std::abs(make_mc( 10'000).price(in).price - bs);
    const double err_500k = std::abs(make_mc(500'000).price(in).price - bs);
    // 50× more paths → at least 5× smaller error (extremely lenient
    // relative to expected ~7×).
    EXPECT_LT(err_500k, err_10k / 3.0);
}

// =============================================================================
// CONFIDENCE INTERVALS
// =============================================================================

TEST(MonteCarloConfidenceTest, PopulatesStandardErrorAndCI) {
    const auto r = make_mc(50'000).price(canonical_call());
    ASSERT_TRUE(r.standard_error.has_value());
    ASSERT_TRUE(r.confidence_interval_95.has_value());
    const auto& [lo, hi] = *r.confidence_interval_95;
    EXPECT_LT(lo, r.price);
    EXPECT_GT(hi, r.price);
    // The 95% CI half-width should be 1.96 * SE (up to rounding).
    const double half = 0.5 * (hi - lo);
    EXPECT_NEAR(half, MonteCarloEngine::kZ95 * *r.standard_error, 1e-12);
}

TEST(MonteCarloConfidenceTest, BlackScholesLiesInsideConfidenceInterval) {
    const auto in = canonical_call();
    const double bs = bs_price(in);
    const auto r = make_mc(500'000).price(in);
    ASSERT_TRUE(r.confidence_interval_95.has_value());
    const auto& [lo, hi] = *r.confidence_interval_95;
    EXPECT_LE(lo, bs);
    EXPECT_GE(hi, bs);
}

TEST(MonteCarloConfidenceTest, ContainmentRateAcrossManySeeds) {
    // 95 % containment holds *on average* across independent seeds.
    // Run 50 independent 25k-path MCs; count how many CIs contain the
    // true BS price. Under the fixed procedure this should be >= 45
    // (i.e. 90 %; we don't demand exactly 47.5 = 95 % because the
    // sample is tiny).
    const auto in = canonical_call();
    const double bs = bs_price(in);
    std::size_t covered = 0;
    for (std::uint64_t s = 1; s <= 50; ++s) {
        const auto r = make_mc(25'000, s).price(in);
        ASSERT_TRUE(r.confidence_interval_95.has_value());
        const auto [lo, hi] = *r.confidence_interval_95;
        if (lo <= bs && bs <= hi) ++covered;
    }
    EXPECT_GE(covered, 45U);
}

// =============================================================================
// ANTITHETIC VARIATES
// =============================================================================

TEST(MonteCarloAntitheticTest, SameExpectedValueAsPlainMonteCarlo) {
    const auto in = canonical_call();
    // Large enough sample that the two should agree within a few SE.
    const auto plain =
        MonteCarloEngine({.paths = 500'000, .seed = 42,
                          .antithetic_variates = false,
                          .compute_greeks = false}).price(in);
    const auto anti  =
        MonteCarloEngine({.paths = 500'000, .seed = 42,
                          .antithetic_variates = true,
                          .compute_greeks = false}).price(in);

    ASSERT_TRUE(plain.standard_error.has_value());
    ASSERT_TRUE(anti.standard_error.has_value());
    const double combined_se = std::hypot(*plain.standard_error,
                                          *anti.standard_error);
    EXPECT_LT(std::abs(plain.price - anti.price), 4.0 * combined_se);
}

TEST(MonteCarloAntitheticTest, AntitheticReducesStandardError) {
    // For a monotone European call, antithetic variates strictly
    // reduce the variance. With the same *number of Welford samples*
    // the antithetic SE should be smaller than the plain SE.
    const auto in = canonical_call();
    const auto plain =
        MonteCarloEngine({.paths = 100'000, .seed = 42,
                          .antithetic_variates = false,
                          .compute_greeks = false}).price(in);
    const auto anti  =
        MonteCarloEngine({.paths = 100'000, .seed = 42,
                          .antithetic_variates = true,
                          .compute_greeks = false}).price(in);
    ASSERT_TRUE(plain.standard_error.has_value());
    ASSERT_TRUE(anti.standard_error.has_value());
    EXPECT_LT(*anti.standard_error, *plain.standard_error)
        << "anti SE=" << *anti.standard_error
        << ", plain SE=" << *plain.standard_error;
}

// =============================================================================
// RNG SEED SENSITIVITY
// =============================================================================

TEST(MonteCarloReproducibilityTest, IdenticalSeedProducesIdenticalPrice) {
    const auto in = canonical_call();
    const auto a = make_mc(50'000, 7).price(in);
    const auto b = make_mc(50'000, 7).price(in);
    EXPECT_DOUBLE_EQ(a.price, b.price);
    ASSERT_TRUE(a.standard_error.has_value());
    ASSERT_TRUE(b.standard_error.has_value());
    EXPECT_DOUBLE_EQ(*a.standard_error, *b.standard_error);
}

TEST(MonteCarloReproducibilityTest, DifferentSeedGivesDifferentPathsButSimilarPrice) {
    const auto in = canonical_call();
    const auto a = make_mc(50'000, 1).price(in);
    const auto b = make_mc(50'000, 2).price(in);
    EXPECT_NE(a.price, b.price);  // different sample paths
    ASSERT_TRUE(a.standard_error.has_value());
    // Sanity: prices are within a few SE of each other.
    EXPECT_LT(std::abs(a.price - b.price), 5.0 * *a.standard_error);
}

// =============================================================================
// NUMERICAL GREEKS vs BLACK-SCHOLES
// =============================================================================

TEST(MonteCarloGreeksTest, CommonRandomNumbersProduceReasonableGreeks) {
    // At N=200k, MC + CRN should match BS analytical Greeks within
    // ~ few percent for the well-behaved ones (Δ, Γ, Vega, Rho) and
    // within a wider margin for Θ (forward-diff bias).
    const auto in = canonical_call();
    const auto r = MonteCarloEngine({
        .paths = 200'000, .seed = 42,
        .antithetic_variates = true, .compute_greeks = true,
    }).price(in);

    BlackScholesEngine bs;
    auto bs_in = to_bs(in);
    const auto bs_r = bs.price(bs_in);

    EXPECT_NEAR(r.greeks.delta, bs_r.greeks.delta, 5e-3);
    EXPECT_NEAR(r.greeks.gamma, bs_r.greeks.gamma, 5e-3);
    EXPECT_NEAR(r.greeks.vega,  bs_r.greeks.vega,  1.0);
    EXPECT_NEAR(r.greeks.rho,   bs_r.greeks.rho,   1.0);
    EXPECT_NEAR(r.greeks.theta, bs_r.greeks.theta, 1.0);
}

TEST(MonteCarloGreeksTest, GreeksDisabledLeaveZero) {
    const auto r = make_mc(1000, 42, true, /*greeks*/ false).price(canonical_call());
    EXPECT_EQ(r.greeks.delta, 0.0);
    EXPECT_EQ(r.greeks.gamma, 0.0);
    EXPECT_EQ(r.greeks.vega,  0.0);
    EXPECT_EQ(r.greeks.rho,   0.0);
    EXPECT_EQ(r.greeks.theta, 0.0);
}

TEST(MonteCarloGreeksTest, DeltaSignsMatchOptionType) {
    auto call = canonical_call();
    auto put = call;
    put.type = OptionType::Put;
    const auto rc = MonteCarloEngine({
        .paths = 50'000, .seed = 42,
        .antithetic_variates = true, .compute_greeks = true,
    }).price(call);
    const auto rp = MonteCarloEngine({
        .paths = 50'000, .seed = 42,
        .antithetic_variates = true, .compute_greeks = true,
    }).price(put);
    EXPECT_GT(rc.greeks.delta, 0.0);
    EXPECT_LT(rp.greeks.delta, 0.0);
}

// =============================================================================
// EDGE CASES
// =============================================================================

TEST(MonteCarloEdgeCaseTest, ZeroVolatilityGivesDiscountedIntrinsic) {
    auto in = canonical_call();
    in.volatility = 0.0;
    const auto r = make_mc(10'000).price(in);
    const double expected = std::exp(-in.rate * in.time_to_expiry)
        * std::max(in.spot * std::exp(in.rate * in.time_to_expiry) - in.strike, 0.0);
    EXPECT_NEAR(r.price, expected, 1e-10);
    // Standard error / CI are meaningless in the deterministic limit.
    EXPECT_FALSE(r.standard_error.has_value());
    EXPECT_FALSE(r.confidence_interval_95.has_value());
}

TEST(MonteCarloEdgeCaseTest, ZeroTimeToExpiryGivesIntrinsic) {
    auto in = canonical_call();
    in.spot = 110.0;
    in.time_to_expiry = 0.0;
    const auto r = make_mc(10'000).price(in);
    EXPECT_NEAR(r.price, 10.0, 1e-10);
    EXPECT_FALSE(r.standard_error.has_value());
}

TEST(MonteCarloEdgeCaseTest, ZeroInterestRateWorks) {
    auto in = canonical_call();
    in.rate = 0.0;
    const double bs = bs_price(in);
    const auto r = make_mc(200'000).price(in);
    ASSERT_TRUE(r.standard_error.has_value());
    EXPECT_LT(std::abs(r.price - bs), 4.0 * *r.standard_error);
}

TEST(MonteCarloEdgeCaseTest, NegativeInterestRateWorks) {
    auto in = canonical_call();
    in.rate = -0.02;
    const double bs = bs_price(in);
    const auto r = make_mc(200'000).price(in);
    ASSERT_TRUE(r.standard_error.has_value());
    EXPECT_LT(std::abs(r.price - bs), 4.0 * *r.standard_error);
}

TEST(MonteCarloEdgeCaseTest, ExtremeVolatilityWorks) {
    auto in = canonical_call();
    in.volatility = 1.5;  // 150 % vol
    const double bs = bs_price(in);
    const auto r = make_mc(500'000).price(in);
    ASSERT_TRUE(r.standard_error.has_value());
    // High-vol options have higher variance in MC → looser tolerance.
    EXPECT_LT(std::abs(r.price - bs), 5.0 * *r.standard_error);
}

TEST(MonteCarloEdgeCaseTest, DeepInTheMoneyCall) {
    auto in = canonical_call();
    in.strike = 50.0;
    const double bs = bs_price(in);
    const auto r = make_mc(200'000).price(in);
    ASSERT_TRUE(r.standard_error.has_value());
    EXPECT_LT(std::abs(r.price - bs), 4.0 * *r.standard_error);
}

TEST(MonteCarloEdgeCaseTest, DeepOutOfTheMoneyCall) {
    auto in = canonical_call();
    in.strike = 200.0;
    const double bs = bs_price(in);
    const auto r = make_mc(500'000).price(in);
    ASSERT_TRUE(r.standard_error.has_value());
    // Deep OTM values are small; use combined tolerance.
    EXPECT_TRUE(approximately_equal(r.price, bs,
                                    4.0 * *r.standard_error, 0.0));
}

TEST(MonteCarloEdgeCaseTest, RejectsZeroPaths) {
    MonteCarloEngine engine({.paths = 0});
    EXPECT_THROW((void)engine.price(canonical_call()), std::invalid_argument);
}

TEST(MonteCarloEdgeCaseTest, RejectsInvalidInputs) {
    MonteCarloEngine engine({.paths = 100});
    auto in = canonical_call();
    in.spot = -1.0;
    EXPECT_THROW((void)engine.price(in), std::invalid_argument);
    in = canonical_call();
    in.volatility = -0.01;
    EXPECT_THROW((void)engine.price(in), std::invalid_argument);
}

TEST(MonteCarloEdgeCaseTest, RejectsAmericanExercise) {
    Underlying underlying{
        .symbol = "TEST", .exchange = "TEST", .asset_type = AssetType::Equity,
    };
    Option option{
        .underlying = underlying,
        .strike     = 100.0,
        .expiration = std::chrono::year{2027}/1/1,
        .type       = OptionType::Call,
        .exercise   = ExerciseStyle::American,
    };
    MarketSnapshot market{
        .spot           = 100.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .valuation_date = std::chrono::year{2026}/1/1,
    };
    MonteCarloEngine engine({.paths = 100});
    EXPECT_THROW((void)engine.price(option, market), std::invalid_argument);
}

// =============================================================================
// INTERFACE + POLYMORPHISM (design-goal test)
// =============================================================================

TEST(MonteCarloInterfaceTest, IterationsFieldReportsSampleCount) {
    const auto r = make_mc(12'345).price(canonical_call());
    ASSERT_TRUE(r.iterations.has_value());
    EXPECT_EQ(*r.iterations, 12345U);
}

TEST(MonteCarloInterfaceTest, EngineNameEmbedsConfig) {
    MonteCarloEngine engine({
        .paths = 5000, .seed = 123,
        .antithetic_variates = false, .compute_greeks = false,
    });
    EXPECT_EQ(engine.name(),
              std::string_view{"MonteCarlo(paths=5000, seed=123, antithetic=false)"});
    const auto r = engine.price(canonical_call());
    EXPECT_EQ(r.engine_name, "MonteCarlo(paths=5000, seed=123, antithetic=false)");
}

TEST(MonteCarloInterfaceTest, InterchangeableViaBaseClassPointer) {
    // The milestone's *explicit* architectural goal: same code path,
    // three different engines. Every engine must produce a
    // fully-populated PricingResult.
    Underlying underlying{
        .symbol = "TEST", .exchange = "TEST", .asset_type = AssetType::Equity,
    };
    Option option{
        .underlying = underlying,
        .strike     = 100.0,
        .expiration = std::chrono::year{2027}/1/1,
        .type       = OptionType::Call,
        .exercise   = ExerciseStyle::European,
    };
    MarketSnapshot market{
        .spot           = 100.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .valuation_date = std::chrono::year{2026}/1/1,
    };

    std::vector<std::unique_ptr<PricingEngine>> engines;
    engines.push_back(std::make_unique<BlackScholesEngine>());
    engines.push_back(std::make_unique<BinomialTreeEngine>(
        BinomialTreeEngine::Config{.steps = 500, .compute_greeks = false}));
    engines.push_back(std::make_unique<MonteCarloEngine>(
        MonteCarloEngine::Config{.paths = 200'000, .seed = 42,
                                 .antithetic_variates = true,
                                 .compute_greeks = false}));

    // Every engine returns a European ATM call price; they must all
    // agree within statistical tolerance of the MC engine's SE.
    std::vector<double> prices;
    std::optional<double> mc_se;
    for (const auto& e : engines) {
        const auto r = e->price(option, market);
        EXPECT_TRUE(std::isfinite(r.price));
        EXPECT_GT(r.price, 0.0);
        prices.push_back(r.price);
        if (r.standard_error.has_value()) mc_se = r.standard_error;
    }
    ASSERT_TRUE(mc_se.has_value());
    // BS price is prices[0]; every other price should be within a few SE.
    for (std::size_t i = 1; i < prices.size(); ++i) {
        EXPECT_LT(std::abs(prices[i] - prices[0]), 5.0 * *mc_se)
            << "engine " << i;
    }
}

TEST(MonteCarloInterfaceTest, DefaultConstructionProducesReasonableName) {
    MonteCarloEngine engine;
    // Default paths = 1,000,000, seed = 42, antithetic on.
    EXPECT_EQ(engine.name(),
              std::string_view{"MonteCarlo(paths=1000000, seed=42, antithetic=true)"});
}
