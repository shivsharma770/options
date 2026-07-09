/**
 * @file test_binomial_tree_engine.cpp
 * @brief Correctness tests for `ore::pricing::BinomialTreeEngine`.
 *
 * The suite exercises:
 *   1.  European pricing vs Black-Scholes reference.
 *   2.  Convergence: error decreases as steps → ∞.
 *   3.  American option identities:
 *         - American Call (no divs) ≈ European Call
 *         - American Put ≥ European Put, strictly for high enough r/T
 *   4.  Numerical Greeks vs Black-Scholes analytical Greeks.
 *   5.  Edge cases: zero vol, zero rate, negative rate, N=1, large N,
 *       near expiration, long expiration.
 *   6.  PricingEngine polymorphism: `PricingEngine* -> price(...)`
 *       returns a fully populated PricingResult identical in shape to
 *       BlackScholesEngine's.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string_view>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/types.hpp>
#include <ore/core/underlying.hpp>
#include <ore/numerics/comparison.hpp>
#include <ore/pricing/binomial_tree_engine.hpp>
#include <ore/pricing/black_scholes_engine.hpp>
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
using ore::pricing::PricingEngine;
using ore::pricing::PricingResult;

namespace {

BinomialTreeEngine::Inputs canonical_euro_call() {
    return {
        .spot           = 100.0,
        .strike         = 100.0,
        .rate           = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .time_to_expiry = 1.0,
        .type           = OptionType::Call,
        .exercise       = ExerciseStyle::European,
    };
}

BlackScholesEngine::Inputs to_bs(const BinomialTreeEngine::Inputs& in) {
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

double bs_price(const BinomialTreeEngine::Inputs& in) {
    BlackScholesEngine bs;
    auto bs_in = to_bs(in);
    return bs.price(bs_in).price;
}

double tree_price(const BinomialTreeEngine::Inputs& in, std::size_t steps) {
    BinomialTreeEngine tree({.steps = steps, .compute_greeks = false});
    return tree.price(in).price;
}

}  // namespace

// =============================================================================
// EUROPEAN PRICING vs BLACK-SCHOLES
// =============================================================================

TEST(BinomialTreeEuropeanTest, AtmCallMatchesBlackScholes) {
    // At N=500 CRR is accurate to a few cents at ATM.
    const auto in = canonical_euro_call();
    const double bs = bs_price(in);
    const double tr = tree_price(in, 500);
    EXPECT_NEAR(tr, bs, 0.01);
}

TEST(BinomialTreeEuropeanTest, AtmPutMatchesBlackScholes) {
    auto in = canonical_euro_call();
    in.type = OptionType::Put;
    const double bs = bs_price(in);
    const double tr = tree_price(in, 500);
    EXPECT_NEAR(tr, bs, 0.01);
}

TEST(BinomialTreeEuropeanTest, ItmCallMatchesBlackScholes) {
    auto in = canonical_euro_call();
    in.strike = 80.0;  // deep ITM call
    const double bs = bs_price(in);
    const double tr = tree_price(in, 500);
    EXPECT_NEAR(tr, bs, 0.02);
}

TEST(BinomialTreeEuropeanTest, OtmCallMatchesBlackScholes) {
    auto in = canonical_euro_call();
    in.strike = 120.0;  // OTM call
    const double bs = bs_price(in);
    const double tr = tree_price(in, 500);
    EXPECT_NEAR(tr, bs, 0.02);
}

TEST(BinomialTreeEuropeanTest, MultipleMaturitiesMatchBlackScholes) {
    for (double T : {0.1, 0.25, 0.5, 1.0, 2.0, 5.0}) {
        auto in = canonical_euro_call();
        in.time_to_expiry = T;
        const double bs = bs_price(in);
        const double tr = tree_price(in, 1000);
        // Longer T means bigger option value; use combined tol.
        EXPECT_TRUE(approximately_equal(tr, bs, 0.02, 5e-4))
            << "T=" << T << " tr=" << tr << " bs=" << bs;
    }
}

TEST(BinomialTreeEuropeanTest, DividendPayingUnderlyingMatchesBlackScholes) {
    auto in = canonical_euro_call();
    in.dividend_yield = 0.03;
    const double bs = bs_price(in);
    const double tr = tree_price(in, 1000);
    EXPECT_NEAR(tr, bs, 0.02);
}

// =============================================================================
// CONVERGENCE
// =============================================================================

TEST(BinomialTreeConvergenceTest, ErrorDecreasesWithStepsForEuropeanCall) {
    // CRR oscillates around the BS price (the "sawtooth"), but the
    // *envelope* decreases as 1/N. We compare N=50 with N=2000 and
    // require the higher-N error to be smaller.
    const auto in = canonical_euro_call();
    const double bs = bs_price(in);
    const double err_low  = std::abs(tree_price(in,   50) - bs);
    const double err_high = std::abs(tree_price(in, 2000) - bs);
    EXPECT_LT(err_high, err_low);
}

TEST(BinomialTreeConvergenceTest, ErrorEnvelopeShrinksAtLeastOncePerDecade) {
    // Between N=10 and N=1000 we should see the *maximum* error over
    // the two neighbouring N's contract by at least 10× overall.
    const auto in = canonical_euro_call();
    const double bs = bs_price(in);
    const double err_10   = std::abs(tree_price(in,   10) - bs);
    const double err_1000 = std::abs(tree_price(in, 1000) - bs);
    EXPECT_LT(err_1000, err_10 / 5.0);  // conservative 5× rather than 10×
}

TEST(BinomialTreeConvergenceTest, StandardConvergenceLadderPricesFinite) {
    // Verify prices are finite and positive for every step count in the
    // canonical convergence ladder — no NaN/Inf leakage even at N=10.
    const auto in = canonical_euro_call();
    for (std::size_t N : {10U, 25U, 50U, 100U, 250U, 500U, 1000U}) {
        const double p = tree_price(in, N);
        EXPECT_TRUE(std::isfinite(p)) << "N=" << N;
        EXPECT_GT(p, 0.0) << "N=" << N;
    }
}

// =============================================================================
// AMERICAN OPTIONS
// =============================================================================

TEST(BinomialTreeAmericanTest, AmericanCallWithoutDividendsEqualsEuropean) {
    // Merton (1973): an American call on a non-dividend-paying stock
    // should never be exercised early, hence American Call = European Call.
    auto euro = canonical_euro_call();
    auto amer = euro;
    amer.exercise = ExerciseStyle::American;

    const double v_euro = tree_price(euro, 500);
    const double v_amer = tree_price(amer, 500);
    EXPECT_NEAR(v_amer, v_euro, 1e-6);
}

TEST(BinomialTreeAmericanTest, AmericanPutStrictlyExceedsEuropeanPut) {
    // American puts have early-exercise value even without dividends
    // (unlike calls) because the time value of receiving the strike
    // early can outweigh the extrinsic value of holding.
    auto euro = canonical_euro_call();
    euro.type = OptionType::Put;
    euro.strike = 110.0;  // in-the-money put has clear early-exercise incentive
    auto amer = euro;
    amer.exercise = ExerciseStyle::American;

    const double v_euro = tree_price(euro, 500);
    const double v_amer = tree_price(amer, 500);
    EXPECT_GT(v_amer, v_euro + 1e-4);  // strictly greater by a measurable margin
}

TEST(BinomialTreeAmericanTest, AmericanCallWithDividendsMayExceedEuropean) {
    // With dividends, early exercise of an American call *can* be
    // optimal (to capture the dividend). The two prices should therefore
    // differ; the American should be >= European (never less).
    auto euro = canonical_euro_call();
    euro.dividend_yield = 0.06;  // high dividend
    euro.strike = 90.0;           // ITM call
    auto amer = euro;
    amer.exercise = ExerciseStyle::American;

    const double v_euro = tree_price(euro, 1000);
    const double v_amer = tree_price(amer, 1000);
    EXPECT_GE(v_amer, v_euro - 1e-6);
}

TEST(BinomialTreeAmericanTest, DeepItmAmericanPutApproachesIntrinsic) {
    // Deep-ITM American put's price should be close to its intrinsic
    // value (K - S) — the option holder can just exercise now.
    BinomialTreeEngine::Inputs in{
        .spot           = 50.0,
        .strike         = 100.0,
        .rate           = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .time_to_expiry = 1.0,
        .type           = OptionType::Put,
        .exercise       = ExerciseStyle::American,
    };
    const double p = tree_price(in, 500);
    const double intrinsic = 100.0 - 50.0;
    EXPECT_GE(p, intrinsic - 1e-6);
    EXPECT_LT(p, intrinsic + 5.0);  // upper bound: not far above intrinsic
}

TEST(BinomialTreeAmericanTest, DeepOtmAmericanPutIsNearZero) {
    // Deep-OTM American put should be close to zero: even with early
    // exercise, the option is worthless.
    BinomialTreeEngine::Inputs in{
        .spot           = 200.0,
        .strike         = 100.0,
        .rate           = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,
        .time_to_expiry = 1.0,
        .type           = OptionType::Put,
        .exercise       = ExerciseStyle::American,
    };
    const double p = tree_price(in, 500);
    EXPECT_GE(p, 0.0);
    EXPECT_LT(p, 1e-2);
}

// =============================================================================
// EARLY EXERCISE BEHAVIOUR
// =============================================================================

TEST(BinomialTreeAmericanTest, EarlyExerciseActivatedForInMoneyPut) {
    // Compare American put price against the naive backward-induction
    // "European put with the American structure but without the max()".
    // Using a moderately-in-the-money put with meaningful rate makes
    // early exercise strictly optimal.
    auto in = canonical_euro_call();
    in.type = OptionType::Put;
    in.strike = 105.0;
    in.rate = 0.08;
    in.exercise = ExerciseStyle::American;

    const double v_amer = tree_price(in, 500);

    auto euro = in;
    euro.exercise = ExerciseStyle::European;
    const double v_euro = tree_price(euro, 500);

    // Early-exercise premium is meaningfully positive:
    EXPECT_GT(v_amer - v_euro, 1e-3);
}

// =============================================================================
// GREEKS vs BLACK-SCHOLES ANALYTICAL
// =============================================================================

TEST(BinomialTreeGreeksTest, MatchBlackScholesForEuropeanAtm) {
    // At N=1000 the numerical Greeks should agree with BS to a few
    // basis points in relative terms.
    const auto in = canonical_euro_call();
    BinomialTreeEngine tree({.steps = 1000, .compute_greeks = true});
    const auto tr = tree.price(in);

    BlackScholesEngine bs;
    auto bs_in = to_bs(in);
    const auto bs_r = bs.price(bs_in);

    EXPECT_NEAR(tr.greeks.delta, bs_r.greeks.delta, 5e-3);
    EXPECT_NEAR(tr.greeks.gamma, bs_r.greeks.gamma, 5e-3);
    EXPECT_NEAR(tr.greeks.vega,  bs_r.greeks.vega,  5e-2);
    EXPECT_NEAR(tr.greeks.rho,   bs_r.greeks.rho,   5e-2);
    // Theta from a 1-day forward diff is inherently biased vs the true
    // instantaneous derivative; tolerate ~1% of vega scale.
    EXPECT_NEAR(tr.greeks.theta, bs_r.greeks.theta, 0.5);
}

TEST(BinomialTreeGreeksTest, GreeksDisabledLeaveDefaultZero) {
    BinomialTreeEngine tree({.steps = 100, .compute_greeks = false});
    const auto r = tree.price(canonical_euro_call());
    EXPECT_EQ(r.greeks.delta, 0.0);
    EXPECT_EQ(r.greeks.gamma, 0.0);
    EXPECT_EQ(r.greeks.vega,  0.0);
    EXPECT_EQ(r.greeks.rho,   0.0);
    EXPECT_EQ(r.greeks.theta, 0.0);
}

TEST(BinomialTreeGreeksTest, DeltaSignsMatchOptionType) {
    // Call delta ∈ [0, 1], put delta ∈ [-1, 0]. Basic sanity check.
    BinomialTreeEngine tree({.steps = 200});
    auto call = canonical_euro_call();
    auto put  = call;
    put.type = OptionType::Put;

    const auto rc = tree.price(call);
    const auto rp = tree.price(put);
    EXPECT_GT(rc.greeks.delta, 0.0);
    EXPECT_LT(rc.greeks.delta, 1.0);
    EXPECT_LT(rp.greeks.delta, 0.0);
    EXPECT_GT(rp.greeks.delta, -1.0);
}

TEST(BinomialTreeGreeksTest, GammaIsPositive) {
    // Vanilla options have non-negative gamma.
    BinomialTreeEngine tree({.steps = 500});
    const auto r = tree.price(canonical_euro_call());
    EXPECT_GT(r.greeks.gamma, 0.0);
}

// =============================================================================
// EDGE CASES
// =============================================================================

TEST(BinomialTreeEdgeCaseTest, ZeroVolatilityGivesDiscountedIntrinsic) {
    // sigma = 0: forward is known; price is df * max(F - K, 0) for call.
    auto in = canonical_euro_call();
    in.volatility = 0.0;
    const double tr = tree_price(in, 200);
    const double expected = std::exp(-in.rate * in.time_to_expiry)
        * std::max(in.spot * std::exp(in.rate * in.time_to_expiry) - in.strike, 0.0);
    EXPECT_NEAR(tr, expected, 1e-10);
}

TEST(BinomialTreeEdgeCaseTest, ZeroInterestRateWorks) {
    auto in = canonical_euro_call();
    in.rate = 0.0;
    const double bs = bs_price(in);
    const double tr = tree_price(in, 500);
    EXPECT_NEAR(tr, bs, 0.01);
}

TEST(BinomialTreeEdgeCaseTest, NegativeInterestRateWorks) {
    // Negative rates are supported: p just becomes smaller.
    auto in = canonical_euro_call();
    in.rate = -0.02;
    const double bs = bs_price(in);
    const double tr = tree_price(in, 500);
    EXPECT_NEAR(tr, bs, 0.01);
}

TEST(BinomialTreeEdgeCaseTest, OneStepTreeGivesFiniteFinitePrice) {
    // N=1 is a valid but very inaccurate tree — mainly checked for
    // absence of divide-by-zero or off-by-one crashes.
    auto in = canonical_euro_call();
    BinomialTreeEngine tree({.steps = 1, .compute_greeks = false});
    const auto r = tree.price(in);
    EXPECT_TRUE(std::isfinite(r.price));
    EXPECT_GT(r.price, 0.0);
}

TEST(BinomialTreeEdgeCaseTest, LargeTreeIsAccurate) {
    // N=5000 should be very close to BS. This also acts as a smoke test
    // that memory doesn't blow up (we use rolling O(N)).
    const auto in = canonical_euro_call();
    const double bs = bs_price(in);
    const double tr = tree_price(in, 5000);
    EXPECT_NEAR(tr, bs, 5e-3);
}

TEST(BinomialTreeEdgeCaseTest, ExpiredOptionEqualsIntrinsic) {
    auto in = canonical_euro_call();
    in.time_to_expiry = 0.0;
    in.spot = 110.0;
    // ATM+10 call with T=0 -> exactly 10.0
    BinomialTreeEngine tree;
    EXPECT_NEAR(tree.price(in).price, 10.0, 1e-10);
}

TEST(BinomialTreeEdgeCaseTest, NearExpirationCallMatchesBs) {
    // One-day-to-expiry call: very short T, tiny value but non-zero.
    auto in = canonical_euro_call();
    in.time_to_expiry = 1.0 / 365.0;
    const double bs = bs_price(in);
    const double tr = tree_price(in, 500);
    EXPECT_NEAR(tr, bs, 5e-3);
}

TEST(BinomialTreeEdgeCaseTest, LongExpirationWorks) {
    auto in = canonical_euro_call();
    in.time_to_expiry = 10.0;
    const double bs = bs_price(in);
    const double tr = tree_price(in, 1000);
    // Tolerance scales with option value at 10y (~30+ for ATM call).
    EXPECT_NEAR(tr, bs, 0.5);
}

// =============================================================================
// PricingEngine POLYMORPHISM AND INTERFACE
// =============================================================================

TEST(BinomialTreeInterfaceTest, PricesViaOptionAndMarketSnapshot) {
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
    BinomialTreeEngine tree({.steps = 500});
    const auto r = tree.price(option, market);
    EXPECT_TRUE(std::isfinite(r.price));
    EXPECT_GT(r.price, 0.0);
}

TEST(BinomialTreeInterfaceTest, IterationsFieldReportsStepCount) {
    BinomialTreeEngine tree({.steps = 250});
    const auto r = tree.price(canonical_euro_call());
    ASSERT_TRUE(r.iterations.has_value());
    EXPECT_EQ(*r.iterations, 250U);
    EXPECT_FALSE(r.standard_error.has_value());
}

TEST(BinomialTreeInterfaceTest, EngineNameEmbedsStepCount) {
    BinomialTreeEngine tree({.steps = 42});
    EXPECT_EQ(tree.name(), std::string_view{"Binomial(CRR, N=42)"});
    const auto r = tree.price(canonical_euro_call());
    EXPECT_EQ(r.engine_name, "Binomial(CRR, N=42)");
}

TEST(BinomialTreeInterfaceTest, InterchangeableViaBaseClassPointer) {
    // This is the *architectural* test the milestone spec calls out:
    // a caller holding a `PricingEngine*` can substitute either engine
    // without changing its own code.
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

    std::unique_ptr<PricingEngine> engines[] = {
        std::make_unique<BlackScholesEngine>(),
        std::make_unique<BinomialTreeEngine>(
            BinomialTreeEngine::Config{.steps = 1000, .compute_greeks = false}),
    };
    double last = -1.0;
    for (const auto& e : engines) {
        const auto r = e->price(option, market);
        EXPECT_TRUE(std::isfinite(r.price));
        EXPECT_GT(r.price, 0.0);
        if (last >= 0.0) {
            EXPECT_NEAR(r.price, last, 0.01);  // both engines agree on European
        }
        last = r.price;
    }
}

TEST(BinomialTreeInterfaceTest, DefaultConstructionProducesReasonableName) {
    BinomialTreeEngine tree;
    EXPECT_EQ(tree.name(), std::string_view{"Binomial(CRR, N=500)"});
}

TEST(BinomialTreeInterfaceTest, ZeroStepsRejected) {
    EXPECT_THROW(
        BinomialTreeEngine tree({.steps = 0}); (void)tree.price(canonical_euro_call()),
        std::invalid_argument);
}

TEST(BinomialTreeInterfaceTest, InvalidInputsRejected) {
    BinomialTreeEngine tree({.steps = 100});
    auto in = canonical_euro_call();
    in.spot = -1.0;
    EXPECT_THROW((void)tree.price(in), std::invalid_argument);

    in = canonical_euro_call();
    in.volatility = -0.01;
    EXPECT_THROW((void)tree.price(in), std::invalid_argument);

    in = canonical_euro_call();
    in.time_to_expiry = -1.0;
    EXPECT_THROW((void)tree.price(in), std::invalid_argument);
}
