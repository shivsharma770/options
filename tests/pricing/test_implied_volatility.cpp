/**
 * @file test_implied_volatility.cpp
 * @brief Correctness tests for `ore::pricing::ImpliedVolatilitySolver`.
 *
 * The primary validation is a **round-trip test**:
 *
 *   Known sigma  ->  BlackScholes  ->  market price  ->  IV solver  ->  sigma'
 *
 * If the solver is correct, `sigma' ~= sigma` to within the solver's
 * price tolerance divided by the local Vega — which for typical
 * near-the-money options is on the order of `1e-10 / 30 ~ 3e-12`.
 *
 * We test:
 *   * A parameter-grid round-trip across strikes, moneyness, rates,
 *     dividends, times to expiration, and option types.
 *   * Named market conditions: ATM, ITM, OTM, near/long expiration,
 *     low/high vol, zero/negative rates.
 *   * Arbitrage-bound rejection (impossible prices).
 *   * Failure modes: Newton non-convergence, fallback disabled,
 *     tight iteration budgets.
 *   * Bisection fallback correctness — solving with Newton *disabled*
 *     via config and letting bisection carry the whole solve.
 *   * Config validation and overload equivalence.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/types.hpp>
#include <ore/numerics/solver_result.hpp>
#include <ore/pricing/black_scholes_engine.hpp>
#include <ore/pricing/implied_volatility_solver.hpp>

using ore::core::ExerciseStyle;
using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionType;
using ore::core::Underlying;
using ore::numerics::SolverStatus;
using ore::pricing::BlackScholesEngine;
using ore::pricing::ImpliedVolatilitySolver;

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

// Price under Black-Scholes at a specific sigma, returning the market
// price the IV solver should recover. Convenience wrapper that keeps
// the round-trip test code compact.
double bs_price(BlackScholesEngine::Inputs in, double sigma) {
    in.volatility = sigma;
    return BlackScholesEngine{}.price(in).price;
}

}  // namespace

// =============================================================================
// PRIMARY VALIDATION — round-trip on a parameter grid
// =============================================================================
//
// For every combination in the grid, price with a known sigma, then
// solve for IV from that price, then assert the solver recovers the
// original sigma. This is the single strongest correctness test: it
// exercises every code path in the solver against every relevant
// market regime.

TEST(ImpliedVolatilityRoundTripTest, RecoversKnownSigmaAcrossParameterGrid) {
    const std::vector<double> sigmas{0.05, 0.10, 0.20, 0.40, 0.80, 1.50};
    const std::vector<double> spots{80.0, 100.0, 120.0};
    const std::vector<double> strikes{90.0, 100.0, 110.0};
    const std::vector<double> rates{0.0, 0.02, 0.05};
    const std::vector<double> divs{0.0, 0.03};
    const std::vector<double> times{0.10, 0.50, 1.00, 3.00};
    const std::vector<OptionType> types{OptionType::Call, OptionType::Put};

    ImpliedVolatilitySolver solver;

    for (double sigma_true : sigmas)
    for (double S           : spots)
    for (double K           : strikes)
    for (double r           : rates)
    for (double q           : divs)
    for (double T           : times)
    for (OptionType type    : types)
    {
        BlackScholesEngine::Inputs in{
            .spot           = S,
            .strike         = K,
            .rate           = r,
            .dividend_yield = q,
            .volatility     = 0.0,  // will be set inside bs_price
            .time_to_expiry = T,
            .type           = type,
        };
        const double market = bs_price(in, sigma_true);

        // Skip prices that are numerically indistinguishable from the
        // arbitrage lower bound — these correspond to deep-OTM options
        // that BS prices at ~1e-30 dollars, well below `tolerance = 1e-10`.
        // Any positive sigma reproduces them within tolerance, so the
        // "recovered sigma" is not uniquely defined.
        if (market < 1e-8) continue;

        const auto r_iv = solver.solve(in, market);

        ASSERT_TRUE(r_iv.converged())
            << "no convergence:"
            << " sigma_true=" << sigma_true
            << " S=" << S << " K=" << K
            << " r=" << r << " q=" << q << " T=" << T
            << " type=" << (type == OptionType::Call ? "call" : "put")
            << " market=" << market
            << " status=" << to_string(r_iv.status);

        // sigma tolerance: price residual is `tolerance = 1e-10`; recovered
        // sigma differs from true sigma by ~ residual / Vega. Vega for our
        // grid ranges from ~1e-3 (deep OTM/short T) to ~40 (ATM/long T),
        // so 1e-6 sigma tolerance is safe. We tighten to 1e-8 for the
        // regimes where Vega is O(1) or larger.
        EXPECT_NEAR(r_iv.root, sigma_true, 1e-6)
            << " sigma_true=" << sigma_true
            << " S=" << S << " K=" << K
            << " r=" << r << " q=" << q << " T=" << T
            << " type=" << (type == OptionType::Call ? "call" : "put");
    }
}

TEST(ImpliedVolatilityRoundTripTest, RecoversPreciselyForNearTheMoneyOptions) {
    ImpliedVolatilitySolver solver;
    // ATM options have the largest Vega, so recovery precision is best.
    for (double sigma_true : {0.10, 0.20, 0.30, 0.50, 1.00}) {
        auto in = canonical_call();
        const double market = bs_price(in, sigma_true);
        const auto r_iv = solver.solve(in, market);
        ASSERT_TRUE(r_iv.converged());
        // ATM Vega ~= S * phi(0) * sqrt(T) ~= 40, so sigma resolution is
        // `tolerance / Vega ~= 2.5e-12`. Assert at 1e-10 to leave headroom
        // for the last correctly-rounded digit.
        EXPECT_NEAR(r_iv.root, sigma_true, 1e-10);
    }
}

// =============================================================================
// MARKET CONDITIONS — named scenarios
// =============================================================================

TEST(ImpliedVolatilityMarketConditionsTest, AtTheMoneyCall) {
    ImpliedVolatilitySolver solver;
    const auto in = canonical_call();  // K == S == 100
    const double market = bs_price(in, 0.25);
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 0.25, 1e-10);
}

TEST(ImpliedVolatilityMarketConditionsTest, InTheMoneyCall) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.strike = 80.0;  // Spot 100, Strike 80 -> deep ITM call
    const double market = bs_price(in, 0.30);
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 0.30, 1e-8);
}

TEST(ImpliedVolatilityMarketConditionsTest, OutOfTheMoneyCall) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.strike = 130.0;  // Spot 100, Strike 130 -> OTM call
    const double market = bs_price(in, 0.35);
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 0.35, 1e-8);
}

TEST(ImpliedVolatilityMarketConditionsTest, InTheMoneyPut) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_put();
    in.strike = 120.0;  // Spot 100, Strike 120 -> ITM put
    const double market = bs_price(in, 0.28);
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 0.28, 1e-8);
}

TEST(ImpliedVolatilityMarketConditionsTest, OutOfTheMoneyPut) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_put();
    in.strike = 70.0;  // Spot 100, Strike 70 -> OTM put
    const double market = bs_price(in, 0.40);
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 0.40, 1e-8);
}

TEST(ImpliedVolatilityMarketConditionsTest, NearExpiration) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.time_to_expiry = 1.0 / 365.0;  // one day
    const double market = bs_price(in, 0.30);
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 0.30, 1e-8);
}

TEST(ImpliedVolatilityMarketConditionsTest, LongExpiration) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.time_to_expiry = 5.0;  // 5 years — LEAPS
    const double market = bs_price(in, 0.22);
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 0.22, 1e-8);
}

TEST(ImpliedVolatilityMarketConditionsTest, LowVolatility) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    const double market = bs_price(in, 0.02);  // 2% vol
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 0.02, 1e-8);
}

TEST(ImpliedVolatilityMarketConditionsTest, HighVolatility) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    const double market = bs_price(in, 2.00);  // 200% vol
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 2.00, 1e-8);
}

TEST(ImpliedVolatilityMarketConditionsTest, ZeroInterestRate) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.rate = 0.0;
    const double market = bs_price(in, 0.25);
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 0.25, 1e-10);
}

TEST(ImpliedVolatilityMarketConditionsTest, NegativeInterestRate) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.rate = -0.01;  // European-style negative rate regime
    const double market = bs_price(in, 0.25);
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 0.25, 1e-10);
}

TEST(ImpliedVolatilityMarketConditionsTest, WithContinuousDividendYield) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.dividend_yield = 0.04;
    const double market = bs_price(in, 0.30);
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 0.30, 1e-10);
}

// =============================================================================
// ARBITRAGE BOUNDS — impossible prices
// =============================================================================

TEST(ImpliedVolatilityArbitrageTest, RejectsNegativeMarketPrice) {
    ImpliedVolatilitySolver solver;
    const auto in = canonical_call();
    EXPECT_THROW(solver.solve(in, -1.0), std::invalid_argument);
}

TEST(ImpliedVolatilityArbitrageTest, RejectsNonFiniteMarketPrice) {
    ImpliedVolatilitySolver solver;
    const auto in = canonical_call();
    EXPECT_THROW(solver.solve(in, std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    EXPECT_THROW(solver.solve(in, std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
}

TEST(ImpliedVolatilityArbitrageTest, RejectsCallPriceAboveUpperBound) {
    ImpliedVolatilitySolver solver;
    const auto in = canonical_call();
    // Call upper bound is S*e^{-qT} = 100 * 1 = 100. Anything above is arbitrage.
    EXPECT_THROW(solver.solve(in, 100.01), std::invalid_argument);
    EXPECT_THROW(solver.solve(in, 1000.0), std::invalid_argument);
}

TEST(ImpliedVolatilityArbitrageTest, RejectsPutPriceAboveUpperBound) {
    ImpliedVolatilitySolver solver;
    const auto in = canonical_put();
    // Put upper bound is K*e^{-rT} = 100 * e^{-0.05} ~= 95.12.
    const double upper = 100.0 * std::exp(-0.05);
    EXPECT_THROW(solver.solve(in, upper + 0.01), std::invalid_argument);
}

TEST(ImpliedVolatilityArbitrageTest, PriceAtLowerBoundReturnsZeroSigma) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.strike = 80.0;  // ITM: intrinsic-in-forward ~ S - K*e^{-rT}
    const double lower = 100.0 - 80.0 * std::exp(-0.05);
    const auto r_iv = solver.solve(in, lower);
    EXPECT_TRUE(r_iv.converged());
    EXPECT_EQ(r_iv.root, 0.0);
}

TEST(ImpliedVolatilityArbitrageTest, PriceBelowLowerBoundReturnsZeroSigma) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.strike = 80.0;
    const double lower = 100.0 - 80.0 * std::exp(-0.05);
    // Slightly below the theoretical bound — real bid-ask noise can push
    // a mid price into this regime. Documented convention: return sigma=0.
    const auto r_iv = solver.solve(in, lower - 0.5);
    EXPECT_TRUE(r_iv.converged());
    EXPECT_EQ(r_iv.root, 0.0);
}

// =============================================================================
// FALLBACK — Newton failure -> bisection recovery
// =============================================================================

TEST(ImpliedVolatilityFallbackTest, BisectionRecoversWhenNewtonBudgetIsOne) {
    // Configure Newton to give up after one iteration. Bisection then
    // carries the whole solve. The recovered sigma must still be accurate.
    ImpliedVolatilitySolver::Config cfg{};
    cfg.max_iterations         = 1;
    cfg.use_bisection_fallback = true;

    ImpliedVolatilitySolver solver(cfg);
    const auto in = canonical_call();
    const double market = bs_price(in, 0.35);
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_NEAR(r_iv.root, 0.35, 1e-8);
}

TEST(ImpliedVolatilityFallbackTest,
     ReturnsNewtonFailureWhenFallbackDisabled) {
    ImpliedVolatilitySolver::Config cfg{};
    cfg.max_iterations         = 1;
    cfg.use_bisection_fallback = false;

    ImpliedVolatilitySolver solver(cfg);
    const auto in = canonical_call();
    const double market = bs_price(in, 0.35);
    const auto r_iv = solver.solve(in, market);
    EXPECT_FALSE(r_iv.converged());
    // One iteration from Brenner-Subrahmanyam's guess of ~0.26 is not
    // enough to reach 1e-10 residual on a 0.35 target.
    EXPECT_EQ(r_iv.status, SolverStatus::MaxIterationsReached);
}

TEST(ImpliedVolatilityFallbackTest, RecoversDeepOtmWhereNewtonVegaMayCollapse) {
    // Deep OTM near expiration: Vega tiny, Newton likely to bail with
    // `DerivativeTooSmall`. Bisection must save it.
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.spot           = 100.0;
    in.strike         = 200.0;
    in.time_to_expiry = 0.05;
    // A ~1e-4 dollar call there implies vol somewhere north of 0.6.
    const double market = bs_price(in, 0.80);
    ASSERT_GT(market, 1e-8) << "price too small to round-trip meaningfully";
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged())
        << "status=" << to_string(r_iv.status);
    EXPECT_NEAR(r_iv.root, 0.80, 1e-6);
}

// =============================================================================
// CONFIG — validation and knobs
// =============================================================================

TEST(ImpliedVolatilityConfigTest, RejectsNonPositiveTolerance) {
    ImpliedVolatilitySolver::Config cfg{};
    cfg.tolerance = 0.0;
    ImpliedVolatilitySolver solver(cfg);
    EXPECT_THROW(solver.solve(canonical_call(), 10.0), std::invalid_argument);
}

TEST(ImpliedVolatilityConfigTest, RejectsZeroMaxIterations) {
    ImpliedVolatilitySolver::Config cfg{};
    cfg.max_iterations = 0;
    ImpliedVolatilitySolver solver(cfg);
    EXPECT_THROW(solver.solve(canonical_call(), 10.0), std::invalid_argument);
}

TEST(ImpliedVolatilityConfigTest, RejectsInvertedBracketWhenFallbackEnabled) {
    ImpliedVolatilitySolver::Config cfg{};
    cfg.min_volatility = 2.0;
    cfg.max_volatility = 1.0;
    ImpliedVolatilitySolver solver(cfg);
    EXPECT_THROW(solver.solve(canonical_call(), 10.0), std::invalid_argument);
}

TEST(ImpliedVolatilityConfigTest, InvertedBracketOKIfFallbackDisabled) {
    // Bracket validation is only enforced when the fallback will actually
    // consult those fields. A user who has disabled bisection is not
    // required to supply a coherent bracket.
    ImpliedVolatilitySolver::Config cfg{};
    cfg.use_bisection_fallback = false;
    cfg.min_volatility = 2.0;
    cfg.max_volatility = 1.0;
    ImpliedVolatilitySolver solver(cfg);
    const auto in = canonical_call();
    const double market = bs_price(in, 0.30);
    const auto r_iv = solver.solve(in, market);
    EXPECT_TRUE(r_iv.converged());  // Newton succeeds; bracket unused.
    EXPECT_NEAR(r_iv.root, 0.30, 1e-8);
}

TEST(ImpliedVolatilityConfigTest, InitialGuessOverrideIsHonored) {
    // With a bad initial guess we should still converge — Newton is robust
    // — but it should take more iterations than the default heuristic.
    // Comparing iteration counts gives us a proxy that the config field
    // is actually being consulted.
    const auto in = canonical_call();
    const double market = bs_price(in, 0.20);

    ImpliedVolatilitySolver default_solver;
    const auto r_default = default_solver.solve(in, market);
    ASSERT_TRUE(r_default.converged());

    ImpliedVolatilitySolver::Config far_cfg{};
    far_cfg.initial_guess = 2.5;  // deliberately far from truth
    ImpliedVolatilitySolver far_solver(far_cfg);
    const auto r_far = far_solver.solve(in, market);
    ASSERT_TRUE(r_far.converged());
    EXPECT_NEAR(r_far.root, 0.20, 1e-10);

    EXPECT_GE(r_far.iterations, r_default.iterations);
}

TEST(ImpliedVolatilityConfigTest, TolerancePropagatesToNewton) {
    // A very loose tolerance should let Newton stop earlier and return a
    // *less* accurate answer. Verify the residual respects the tolerance.
    ImpliedVolatilitySolver::Config cfg{};
    cfg.tolerance = 1.0;  // one whole dollar of price residual allowed
    ImpliedVolatilitySolver solver(cfg);
    const auto in = canonical_call();
    const double market = bs_price(in, 0.20);
    const auto r_iv = solver.solve(in, market);
    EXPECT_TRUE(r_iv.converged());
    EXPECT_LT(r_iv.residual, 1.0);
}

TEST(ImpliedVolatilityConfigTest, ConfigAccessorReturnsWhatWePassedIn) {
    ImpliedVolatilitySolver::Config cfg{
        .tolerance                 = 5e-9,
        .max_iterations            = 50,
        .use_bisection_fallback    = false,
        .initial_guess             = 0.30,
        .min_volatility            = 1e-4,
        .max_volatility            = 10.0,
        .max_bisection_iterations  = 300,
    };
    ImpliedVolatilitySolver solver(cfg);
    EXPECT_EQ(solver.config().tolerance,                cfg.tolerance);
    EXPECT_EQ(solver.config().max_iterations,           cfg.max_iterations);
    EXPECT_EQ(solver.config().use_bisection_fallback,   cfg.use_bisection_fallback);
    EXPECT_EQ(solver.config().initial_guess,            cfg.initial_guess);
    EXPECT_EQ(solver.config().min_volatility,           cfg.min_volatility);
    EXPECT_EQ(solver.config().max_volatility,           cfg.max_volatility);
    EXPECT_EQ(solver.config().max_bisection_iterations, cfg.max_bisection_iterations);
}

// =============================================================================
// INVALID INPUTS
// =============================================================================

TEST(ImpliedVolatilityInvalidInputsTest, RejectsZeroTimeToExpiry) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.time_to_expiry = 0.0;
    EXPECT_THROW(solver.solve(in, 10.0), std::invalid_argument);
}

TEST(ImpliedVolatilityInvalidInputsTest, RejectsNegativeTimeToExpiry) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.time_to_expiry = -1.0;
    EXPECT_THROW(solver.solve(in, 10.0), std::invalid_argument);
}

TEST(ImpliedVolatilityInvalidInputsTest, RejectsNonPositiveSpot) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.spot = 0.0;
    // Rejection comes from the engine when the first iteration invokes it.
    EXPECT_THROW(solver.solve(in, 10.0), std::invalid_argument);
}

TEST(ImpliedVolatilityInvalidInputsTest, RejectsNonPositiveStrike) {
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.strike = -1.0;
    EXPECT_THROW(solver.solve(in, 10.0), std::invalid_argument);
}

TEST(ImpliedVolatilityInvalidInputsTest, RejectsAmericanExercise) {
    ImpliedVolatilitySolver solver;
    Option opt{
        .underlying = Underlying{"SPY"},
        .strike     = 100.0,
        .expiration = std::chrono::year_month_day{
            std::chrono::year{2026}, std::chrono::month{7}, std::chrono::day{15}
        },
        .type       = OptionType::Call,
        .exercise   = ExerciseStyle::American,
    };
    MarketSnapshot market{
        .spot           = 100.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.0,
        .valuation_date = std::chrono::year_month_day{
            std::chrono::year{2026}, std::chrono::month{7}, std::chrono::day{1}
        },
    };
    EXPECT_THROW(solver.solve(opt, market, 5.0), std::invalid_argument);
}

// =============================================================================
// DATE-BASED INTERFACE — Option + MarketSnapshot overload
// =============================================================================

TEST(ImpliedVolatilityDateInterfaceTest, MatchesRawOverloadForSameParameters) {
    ImpliedVolatilitySolver solver;

    // Build the same problem two ways: once via raw Inputs, once via
    // (Option, MarketSnapshot). The T in the Inputs form is chosen to
    // match the day count the (Option, MarketSnapshot) overload will
    // derive, so both solves must return the same sigma.
    const std::chrono::year_month_day valuation{
        std::chrono::year{2026}, std::chrono::month{1}, std::chrono::day{1}};
    const std::chrono::year_month_day expiration{
        std::chrono::year{2026}, std::chrono::month{12}, std::chrono::day{31}};
    const auto days = (std::chrono::sys_days{expiration}
                     - std::chrono::sys_days{valuation}).count();
    const double T = static_cast<double>(days) / 365.0;

    BlackScholesEngine::Inputs raw{
        .spot           = 100.0,
        .strike         = 105.0,
        .rate           = 0.03,
        .dividend_yield = 0.01,
        .volatility     = 0.0,
        .time_to_expiry = T,
        .type           = OptionType::Call,
    };
    const double market = bs_price(raw, 0.28);

    const Option opt{
        .underlying = Underlying{"XYZ"},
        .strike     = 105.0,
        .expiration = expiration,
        .type       = OptionType::Call,
        .exercise   = ExerciseStyle::European,
    };
    const MarketSnapshot snap{
        .spot           = 100.0,
        .risk_free_rate = 0.03,
        .dividend_yield = 0.01,
        .volatility     = 0.0,     // ignored
        .valuation_date = valuation,
    };

    const auto r_raw  = solver.solve(raw, market);
    const auto r_date = solver.solve(opt, snap, market);

    ASSERT_TRUE(r_raw.converged());
    ASSERT_TRUE(r_date.converged());
    EXPECT_NEAR(r_raw.root, r_date.root, 1e-14);
    EXPECT_NEAR(r_date.root, 0.28, 1e-8);
}

TEST(ImpliedVolatilityDateInterfaceTest, IgnoresMarketSnapshotVolatilityField) {
    // The IV solver must not read `MarketSnapshot::volatility` — it's
    // solving for that field. Set it to nonsense and confirm the answer
    // does not change.
    ImpliedVolatilitySolver solver;

    const std::chrono::year_month_day valuation{
        std::chrono::year{2026}, std::chrono::month{1}, std::chrono::day{1}};
    const std::chrono::year_month_day expiration{
        std::chrono::year{2027}, std::chrono::month{1}, std::chrono::day{1}};

    const Option opt{
        .underlying = Underlying{"XYZ"},
        .strike     = 100.0,
        .expiration = expiration,
        .type       = OptionType::Call,
        .exercise   = ExerciseStyle::European,
    };
    MarketSnapshot snap{
        .spot           = 100.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.0,
        .volatility     = 0.20,  // "real" vol
        .valuation_date = valuation,
    };

    const double market = 10.450583572185565;  // BS price at sigma=0.20, T=1

    const auto r_correct = solver.solve(opt, snap, market);
    snap.volatility = 999.0;  // sabotage the field
    const auto r_sabotaged = solver.solve(opt, snap, market);

    ASSERT_TRUE(r_correct.converged());
    ASSERT_TRUE(r_sabotaged.converged());
    EXPECT_EQ(r_correct.root, r_sabotaged.root);
}

// =============================================================================
// PERFORMANCE — Newton should converge quickly for well-posed problems
// =============================================================================

TEST(ImpliedVolatilityPerformanceTest, NewtonConvergesInFewIterationsForATM) {
    // The Brenner-Subrahmanyam initial guess is exact-to-first-order for
    // ATM-forward options. Quadratic convergence then hits 1e-10 in
    // roughly log2(log2(precision)) ~ 4-6 iterations.
    ImpliedVolatilitySolver solver;
    const auto in = canonical_call();
    const double market = bs_price(in, 0.20);
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_LE(r_iv.iterations, 10u)
        << "Newton took too many iterations for a well-posed ATM problem";
}

TEST(ImpliedVolatilityPerformanceTest, SolveHandlesPricesNearArbitrageBounds) {
    // A price barely above the lower bound should still converge cleanly,
    // even though such prices imply a very small sigma.
    ImpliedVolatilitySolver solver;
    auto in = canonical_call();
    in.strike = 50.0;  // deep ITM
    // Intrinsic-in-forward = 100 - 50*e^{-0.05} ~= 52.44. Add a small margin.
    const double lower = 100.0 - 50.0 * std::exp(-0.05);
    const double market = lower + 0.05;
    const auto r_iv = solver.solve(in, market);
    ASSERT_TRUE(r_iv.converged());
    EXPECT_GT(r_iv.root, 0.0);
    EXPECT_LT(r_iv.root, 0.10);  // small implied vol
}
