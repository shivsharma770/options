/**
 * @file test_volatility_analytics.cpp
 * @brief Correctness tests for `ore::analytics::volatility_analytics`.
 *
 * The suite exercises smile grouping, moneyness variants, term-structure
 * ATM interpolation, surface layout, skew metrics, and IV statistics.
 * Every calibration report is synthesised in-memory from
 * `BlackScholesEngine` prices so the round-trip precision floor is
 * ~1e-8 (the IV solver's tolerance divided by ATM Vega).
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ore/analytics/option_chain_calibrator.hpp>
#include <ore/analytics/volatility_analytics.hpp>
#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/option_chain.hpp>
#include <ore/core/option_market_snapshot.hpp>
#include <ore/core/quote.hpp>
#include <ore/core/types.hpp>
#include <ore/core/underlying.hpp>
#include <ore/pricing/black_scholes_engine.hpp>

using ore::analytics::CalibrationReport;
using ore::analytics::compute_skew_metrics;
using ore::analytics::compute_statistics;
using ore::analytics::compute_statistics_by_expiration;
using ore::analytics::compute_statistics_by_type;
using ore::analytics::build_smiles;
using ore::analytics::build_surface;
using ore::analytics::build_term_structure;
using ore::analytics::IVStatistics;
using ore::analytics::Moneyness;
using ore::analytics::OptionChainCalibrator;
using ore::analytics::SkewMetrics;
using ore::analytics::TermStructure;
using ore::analytics::VolatilitySmile;
using ore::analytics::VolatilitySurface;
using ore::core::AssetType;
using ore::core::ExerciseStyle;
using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionChain;
using ore::core::OptionMarketSnapshot;
using ore::core::OptionType;
using ore::core::Quote;
using ore::core::Underlying;
using ore::pricing::BlackScholesEngine;

namespace {

std::chrono::year_month_day ymd(int y, unsigned m, unsigned d) {
    return std::chrono::year{y} / std::chrono::month{m} / std::chrono::day{d};
}

MarketSnapshot canonical_market() {
    return MarketSnapshot{
        .spot           = 100.0,
        .risk_free_rate = 0.05,
        .dividend_yield = 0.02,
        .volatility     = 0.0,
        .valuation_date = ymd(2026, 1, 1),
    };
}

Underlying canonical_underlying() {
    return Underlying{
        .symbol = "TEST", .exchange = "TEST", .asset_type = AssetType::Equity
    };
}

/**
 * Construct a calibrated snapshot at (K, expiration, type, sigma) by
 * setting the market mid to `BlackScholes(sigma)` — the IV solver will
 * then recover `sigma` to full precision.
 */
OptionMarketSnapshot synth_snap(
    const MarketSnapshot& market,
    double strike,
    std::chrono::year_month_day expiration,
    OptionType type,
    double sigma_true,
    std::string contract_symbol)
{
    const double T =
        static_cast<double>((std::chrono::sys_days{expiration}
                          - std::chrono::sys_days{market.valuation_date}).count())
        / 365.0;

    BlackScholesEngine engine;
    const double mid = engine.price(BlackScholesEngine::Inputs{
        .spot           = market.spot,
        .strike         = strike,
        .rate           = market.risk_free_rate,
        .dividend_yield = market.dividend_yield,
        .volatility     = sigma_true,
        .time_to_expiry = T,
        .type           = type,
    }).price;

    OptionMarketSnapshot s{};
    s.option = Option{
        .underlying = canonical_underlying(),
        .strike     = strike,
        .expiration = expiration,
        .type       = type,
        .exercise   = ExerciseStyle::European,
    };
    s.quote = Quote{
        .bid                = mid,
        .ask                = mid,
        .last               = mid,
        .volume             = 1.0,
        .open_interest      = 1.0,
        .implied_volatility = sigma_true,
        .timestamp          = std::chrono::system_clock::now(),
    };
    s.contract_symbol = std::move(contract_symbol);
    return s;
}

/**
 * Build a synthetic 2-expiration, 5-strike chain with per-strike
 * skew (deep-OTM puts have higher IV than deep-OTM calls — a stylised
 * equity smile). Used across many tests.
 */
CalibrationReport build_stylized_report(const MarketSnapshot& market) {
    const auto exp1 = ymd(2026, 4, 1);   // ~3 months
    const auto exp2 = ymd(2026, 10, 1);  // ~9 months

    // Skew pattern: puts > ATM > calls, and short expiry has more skew.
    const auto row = [&](double strike, std::chrono::year_month_day exp,
                         double put_iv, double call_iv,
                         const std::string& sym_prefix)
    {
        std::vector<OptionMarketSnapshot> pair;
        pair.push_back(synth_snap(market, strike, exp, OptionType::Put,
                                  put_iv, sym_prefix + "P"));
        pair.push_back(synth_snap(market, strike, exp, OptionType::Call,
                                  call_iv, sym_prefix + "C"));
        return pair;
    };

    std::vector<OptionMarketSnapshot> snaps;
    for (auto s : row( 80.0, exp1, 0.35, 0.30, "Sx80x")) snaps.push_back(std::move(s));
    for (auto s : row( 90.0, exp1, 0.28, 0.24, "Sx90x")) snaps.push_back(std::move(s));
    for (auto s : row(100.0, exp1, 0.22, 0.22, "Sx100x")) snaps.push_back(std::move(s));
    for (auto s : row(110.0, exp1, 0.22, 0.20, "Sx110x")) snaps.push_back(std::move(s));
    for (auto s : row(120.0, exp1, 0.24, 0.20, "Sx120x")) snaps.push_back(std::move(s));

    for (auto s : row( 80.0, exp2, 0.30, 0.27, "Lx80x")) snaps.push_back(std::move(s));
    for (auto s : row( 90.0, exp2, 0.26, 0.24, "Lx90x")) snaps.push_back(std::move(s));
    for (auto s : row(100.0, exp2, 0.24, 0.24, "Lx100x")) snaps.push_back(std::move(s));
    for (auto s : row(110.0, exp2, 0.23, 0.22, "Lx110x")) snaps.push_back(std::move(s));
    for (auto s : row(120.0, exp2, 0.24, 0.22, "Lx120x")) snaps.push_back(std::move(s));

    OptionChain chain(canonical_underlying(), market, std::move(snaps));
    return OptionChainCalibrator{}.calibrate(chain);
}

}  // namespace

// =============================================================================
// SMILE GROUPING & MONEYNESS
// =============================================================================

TEST(VolatilityAnalyticsSmileTest, GroupsByExpirationAndSortsByStrike) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto smiles = build_smiles(report, market);

    ASSERT_EQ(smiles.size(), 2U);  // two expirations
    EXPECT_LT(smiles[0].expiration, smiles[1].expiration);

    for (const auto& s : smiles) {
        // 5 strikes x 2 types = 10 rows per expiration.
        EXPECT_EQ(s.size(), 10U);
        // Strikes ascending.
        for (std::size_t i = 1; i < s.strikes.size(); ++i) {
            EXPECT_LE(s.strikes[i - 1], s.strikes[i]);
        }
    }
}

TEST(VolatilityAnalyticsSmileTest, IgnoresSkippedAndFailedContracts) {
    const auto market = canonical_market();
    // Chain with one good contract and one that will be skipped
    // (crossed market -> skipped by calibrator).
    OptionMarketSnapshot good = synth_snap(market, 100.0, ymd(2026,7,1),
                                           OptionType::Call, 0.20, "OK");
    OptionMarketSnapshot bad{};
    bad.option = good.option;
    bad.option.strike = 110.0;
    bad.contract_symbol = "BAD";
    bad.quote.bid = 5.0; bad.quote.ask = 4.0;  // crossed

    OptionChain chain(canonical_underlying(), market, {good, bad});
    const auto report = OptionChainCalibrator{}.calibrate(chain);
    const auto smiles = build_smiles(report, market);

    ASSERT_EQ(smiles.size(), 1U);
    EXPECT_EQ(smiles[0].size(), 1U);  // only the good one
    EXPECT_DOUBLE_EQ(smiles[0].strikes[0], 100.0);
}

TEST(VolatilityAnalyticsSmileTest, LogSimpleMoneynessIsSymmetricAroundAtm) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto smiles = build_smiles(report, market, Moneyness::LogSimple);

    ASSERT_FALSE(smiles.empty());
    // In our chain, K=80 and K=125 aren't symmetric — but K=80 and K=125
    // are set up so that ln(80/100) = -ln(125/100) is not present. Test
    // with the K=100 ATM strikes: moneyness must be exactly 0.
    for (const auto& s : smiles) {
        for (std::size_t i = 0; i < s.strikes.size(); ++i) {
            if (s.strikes[i] == 100.0) {
                EXPECT_NEAR(s.moneyness[i], 0.0, 1e-14) << "expected atm=0";
            }
        }
    }
}

TEST(VolatilityAnalyticsSmileTest, SimpleMoneynessIsRatio) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto smiles = build_smiles(report, market, Moneyness::Simple);
    for (const auto& s : smiles) {
        for (std::size_t i = 0; i < s.strikes.size(); ++i) {
            EXPECT_DOUBLE_EQ(s.moneyness[i], s.strikes[i] / market.spot);
        }
    }
}

TEST(VolatilityAnalyticsSmileTest, LogForwardShiftsByCostOfCarry) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto simple = build_smiles(report, market, Moneyness::LogSimple);
    const auto fwd    = build_smiles(report, market, Moneyness::LogForward);

    // LogForward = LogSimple - (r - q) * T. Verify per-point.
    ASSERT_EQ(simple.size(), fwd.size());
    for (std::size_t k = 0; k < simple.size(); ++k) {
        const double T = simple[k].time_to_expiry;
        const double shift = (market.risk_free_rate - market.dividend_yield) * T;
        for (std::size_t i = 0; i < simple[k].size(); ++i) {
            EXPECT_NEAR(fwd[k].moneyness[i],
                        simple[k].moneyness[i] - shift,
                        1e-12);
        }
    }
}

TEST(VolatilityAnalyticsSmileTest, RecoveredIVsMatchInjectedIVs) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto smiles = build_smiles(report, market);

    // For each expiration, verify each (strike, type) has IV close to
    // what we injected. The injected map:
    struct Key { double K; OptionType t; std::chrono::year_month_day exp; };
    // Not worth building an actual map here — just spot-check the ATM
    // K=100 call which we injected at 0.22 for exp1 and 0.24 for exp2.
    const auto find_iv = [&](std::size_t s_idx, double K, OptionType type) -> double {
        const auto& s = smiles[s_idx];
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (s.strikes[i] == K && s.types[i] == type) {
                return s.implied_volatility[i];
            }
        }
        return -1.0;
    };
    EXPECT_NEAR(find_iv(0, 100.0, OptionType::Call), 0.22, 1e-8);
    EXPECT_NEAR(find_iv(1, 100.0, OptionType::Call), 0.24, 1e-8);
    EXPECT_NEAR(find_iv(0,  80.0, OptionType::Put),  0.35, 1e-8);
    EXPECT_NEAR(find_iv(1, 120.0, OptionType::Call), 0.22, 1e-8);
}

// =============================================================================
// TERM STRUCTURE
// =============================================================================

TEST(VolatilityAnalyticsTermStructureTest, InterpolatesAtmIvAtSpot) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto term = build_term_structure(report, market);

    ASSERT_EQ(term.size(), 2U);

    // K=100 ATM strikes are calibrated at 0.22 (exp1) and 0.24 (exp2).
    // Since we have a K=100 strike exactly at spot, ATM IV should equal
    // the OTM IV at K=100. For K=100 with spot=100, call is "OTM" by
    // our >= convention, so we use the call IV (which was 0.22, 0.24).
    EXPECT_NEAR(term.atm_iv[0], 0.22, 1e-8);
    EXPECT_NEAR(term.atm_iv[1], 0.24, 1e-8);

    // Maturities ascending.
    EXPECT_LT(term.maturities[0], term.maturities[1]);
}

TEST(VolatilityAnalyticsTermStructureTest, NanWhenSpotOutsideStrikeRange) {
    // A chain with only strikes above spot -> can't interpolate down to spot.
    const auto market = canonical_market();  // spot = 100
    const auto exp = ymd(2026, 4, 1);

    std::vector<OptionMarketSnapshot> snaps;
    snaps.push_back(synth_snap(market, 110.0, exp, OptionType::Call, 0.20, "A"));
    snaps.push_back(synth_snap(market, 120.0, exp, OptionType::Call, 0.22, "B"));

    OptionChain chain(canonical_underlying(), market, std::move(snaps));
    const auto report = OptionChainCalibrator{}.calibrate(chain);
    const auto term = build_term_structure(report, market);

    ASSERT_EQ(term.size(), 1U);
    EXPECT_TRUE(std::isnan(term.atm_iv[0]));
}

// =============================================================================
// SURFACE
// =============================================================================

TEST(VolatilityAnalyticsSurfaceTest, HasCorrectDimensions) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto surface = build_surface(report, market);

    EXPECT_EQ(surface.rows(), 2U);       // 2 expirations
    EXPECT_EQ(surface.cols(), 5U);       // 5 unique strikes
    ASSERT_EQ(surface.implied_vols.size(), 2U);
    EXPECT_EQ(surface.implied_vols[0].size(), 5U);
}

TEST(VolatilityAnalyticsSurfaceTest, StrikesAreSortedUnion) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto surface = build_surface(report, market);

    const std::vector<double> expected{80.0, 90.0, 100.0, 110.0, 120.0};
    ASSERT_EQ(surface.strikes.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(surface.strikes[i], expected[i]);
    }
}

TEST(VolatilityAnalyticsSurfaceTest, MissingCellsAreNan) {
    // Chain with strikes {80, 100} at exp1 and {100, 120} at exp2.
    // Surface should be 2x3 with two NaN corners.
    const auto market = canonical_market();
    const auto exp1 = ymd(2026, 4, 1);
    const auto exp2 = ymd(2026, 10, 1);

    std::vector<OptionMarketSnapshot> snaps;
    snaps.push_back(synth_snap(market,  80.0, exp1, OptionType::Put,  0.30, "A"));
    snaps.push_back(synth_snap(market, 100.0, exp1, OptionType::Call, 0.22, "B"));
    snaps.push_back(synth_snap(market, 100.0, exp2, OptionType::Call, 0.24, "C"));
    snaps.push_back(synth_snap(market, 120.0, exp2, OptionType::Call, 0.22, "D"));

    OptionChain chain(canonical_underlying(), market, std::move(snaps));
    const auto report = OptionChainCalibrator{}.calibrate(chain);
    const auto surface = build_surface(report, market);

    ASSERT_EQ(surface.rows(), 2U);
    ASSERT_EQ(surface.cols(), 3U);  // {80, 100, 120}

    // Row 0 (exp1): 80=put(0.30), 100=call(0.22), 120=NaN
    EXPECT_NEAR(surface.implied_vols[0][0], 0.30, 1e-8);
    EXPECT_NEAR(surface.implied_vols[0][1], 0.22, 1e-8);
    EXPECT_TRUE(std::isnan(surface.implied_vols[0][2]));
    // Row 1 (exp2): 80=NaN, 100=call(0.24), 120=call(0.22)
    EXPECT_TRUE(std::isnan(surface.implied_vols[1][0]));
    EXPECT_NEAR(surface.implied_vols[1][1], 0.24, 1e-8);
    EXPECT_NEAR(surface.implied_vols[1][2], 0.22, 1e-8);
}

TEST(VolatilityAnalyticsSurfaceTest, PrefersOtmForBelowSpotStrikes) {
    // At K=80 (below spot=100), both call and put are calibrated.
    // Surface should reflect the put IV (OTM by our convention).
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto surface = build_surface(report, market);

    // Row 0, col 0 -> K=80 at exp1. Injected put=0.35, call=0.30.
    EXPECT_NEAR(surface.implied_vols[0][0], 0.35, 1e-8);
}

TEST(VolatilityAnalyticsSurfaceTest, PrefersOtmForAboveSpotStrikes) {
    // At K=120 (above spot=100), both call and put are calibrated.
    // Surface should reflect the call IV (OTM by our convention).
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto surface = build_surface(report, market);

    // Row 0, col 4 -> K=120 at exp1. Injected put=0.24, call=0.20.
    EXPECT_NEAR(surface.implied_vols[0][4], 0.20, 1e-8);
}

// =============================================================================
// SKEW METRICS
// =============================================================================

TEST(VolatilityAnalyticsSkewTest, RecoversAtmIvExactly) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto smiles = build_smiles(report, market);
    const auto skews = compute_skew_metrics(smiles, market);
    ASSERT_EQ(skews.size(), 2U);
    ASSERT_TRUE(skews[0].atm_iv.has_value());
    // K=100 call is OTM (>= spot), IV = 0.22 for exp1.
    EXPECT_NEAR(*skews[0].atm_iv, 0.22, 1e-8);
    EXPECT_NEAR(*skews[1].atm_iv, 0.24, 1e-8);
}

TEST(VolatilityAnalyticsSkewTest, RiskReversalIsCallMinusPut) {
    // Build a chain with enough deep-OTM points so 25-delta interpolation
    // has a bracket on both sides. Use a wide range.
    const auto market = canonical_market();
    const auto exp = ymd(2026, 7, 1);

    std::vector<OptionMarketSnapshot> snaps;
    // Deep OTM calls (small delta), through ATM, then deep OTM puts.
    for (double K : {70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0}) {
        // Slight per-strike skew so RR is nonzero.
        const double vol = 0.20 + 0.05 * std::abs(std::log(K / 100.0));
        snaps.push_back(synth_snap(market, K, exp, OptionType::Call, vol,
                                   "C" + std::to_string(int(K))));
        snaps.push_back(synth_snap(market, K, exp, OptionType::Put, vol + 0.02,
                                   "P" + std::to_string(int(K))));
    }

    OptionChain chain(canonical_underlying(), market, std::move(snaps));
    const auto report = OptionChainCalibrator{}.calibrate(chain);
    const auto smiles = build_smiles(report, market);
    const auto skews = compute_skew_metrics(smiles, market);

    ASSERT_EQ(skews.size(), 1U);
    ASSERT_TRUE(skews[0].call_25delta_iv.has_value());
    ASSERT_TRUE(skews[0].put_25delta_iv.has_value());
    ASSERT_TRUE(skews[0].risk_reversal.has_value());

    const double expected_rr =
        *skews[0].call_25delta_iv - *skews[0].put_25delta_iv;
    EXPECT_NEAR(*skews[0].risk_reversal, expected_rr, 1e-12);
    // Puts injected 0.02 higher than calls at every strike, so RR < 0.
    EXPECT_LT(*skews[0].risk_reversal, 0.0);
}

TEST(VolatilityAnalyticsSkewTest, ButterflyIsWingsMinusAtm) {
    const auto market = canonical_market();
    const auto exp = ymd(2026, 7, 1);
    std::vector<OptionMarketSnapshot> snaps;
    // Convex smile: ATM = 0.18, wings up to 0.30.
    for (double K : {70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0}) {
        const double vol = 0.18 + 0.15 * std::pow(std::log(K / 100.0), 2);
        snaps.push_back(synth_snap(market, K, exp, OptionType::Call, vol,
                                   "C" + std::to_string(int(K))));
        snaps.push_back(synth_snap(market, K, exp, OptionType::Put, vol,
                                   "P" + std::to_string(int(K))));
    }
    OptionChain chain(canonical_underlying(), market, std::move(snaps));
    const auto report = OptionChainCalibrator{}.calibrate(chain);
    const auto smiles = build_smiles(report, market);
    const auto skews = compute_skew_metrics(smiles, market);

    ASSERT_EQ(skews.size(), 1U);
    ASSERT_TRUE(skews[0].butterfly.has_value());
    // With a convex smile the butterfly must be strictly positive.
    EXPECT_GT(*skews[0].butterfly, 0.0);
}

TEST(VolatilityAnalyticsSkewTest, ReturnsNullOptWhenChainTooNarrow) {
    // Only ATM and one wing -> 25-delta interpolation impossible on one side.
    const auto market = canonical_market();
    const auto exp = ymd(2026, 7, 1);
    std::vector<OptionMarketSnapshot> snaps;
    snaps.push_back(synth_snap(market, 100.0, exp, OptionType::Call, 0.20, "A"));
    snaps.push_back(synth_snap(market, 100.0, exp, OptionType::Put,  0.20, "B"));
    OptionChain chain(canonical_underlying(), market, std::move(snaps));
    const auto report = OptionChainCalibrator{}.calibrate(chain);
    const auto smiles = build_smiles(report, market);
    const auto skews = compute_skew_metrics(smiles, market);

    ASSERT_EQ(skews.size(), 1U);
    EXPECT_FALSE(skews[0].call_25delta_iv.has_value());
    EXPECT_FALSE(skews[0].put_25delta_iv.has_value());
    EXPECT_FALSE(skews[0].risk_reversal.has_value());
    EXPECT_FALSE(skews[0].butterfly.has_value());
}

// =============================================================================
// STATISTICS
// =============================================================================

TEST(VolatilityAnalyticsStatsTest, KnownVectorProducesExpectedStatistics) {
    // Hand-computed reference values for {1, 2, 3, 4, 5}:
    //   n=5, min=1, max=5, mean=3, median=3, stddev=sqrt(2)~1.4142
    //   p10 -> rank=0.4 -> 1 + 0.4*(2-1) = 1.4
    //   p25 -> rank=1.0 -> exact 2
    //   p75 -> rank=3.0 -> exact 4
    //   p90 -> rank=3.6 -> 4 + 0.6*(5-4) = 4.6
    const std::vector<double> v{1.0, 2.0, 3.0, 4.0, 5.0};
    const auto s = compute_statistics(std::span<const double>{v});

    EXPECT_EQ(s.count, 5U);
    EXPECT_DOUBLE_EQ(s.min, 1.0);
    EXPECT_DOUBLE_EQ(s.max, 5.0);
    EXPECT_DOUBLE_EQ(s.mean, 3.0);
    EXPECT_DOUBLE_EQ(s.median, 3.0);
    EXPECT_NEAR(s.stddev, std::sqrt(2.0), 1e-12);
    EXPECT_DOUBLE_EQ(s.p10, 1.4);
    EXPECT_DOUBLE_EQ(s.p25, 2.0);
    EXPECT_DOUBLE_EQ(s.p75, 4.0);
    EXPECT_DOUBLE_EQ(s.p90, 4.6);
}

TEST(VolatilityAnalyticsStatsTest, DropsNonFiniteValues) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> v{1.0, nan, 2.0, nan, 3.0};
    const auto s = compute_statistics(std::span<const double>{v});
    EXPECT_EQ(s.count, 3U);
    EXPECT_DOUBLE_EQ(s.mean, 2.0);
}

TEST(VolatilityAnalyticsStatsTest, EmptyProducesAllZeros) {
    const std::vector<double> v{};
    const auto s = compute_statistics(std::span<const double>{v});
    EXPECT_EQ(s.count, 0U);
    EXPECT_DOUBLE_EQ(s.mean, 0.0);
    EXPECT_DOUBLE_EQ(s.stddev, 0.0);
}

TEST(VolatilityAnalyticsStatsTest, OverEntireReportUsesCalibratedOnly) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto s = compute_statistics(report);
    // 2 exps * 5 strikes * 2 types = 20 calibrated contracts.
    EXPECT_EQ(s.count, 20U);
    // Every injected IV is in [0.20, 0.35].
    EXPECT_GE(s.min, 0.20 - 1e-8);
    EXPECT_LE(s.max, 0.35 + 1e-8);
}

TEST(VolatilityAnalyticsStatsTest, PerExpirationBucketsSum) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto by_exp = compute_statistics_by_expiration(report);

    ASSERT_EQ(by_exp.size(), 2U);
    // 10 contracts per expiration.
    EXPECT_EQ(by_exp[0].second.count, 10U);
    EXPECT_EQ(by_exp[1].second.count, 10U);
}

TEST(VolatilityAnalyticsStatsTest, PerTypeSeparatesCallsAndPuts) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto by_type = compute_statistics_by_type(report);

    EXPECT_EQ(by_type.calls.count, 10U);
    EXPECT_EQ(by_type.puts.count,  10U);
    // In this chain puts are systematically higher-vol than calls.
    EXPECT_GE(by_type.puts.mean, by_type.calls.mean);
}

// =============================================================================
// CSV EXPORT
// =============================================================================

TEST(VolatilityAnalyticsCsvTest, SmileHeaderAndRowCount) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto smiles = build_smiles(report, market);

    std::ostringstream oss;
    write_csv(std::span<const VolatilitySmile>(smiles), oss);
    const auto s = oss.str();

    EXPECT_NE(s.find(
        "expiration,time_to_expiry,strike,option_type,"
        "moneyness_convention,moneyness,implied_volatility\n"), std::string::npos);
    // Header + 20 data rows (2 exps * 10 contracts).
    const auto lines = std::count(s.begin(), s.end(), '\n');
    EXPECT_EQ(static_cast<std::size_t>(lines), 1 + 20U);
}

TEST(VolatilityAnalyticsCsvTest, TermStructureHeaderAndRows) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto term = build_term_structure(report, market);

    std::ostringstream oss;
    write_csv(term, oss);
    const auto s = oss.str();
    EXPECT_NE(s.find("expiration,time_to_expiry,atm_iv\n"), std::string::npos);
    const auto lines = std::count(s.begin(), s.end(), '\n');
    EXPECT_EQ(static_cast<std::size_t>(lines), 1 + term.size());
}

TEST(VolatilityAnalyticsCsvTest, SurfaceEmitsLongFormatWithNanBlanks) {
    // 2 exp * 3 strikes = 6 rows; 2 are NaN.
    const auto market = canonical_market();
    const auto exp1 = ymd(2026, 4, 1);
    const auto exp2 = ymd(2026, 10, 1);
    std::vector<OptionMarketSnapshot> snaps;
    snaps.push_back(synth_snap(market,  80.0, exp1, OptionType::Put,  0.30, "A"));
    snaps.push_back(synth_snap(market, 100.0, exp1, OptionType::Call, 0.22, "B"));
    snaps.push_back(synth_snap(market, 100.0, exp2, OptionType::Call, 0.24, "C"));
    snaps.push_back(synth_snap(market, 120.0, exp2, OptionType::Call, 0.22, "D"));

    OptionChain chain(canonical_underlying(), market, std::move(snaps));
    const auto report = OptionChainCalibrator{}.calibrate(chain);
    const auto surface = build_surface(report, market);

    std::ostringstream oss;
    write_csv(surface, oss);
    const auto s = oss.str();

    EXPECT_NE(s.find("expiration,time_to_expiry,strike,implied_volatility\n"),
              std::string::npos);
    const auto lines = std::count(s.begin(), s.end(), '\n');
    EXPECT_EQ(static_cast<std::size_t>(lines), 1 + 6U);
    // A NaN IV renders as an empty last field, producing a `,\n` boundary
    // at end-of-row. Count those to verify pandas will read the expected
    // number of NaNs.
    std::size_t empty_last_fields = 0;
    for (std::size_t i = 1; i < s.size(); ++i) {
        if (s[i] == '\n' && s[i - 1] == ',') ++empty_last_fields;
    }
    EXPECT_EQ(empty_last_fields, 2U);
}

TEST(VolatilityAnalyticsCsvTest, SkewMetricsHeader) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto smiles = build_smiles(report, market);
    const auto skews = compute_skew_metrics(smiles, market);

    std::ostringstream oss;
    write_csv(std::span<const SkewMetrics>(skews), oss);
    const auto s = oss.str();
    EXPECT_NE(s.find(
        "expiration,time_to_expiry,atm_iv,call_25delta_iv,put_25delta_iv,"
        "risk_reversal,butterfly\n"), std::string::npos);
}

TEST(VolatilityAnalyticsCsvTest, FilePathOverloadsWriteToDisk) {
    const auto market = canonical_market();
    const auto report = build_stylized_report(market);
    const auto smiles = build_smiles(report, market);
    const auto term   = build_term_structure(report, market);
    const auto surface = build_surface(report, market);
    const auto skews = compute_skew_metrics(smiles, market);

    const auto dir = std::filesystem::temp_directory_path();
    const auto smile_path  = dir / "ore_test_smiles.csv";
    const auto term_path   = dir / "ore_test_term.csv";
    const auto surf_path   = dir / "ore_test_surface.csv";
    const auto skew_path   = dir / "ore_test_skew.csv";

    write_csv(std::span<const VolatilitySmile>(smiles), smile_path);
    write_csv(term,    term_path);
    write_csv(surface, surf_path);
    write_csv(std::span<const SkewMetrics>(skews), skew_path);

    for (const auto& p : {smile_path, term_path, surf_path, skew_path}) {
        EXPECT_TRUE(std::filesystem::exists(p));
        EXPECT_GT(std::filesystem::file_size(p), 0U);
        std::filesystem::remove(p);
    }
}

// =============================================================================
// MISSING DATA
// =============================================================================

TEST(VolatilityAnalyticsMissingDataTest, EmptyReportProducesEmptyOutputs) {
    const auto market = canonical_market();
    CalibrationReport empty{};
    EXPECT_TRUE(build_smiles(empty, market).empty());

    const auto term = build_term_structure(empty, market);
    EXPECT_TRUE(term.empty());

    const auto surf = build_surface(empty, market);
    EXPECT_TRUE(surf.empty());
    EXPECT_EQ(surf.strikes.size(), 0U);

    const auto st = compute_statistics(empty);
    EXPECT_EQ(st.count, 0U);
}
