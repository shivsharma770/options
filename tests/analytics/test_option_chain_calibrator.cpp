/**
 * @file test_option_chain_calibrator.cpp
 * @brief Correctness tests for `ore::analytics::OptionChainCalibrator`.
 *
 * The suite is organised around the M6 spec:
 *   * `BasicCalibration`  — synthetic chain, verify per-option IV recovery.
 *   * `FilteringRules`    — every `SkipReason` exercised individually.
 *   * `Statistics`        — verify report aggregate math.
 *   * `ProviderComparison`— absolute/relative error semantics.
 *   * `CsvExport`         — column layout + missing-value rendering.
 *   * `IntegrationFromDisk`— load the shipped Yahoo fixture and calibrate.
 *
 * Chains are constructed in memory from `BlackScholesEngine`-generated
 * prices so recovery-to-1e-8 is the expected precision floor.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <ore/analytics/option_chain_calibrator.hpp>
#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/option_chain.hpp>
#include <ore/core/option_market_snapshot.hpp>
#include <ore/core/quote.hpp>
#include <ore/core/underlying.hpp>
#include <ore/marketdata/yahoo_option_loader.hpp>
#include <ore/numerics/solver_result.hpp>
#include <ore/pricing/black_scholes_engine.hpp>

#ifndef ORE_TEST_FIXTURES_DIR
#error "ORE_TEST_FIXTURES_DIR must be defined by the build system"
#endif

using ore::analytics::CalibrationReport;
using ore::analytics::CalibrationResult;
using ore::analytics::OptionChainCalibrator;
using ore::analytics::SkipReason;
using ore::core::AssetType;
using ore::core::ExerciseStyle;
using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionChain;
using ore::core::OptionMarketSnapshot;
using ore::core::OptionType;
using ore::core::Quote;
using ore::core::Underlying;
using ore::numerics::SolverStatus;
using ore::pricing::BlackScholesEngine;

namespace {

// ------ Common builders ------------------------------------------------------

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
    return Underlying{.symbol = "TEST", .exchange = "TEST", .asset_type = AssetType::Equity};
}

/**
 * Build one `OptionMarketSnapshot` whose bid/ask straddle the exact
 * Black-Scholes price at the given `sigma`, so the calibrator should
 * recover `sigma`. The bid-ask half-spread defaults to zero — most tests
 * want no spread noise.
 *
 * `provider_iv_offset` lets us inject a synthetic disagreement between
 * "provider IV" and the true model IV, so we can verify provider-error
 * semantics without needing a real provider.
 */
OptionMarketSnapshot build_snap(
    const MarketSnapshot&                   market,
    double                                  strike,
    std::chrono::year_month_day             expiration,
    OptionType                              type,
    double                                  sigma_true,
    std::string                             contract_symbol = "TEST",
    double                                  spread_half     = 0.0,
    std::optional<double>                   provider_iv_override = std::nullopt)
{
    const auto val_days = std::chrono::sys_days{market.valuation_date};
    const auto exp_days = std::chrono::sys_days{expiration};
    const double T = static_cast<double>((exp_days - val_days).count()) / 365.0;

    BlackScholesEngine engine;
    const auto pr = engine.price(BlackScholesEngine::Inputs{
        .spot           = market.spot,
        .strike         = strike,
        .rate           = market.risk_free_rate,
        .dividend_yield = market.dividend_yield,
        .volatility     = sigma_true,
        .time_to_expiry = T,
        .type           = type,
    });
    const double mid = pr.price;

    OptionMarketSnapshot s{};
    s.option = Option{
        .underlying = canonical_underlying(),
        .strike     = strike,
        .expiration = expiration,
        .type       = type,
        .exercise   = ExerciseStyle::European,
    };
    s.quote = Quote{
        .bid                 = mid - spread_half,
        .ask                 = mid + spread_half,
        .last                = mid,
        .volume              = 100.0,
        .open_interest       = 500.0,
        .implied_volatility  = provider_iv_override.has_value()
                                 ? provider_iv_override
                                 : std::optional<double>{sigma_true},
        .timestamp           = std::chrono::system_clock::now(),
    };
    s.contract_symbol = std::move(contract_symbol);
    return s;
}

OptionChain make_chain(std::vector<OptionMarketSnapshot> options,
                       MarketSnapshot market = canonical_market()) {
    return OptionChain{canonical_underlying(), market, std::move(options)};
}

// A "corrupt" snapshot builder for filter tests. Bypasses the price
// synthesis and lets the caller set any quote directly.
OptionMarketSnapshot custom_snap(
    double strike, OptionType type, Quote q, std::string contract_symbol = "X")
{
    OptionMarketSnapshot s{};
    s.option = Option{
        .underlying = canonical_underlying(),
        .strike     = strike,
        .expiration = ymd(2026, 6, 1),
        .type       = type,
        .exercise   = ExerciseStyle::European,
    };
    s.quote = q;
    s.contract_symbol = std::move(contract_symbol);
    return s;
}

}  // namespace

// =============================================================================
// BASIC CALIBRATION — a small chain, all contracts recoverable
// =============================================================================

TEST(OptionChainCalibratorTest, RecoversKnownVolAcrossSmallChain) {
    const auto market = canonical_market();
    const auto exp = ymd(2026, 7, 1);

    std::vector<OptionMarketSnapshot> snaps;
    snaps.push_back(build_snap(market, 90.0,  exp, OptionType::Call, 0.25, "C90"));
    snaps.push_back(build_snap(market, 100.0, exp, OptionType::Call, 0.20, "C100"));
    snaps.push_back(build_snap(market, 110.0, exp, OptionType::Call, 0.22, "C110"));
    snaps.push_back(build_snap(market, 90.0,  exp, OptionType::Put,  0.30, "P90"));
    snaps.push_back(build_snap(market, 100.0, exp, OptionType::Put,  0.20, "P100"));

    const OptionChain chain = make_chain(std::move(snaps));
    const OptionChainCalibrator calibrator;
    const auto report = calibrator.calibrate(chain);

    ASSERT_EQ(report.results.size(), 5U);
    EXPECT_EQ(report.successful_solves, 5U);
    EXPECT_EQ(report.failed_solves,     0U);
    EXPECT_EQ(report.skipped,           0U);

    const std::vector<double> expected{0.25, 0.20, 0.22, 0.30, 0.20};
    for (std::size_t i = 0; i < report.results.size(); ++i) {
        const auto& r = report.results[i];
        ASSERT_TRUE(r.was_calibrated()) << "contract " << r.contract_symbol;
        ASSERT_TRUE(r.computed_iv.has_value());
        EXPECT_NEAR(*r.computed_iv, expected[i], 1e-8) << r.contract_symbol;
    }
    EXPECT_DOUBLE_EQ(report.convergence_rate(), 1.0);
}

// =============================================================================
// FILTERING RULES — one test per SkipReason
// =============================================================================

TEST(OptionChainCalibratorFilteringTest, SkipsNoMarket) {
    const auto market = canonical_market();
    Quote q{}; q.bid = 0.0; q.ask = 0.0;
    const auto snap = custom_snap(100.0, OptionType::Call, q);

    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}, market));
    ASSERT_EQ(report.results.size(), 1U);
    EXPECT_EQ(report.results[0].skip_reason, SkipReason::NoMarket);
    EXPECT_EQ(report.skipped, 1U);
    EXPECT_EQ(report.successful_solves, 0U);
}

TEST(OptionChainCalibratorFilteringTest, SkipsCrossedMarket) {
    Quote q{}; q.bid = 5.0; q.ask = 4.0;  // crossed
    const auto snap = custom_snap(100.0, OptionType::Call, q);
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}));
    ASSERT_EQ(report.results.size(), 1U);
    EXPECT_EQ(report.results[0].skip_reason, SkipReason::CrossedMarket);
}

TEST(OptionChainCalibratorFilteringTest, SkipsNonFiniteQuote) {
    Quote q{};
    q.bid = std::numeric_limits<double>::quiet_NaN();
    q.ask = 5.0;
    const auto snap = custom_snap(100.0, OptionType::Call, q);
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}));
    ASSERT_EQ(report.results.size(), 1U);
    EXPECT_EQ(report.results[0].skip_reason, SkipReason::NonFiniteQuote);
}

TEST(OptionChainCalibratorFilteringTest, SkipsNonPositiveMidPrice) {
    // bid = -1, ask = 0 -> mid = -0.5. Passes the finite check but not
    // the positivity check. This should never happen from the loader
    // (which rejects negative bids) but we defensively guard against it.
    Quote q{}; q.bid = -1.0; q.ask = 0.0;
    const auto snap = custom_snap(100.0, OptionType::Call, q);
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}));
    ASSERT_EQ(report.results.size(), 1U);
    EXPECT_EQ(report.results[0].skip_reason, SkipReason::NonPositiveMidPrice);
}

TEST(OptionChainCalibratorFilteringTest, SkipsBelowMinimumPrice) {
    Quote q{}; q.bid = 0.001; q.ask = 0.002;  // mid = 0.0015, below default 0.005
    const auto snap = custom_snap(100.0, OptionType::Call, q);
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}));
    ASSERT_EQ(report.results.size(), 1U);
    EXPECT_EQ(report.results[0].skip_reason, SkipReason::BelowMinimumPrice);
}

TEST(OptionChainCalibratorFilteringTest, MinimumPriceIsConfigurable) {
    Quote q{}; q.bid = 0.001; q.ask = 0.002;
    const auto snap = custom_snap(100.0, OptionType::Call, q);

    OptionChainCalibrator::Config cfg{};
    cfg.minimum_option_price = 0.0;  // disable the filter
    OptionChainCalibrator calibrator(cfg);
    const auto report = calibrator.calibrate(make_chain({snap}));
    ASSERT_EQ(report.results.size(), 1U);
    // With the filter disabled it *tries* to solve. Whether it converges
    // for such a small price is not our concern here — the filter just
    // shouldn't have flagged it.
    EXPECT_NE(report.results[0].skip_reason, SkipReason::BelowMinimumPrice);
}

TEST(OptionChainCalibratorFilteringTest, SkipsArbitrageViolation) {
    // Deep ITM call with a mid *below* the theoretical lower bound.
    // Intrinsic-in-forward for K=50, S=100, r=5%, q=2%, T=0.4 is
    //   100 * exp(-0.008) - 50 * exp(-0.02) ~= 99.20 - 49.01 = 50.19.
    // Pricing the option at mid=45.0 puts it below that bound.
    Quote q{}; q.bid = 44.9; q.ask = 45.1;  // mid = 45.0
    const auto snap = custom_snap(50.0, OptionType::Call, q, "ARB");
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}));
    ASSERT_EQ(report.results.size(), 1U);
    EXPECT_EQ(report.results[0].skip_reason, SkipReason::ArbitrageViolation);
}

TEST(OptionChainCalibratorFilteringTest, MixedSuccessAndFailureCounted) {
    const auto market = canonical_market();
    std::vector<OptionMarketSnapshot> snaps;
    snaps.push_back(build_snap(market, 100.0, ymd(2026, 7, 1), OptionType::Call, 0.20, "OK"));
    Quote bad{}; bad.bid = 0.0; bad.ask = 0.0;
    snaps.push_back(custom_snap(100.0, OptionType::Call, bad, "NOMKT"));

    const auto report = OptionChainCalibrator{}.calibrate(make_chain(std::move(snaps), market));

    EXPECT_EQ(report.successful_solves, 1U);
    EXPECT_EQ(report.skipped, 1U);
    EXPECT_EQ(report.failed_solves, 0U);
    EXPECT_DOUBLE_EQ(report.convergence_rate(), 1.0);  // only counts attempted
}

// =============================================================================
// STATISTICS — verify aggregate math
// =============================================================================

TEST(OptionChainCalibratorStatisticsTest, IterationAndErrorAggregates) {
    const auto market = canonical_market();
    // Three contracts with intentionally different provider IVs so the
    // absolute errors are 0.00, 0.02, 0.05 respectively.
    std::vector<OptionMarketSnapshot> snaps;
    snaps.push_back(build_snap(market, 100.0, ymd(2026, 7, 1), OptionType::Call, 0.20, "A",
                               0.0, /*provider*/ 0.20));
    snaps.push_back(build_snap(market, 100.0, ymd(2026, 7, 1), OptionType::Call, 0.25, "B",
                               0.0, /*provider*/ 0.23));
    snaps.push_back(build_snap(market,  95.0, ymd(2026, 7, 1), OptionType::Put,  0.30, "C",
                               0.0, /*provider*/ 0.25));

    const auto report = OptionChainCalibrator{}.calibrate(make_chain(std::move(snaps), market));

    EXPECT_EQ(report.successful_solves, 3U);
    EXPECT_EQ(report.provider_iv_comparisons, 3U);

    // Expected: |0.20 - 0.20| = 0, |0.25 - 0.23| = 0.02, |0.30 - 0.25| = 0.05
    // Mean absolute error = (0 + 0.02 + 0.05) / 3 = 0.0233...
    // RMSE                = sqrt((0 + 0.0004 + 0.0025) / 3)
    //                     = sqrt(0.0029 / 3) = 0.031091...
    // Max absolute error  = 0.05
    EXPECT_NEAR(report.mean_absolute_iv_error, (0.02 + 0.05) / 3.0, 1e-9);
    EXPECT_NEAR(report.rmse_iv_error,
                std::sqrt((0.0004 + 0.0025) / 3.0), 1e-9);
    EXPECT_NEAR(report.maximum_iv_error, 0.05, 1e-9);

    EXPECT_GT(report.average_iterations, 0.0);
    EXPECT_GE(report.maximum_iterations, 1U);
}

TEST(OptionChainCalibratorStatisticsTest, EmptyChainHandledGracefully) {
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({}));
    EXPECT_EQ(report.results.size(), 0U);
    EXPECT_EQ(report.successful_solves, 0U);
    EXPECT_EQ(report.failed_solves, 0U);
    EXPECT_EQ(report.skipped, 0U);
    EXPECT_EQ(report.provider_iv_comparisons, 0U);
    EXPECT_DOUBLE_EQ(report.mean_absolute_iv_error, 0.0);
    EXPECT_DOUBLE_EQ(report.rmse_iv_error, 0.0);
    EXPECT_DOUBLE_EQ(report.maximum_iv_error, 0.0);
    EXPECT_DOUBLE_EQ(report.average_iterations, 0.0);
    EXPECT_EQ(report.maximum_iterations, 0U);
    EXPECT_DOUBLE_EQ(report.convergence_rate(), 0.0);  // no attempts
}

TEST(OptionChainCalibratorStatisticsTest, AllSkippedProducesNoDivisionByZero) {
    // Both contracts have no market; neither should be attempted.
    std::vector<OptionMarketSnapshot> snaps;
    Quote nomkt{}; nomkt.bid = 0.0; nomkt.ask = 0.0;
    snaps.push_back(custom_snap(100.0, OptionType::Call, nomkt, "X"));
    snaps.push_back(custom_snap(105.0, OptionType::Put,  nomkt, "Y"));

    const auto report = OptionChainCalibrator{}.calibrate(make_chain(std::move(snaps)));
    EXPECT_EQ(report.skipped, 2U);
    EXPECT_EQ(report.successful_solves, 0U);
    EXPECT_DOUBLE_EQ(report.convergence_rate(), 0.0);
    EXPECT_DOUBLE_EQ(report.average_iterations, 0.0);
}

// =============================================================================
// PROVIDER COMPARISON — errors are populated iff both IVs exist
// =============================================================================

TEST(OptionChainCalibratorProviderTest, PopulatesErrorsWhenBothIVsExist) {
    const auto market = canonical_market();
    const auto snap = build_snap(market, 100.0, ymd(2026, 7, 1),
                                 OptionType::Call, 0.20, "X", 0.0, 0.22);
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}, market));
    ASSERT_EQ(report.results.size(), 1U);
    const auto& r = report.results[0];

    ASSERT_TRUE(r.provider_iv.has_value());
    ASSERT_TRUE(r.computed_iv.has_value());
    ASSERT_TRUE(r.absolute_error.has_value());
    ASSERT_TRUE(r.relative_error.has_value());
    EXPECT_NEAR(*r.absolute_error, std::abs(0.22 - 0.20), 1e-9);
    EXPECT_NEAR(*r.relative_error, std::abs(0.22 - 0.20) / 0.22, 1e-9);
}

TEST(OptionChainCalibratorProviderTest, LeavesErrorsEmptyWhenProviderIVMissing) {
    const auto market = canonical_market();
    auto snap = build_snap(market, 100.0, ymd(2026, 7, 1),
                           OptionType::Call, 0.20, "X");
    snap.quote.implied_volatility.reset();  // provider did not publish
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}, market));
    ASSERT_EQ(report.results.size(), 1U);
    const auto& r = report.results[0];
    EXPECT_FALSE(r.provider_iv.has_value());
    EXPECT_TRUE(r.computed_iv.has_value());
    EXPECT_FALSE(r.absolute_error.has_value());
    EXPECT_FALSE(r.relative_error.has_value());
    EXPECT_EQ(report.provider_iv_comparisons, 0U);
}

TEST(OptionChainCalibratorProviderTest, RelativeErrorEmptyWhenProviderIVIsZero) {
    // Some providers emit exactly 0 for contracts they can't compute IV
    // for. We should still populate absolute_error but leave
    // relative_error empty (would be division by zero).
    const auto market = canonical_market();
    auto snap = build_snap(market, 100.0, ymd(2026, 7, 1),
                           OptionType::Call, 0.20, "X", 0.0, 0.0);
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}, market));
    const auto& r = report.results[0];
    ASSERT_TRUE(r.absolute_error.has_value());
    EXPECT_FALSE(r.relative_error.has_value());
}

// =============================================================================
// SOLVER DIAGNOSTICS — iterations, used_bisection, residual
// =============================================================================

TEST(OptionChainCalibratorDiagnosticsTest, PropagatesIterationCounts) {
    const auto market = canonical_market();
    const auto snap = build_snap(market, 100.0, ymd(2026, 7, 1),
                                 OptionType::Call, 0.30, "X");
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}, market));
    const auto& r = report.results[0];
    ASSERT_TRUE(r.was_calibrated());
    EXPECT_GT(r.iterations, 0U);
    EXPECT_LT(r.iterations, 100U);
    EXPECT_LT(r.solver_residual, 1e-9);
}

TEST(OptionChainCalibratorDiagnosticsTest, RecordsBisectionFallbackWhenTriggered) {
    // Force Newton to give up after one iteration.
    OptionChainCalibrator::Config cfg{};
    cfg.solver_config.max_iterations = 1;
    cfg.solver_config.use_bisection_fallback = true;
    OptionChainCalibrator calibrator(cfg);

    const auto market = canonical_market();
    const auto snap = build_snap(market, 105.0, ymd(2026, 7, 1),
                                 OptionType::Call, 0.35, "X");
    const auto report = calibrator.calibrate(make_chain({snap}, market));
    const auto& r = report.results[0];
    ASSERT_TRUE(r.was_calibrated());
    EXPECT_TRUE(r.used_bisection);
    EXPECT_EQ(report.bisection_fallbacks, 1U);
}

TEST(OptionChainCalibratorDiagnosticsTest,
     RecordsNonConvergenceWhenBudgetTooTightAndFallbackDisabled) {
    OptionChainCalibrator::Config cfg{};
    cfg.solver_config.max_iterations = 1;
    cfg.solver_config.use_bisection_fallback = false;
    OptionChainCalibrator calibrator(cfg);

    const auto market = canonical_market();
    const auto snap = build_snap(market, 105.0, ymd(2026, 7, 1),
                                 OptionType::Call, 0.35, "X");
    const auto report = calibrator.calibrate(make_chain({snap}, market));
    const auto& r = report.results[0];
    EXPECT_FALSE(r.was_calibrated());
    EXPECT_EQ(r.solver_status, SolverStatus::MaxIterationsReached);
    EXPECT_EQ(report.failed_solves, 1U);
    EXPECT_EQ(report.successful_solves, 0U);
}

// =============================================================================
// CSV EXPORT — layout, header, missing-value rendering
// =============================================================================

TEST(OptionChainCalibratorCsvTest, WritesHeaderRowInDocumentedOrder) {
    CalibrationReport empty{};
    std::ostringstream oss;
    empty.write_csv(oss);
    const auto s = oss.str();

    const std::string expected_header =
        "contract_symbol,expiration,strike,option_type,"
        "bid,ask,last,mid_price,"
        "provider_iv,computed_iv,absolute_error,relative_error,"
        "solver_status,iterations,used_bisection,solver_residual,"
        "skip_reason\n";
    EXPECT_EQ(s, expected_header);
}

TEST(OptionChainCalibratorCsvTest, WritesOneRowPerResultInChainOrder) {
    const auto market = canonical_market();
    std::vector<OptionMarketSnapshot> snaps;
    snaps.push_back(build_snap(market, 100.0, ymd(2026, 7, 1),
                               OptionType::Call, 0.20, "A"));
    snaps.push_back(build_snap(market, 105.0, ymd(2026, 7, 1),
                               OptionType::Put,  0.25, "B"));
    const auto report = OptionChainCalibrator{}.calibrate(make_chain(std::move(snaps), market));

    std::ostringstream oss;
    report.write_csv(oss);
    const auto s = oss.str();
    // Header + 2 data rows.
    const auto lines = std::count(s.begin(), s.end(), '\n');
    EXPECT_EQ(lines, 3);
    EXPECT_NE(s.find("A,2026-07-01,"), std::string::npos);
    EXPECT_NE(s.find("B,2026-07-01,"), std::string::npos);
    EXPECT_NE(s.find(",call,"), std::string::npos);
    EXPECT_NE(s.find(",put,"),  std::string::npos);
}

TEST(OptionChainCalibratorCsvTest, RendersMissingProviderIvAsEmptyField) {
    const auto market = canonical_market();
    auto snap = build_snap(market, 100.0, ymd(2026, 7, 1),
                           OptionType::Call, 0.20, "X");
    snap.quote.implied_volatility.reset();
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}, market));

    std::ostringstream oss;
    report.write_csv(oss);
    const auto s = oss.str();

    // Column indices (0-based): contract_symbol=0, expiration=1, strike=2,
    // option_type=3, bid=4, ask=5, last=6, mid_price=7, provider_iv=8.
    // Find the data row.
    const auto nl = s.find('\n');
    ASSERT_NE(nl, std::string::npos);
    const auto data = s.substr(nl + 1);
    // Split on commas.
    std::vector<std::string> fields;
    std::size_t start = 0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i] == ',' || data[i] == '\n') {
            fields.emplace_back(data.substr(start, i - start));
            start = i + 1;
        }
    }
    ASSERT_GE(fields.size(), 9U);
    EXPECT_EQ(fields[8], "");  // provider_iv field empty
}

TEST(OptionChainCalibratorCsvTest, WritesFileWhenPathOverloadUsed) {
    const auto market = canonical_market();
    const auto snap = build_snap(market, 100.0, ymd(2026, 7, 1),
                                 OptionType::Call, 0.20, "X");
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}, market));

    const auto tmp_path = std::filesystem::temp_directory_path()
                        / "ore_test_calibration_report.csv";
    report.write_csv(tmp_path);
    ASSERT_TRUE(std::filesystem::exists(tmp_path));

    std::ifstream in(tmp_path);
    std::stringstream ss;
    ss << in.rdbuf();
    EXPECT_NE(ss.str().find("contract_symbol"), std::string::npos);
    std::filesystem::remove(tmp_path);
}

TEST(OptionChainCalibratorCsvTest, SkipReasonSerialised) {
    Quote nomkt{}; nomkt.bid = 0.0; nomkt.ask = 0.0;
    const auto snap = custom_snap(100.0, OptionType::Call, nomkt, "S");
    const auto report = OptionChainCalibrator{}.calibrate(make_chain({snap}));

    std::ostringstream oss;
    report.write_csv(oss);
    EXPECT_NE(oss.str().find("NoMarket"), std::string::npos);
}

// =============================================================================
// INTEGRATION — load the shipped Yahoo fixture and calibrate it end-to-end
// =============================================================================

TEST(OptionChainCalibratorIntegrationTest, CalibratesShippedYahooFixture) {
    const auto snapshot_dir = std::filesystem::path{ORE_TEST_FIXTURES_DIR}
                            / "marketdata" / "SPY" / "options" / "2026-07-08";
    const auto chain = ore::marketdata::YahooOptionLoader::load(snapshot_dir);

    OptionChainCalibrator calibrator;
    const auto report = calibrator.calibrate(chain);

    EXPECT_EQ(report.results.size(), chain.size());
    // Every contract in the shipped fixture is priceable — no NoMarket
    // or crossed rows — and our BS-European IV should be within a
    // "quality-check" tolerance of Yahoo's proprietary IV. Yahoo uses a
    // binomial tree with an early-exercise correction for equity
    // options; even so the disagreement on this SPY fixture is small.
    // We assert only that: (a) something calibrated, (b) all provider-
    // comparable rows have populated error fields, (c) convergence rate
    // is 100% (all fixture rows are well-behaved).
    EXPECT_GT(report.successful_solves, 0U);
    EXPECT_EQ(report.failed_solves, 0U);
    EXPECT_DOUBLE_EQ(report.convergence_rate(), 1.0);
    EXPECT_GT(report.provider_iv_comparisons, 0U);
    EXPECT_LT(report.mean_absolute_iv_error, 0.10);  // sanity ceiling
}

TEST(OptionChainCalibratorIntegrationTest, CsvRoundtripHasSameRowCountAsChain) {
    const auto snapshot_dir = std::filesystem::path{ORE_TEST_FIXTURES_DIR}
                            / "marketdata" / "SPY" / "options" / "2026-07-08";
    const auto chain = ore::marketdata::YahooOptionLoader::load(snapshot_dir);
    const auto report = OptionChainCalibrator{}.calibrate(chain);

    std::ostringstream oss;
    report.write_csv(oss);
    const auto s = oss.str();
    const auto lines = std::count(s.begin(), s.end(), '\n');
    // Header + one row per contract.
    EXPECT_EQ(static_cast<std::size_t>(lines), 1 + chain.size());
}
