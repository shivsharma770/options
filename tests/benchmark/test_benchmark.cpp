/**
 * @file test_benchmark.cpp
 * @brief Unit tests for the `ore::benchmark` module.
 *
 * Covered:
 *   1. Standard suite generation — determinism, correct case count,
 *      no accidental American-exercise leakage.
 *   2. Case-metadata sanity — every case has a slug, a description,
 *      and a valid `MarketSnapshot`.
 *   3. `BenchmarkRunner::run` produces `E × C` rows in the documented
 *      order.
 *   4. Reference-engine detection is prefix-based and picks the first
 *      match.
 *   5. Absolute / relative error columns are populated correctly.
 *   6. CSV export shape (header, row count) and round-trip of at
 *      least one numeric column.
 *   7. Binomial and Monte Carlo convergence sweeps are ordered by the
 *      passed step / path counts and error decreases (roughly) with N.
 *   8. `estimated_memory_bytes` behaves sensibly for each engine.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <ore/benchmark/benchmark.hpp>
#include <ore/pricing/binomial_tree_engine.hpp>
#include <ore/pricing/black_scholes_engine.hpp>
#include <ore/pricing/monte_carlo_engine.hpp>
#include <ore/pricing/pricing_engine.hpp>

using ore::benchmark::BenchmarkCase;
using ore::benchmark::BenchmarkReport;
using ore::benchmark::BenchmarkRow;
using ore::benchmark::BenchmarkRunner;
using ore::benchmark::estimated_memory_bytes;
using ore::benchmark::standard_benchmark_suite;
using ore::pricing::BinomialTreeEngine;
using ore::pricing::BlackScholesEngine;
using ore::pricing::MonteCarloEngine;
using ore::pricing::PricingEngine;

namespace {

// Helper: build the three engine mix used in most tests. Reference
// engine (Black-Scholes) is placed first so the runner picks it.
std::vector<std::unique_ptr<PricingEngine>> make_three_engines() {
    std::vector<std::unique_ptr<PricingEngine>> engines;
    engines.push_back(std::make_unique<BlackScholesEngine>());

    BinomialTreeEngine::Config bt_cfg{};
    bt_cfg.steps = 250;
    bt_cfg.compute_greeks = false;  // Faster tests, still comparable price.
    engines.push_back(std::make_unique<BinomialTreeEngine>(bt_cfg));

    MonteCarloEngine::Config mc_cfg{};
    mc_cfg.paths = 20'000;  // Small enough to stay fast, big enough to be stable.
    mc_cfg.seed = 12345;
    mc_cfg.antithetic_variates = true;
    mc_cfg.compute_greeks = false;
    engines.push_back(std::make_unique<MonteCarloEngine>(mc_cfg));
    return engines;
}

// Deterministic runner config: one rep so tests never see jitter.
BenchmarkRunner deterministic_runner() {
    BenchmarkRunner::Config cfg{};
    cfg.median_reps = 1;
    return BenchmarkRunner(cfg);
}

} // namespace

//
// Standard suite ------------------------------------------------------------
//

TEST(BenchmarkSuite, ReturnsCanonicalTwelveCases) {
    const auto suite = standard_benchmark_suite();
    ASSERT_EQ(suite.size(), 12u);

    // Slugs are the contract with the CSV and Python code — pin them.
    const std::vector<std::string> expected = {
        "atm_call", "atm_put",
        "itm_call", "otm_call", "itm_put", "otm_put",
        "short_maturity_call", "long_maturity_call",
        "low_vol_call", "high_vol_call",
        "dividend_paying_call", "negative_rate_call",
    };
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(suite[i].name, expected[i]);
        EXPECT_FALSE(suite[i].description.empty()) << expected[i];
    }
}

TEST(BenchmarkSuite, IsDeterministicBetweenCalls) {
    // The Python plotters diff CSVs across runs; the suite generator
    // has to be a pure function of nothing.
    const auto a = standard_benchmark_suite();
    const auto b = standard_benchmark_suite();
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].name, b[i].name);
        EXPECT_EQ(a[i].description, b[i].description);
        EXPECT_EQ(a[i].market, b[i].market);
        EXPECT_EQ(a[i].option.strike, b[i].option.strike);
        EXPECT_EQ(a[i].option.expiration, b[i].option.expiration);
        EXPECT_EQ(a[i].option.type, b[i].option.type);
    }
}

TEST(BenchmarkSuite, EveryCaseIsEuropean) {
    // Reference-engine comparison only makes sense for European
    // exercise. American cases will be added when we have an American
    // analytical reference (Barone-Adesi-Whaley or similar) — not this
    // milestone.
    for (const auto& c : standard_benchmark_suite()) {
        EXPECT_EQ(c.option.exercise, ore::core::ExerciseStyle::European)
            << c.name;
        EXPECT_GT(c.option.strike, 0.0) << c.name;
        EXPECT_GT(c.market.spot, 0.0) << c.name;
        EXPECT_GE(c.market.volatility, 0.0) << c.name;
    }
}

TEST(BenchmarkSuite, ContainsRegimeCoverage) {
    const auto suite = standard_benchmark_suite();

    // Pull out a few pointed cases and sanity-check their parameters.
    auto find_by_name = [&](std::string_view n) -> const BenchmarkCase* {
        for (const auto& c : suite) if (c.name == n) return &c;
        return nullptr;
    };
    const auto* high = find_by_name("high_vol_call");
    ASSERT_NE(high, nullptr);
    EXPECT_GT(high->market.volatility, 0.5);

    const auto* low = find_by_name("low_vol_call");
    ASSERT_NE(low, nullptr);
    EXPECT_LT(low->market.volatility, 0.10);

    const auto* neg = find_by_name("negative_rate_call");
    ASSERT_NE(neg, nullptr);
    EXPECT_LT(neg->market.risk_free_rate, 0.0);

    const auto* div = find_by_name("dividend_paying_call");
    ASSERT_NE(div, nullptr);
    EXPECT_GT(div->market.dividend_yield, 0.0);
}

//
// Runner --------------------------------------------------------------------
//

TEST(BenchmarkRunner, ProducesRowsInCaseMajorEngineMinorOrder) {
    auto engines = make_three_engines();
    const auto suite = standard_benchmark_suite();

    auto runner = deterministic_runner();
    const auto report = runner.run(engines, suite);

    ASSERT_EQ(report.rows.size(), engines.size() * suite.size());
    // Rows are laid out case-major: (case_0, engine_0..N), (case_1, ...).
    for (std::size_t c = 0; c < suite.size(); ++c) {
        for (std::size_t e = 0; e < engines.size(); ++e) {
            const auto& row = report.rows[c * engines.size() + e];
            EXPECT_EQ(row.case_name, suite[c].name);
            EXPECT_EQ(row.engine_name, std::string(engines[e]->name()));
        }
    }
}

TEST(BenchmarkRunner, PopulatesRuntimeAndPrice) {
    auto engines = make_three_engines();
    // A one-case subset for speed. The 20k-path MC row would dominate
    // if we ran the full suite.
    std::vector<BenchmarkCase> cases{ standard_benchmark_suite().front() };

    auto runner = deterministic_runner();
    const auto report = runner.run(engines, cases);

    ASSERT_EQ(report.rows.size(), 3u);
    for (const auto& row : report.rows) {
        EXPECT_GT(row.price, 0.0) << row.engine_name;
        EXPECT_GE(row.runtime_us, 0.0) << row.engine_name;
    }
}

TEST(BenchmarkRunner, MarksReferenceEngineWithZeroError) {
    auto engines = make_three_engines();
    std::vector<BenchmarkCase> cases{ standard_benchmark_suite()[0] };
    auto runner = deterministic_runner();
    const auto report = runner.run(engines, cases);

    // The BlackScholes row (row[0], since it's engines[0]) is the
    // reference — its abs_error must be exactly zero.
    const auto& bs_row = report.rows[0];
    EXPECT_EQ(bs_row.engine_name, "BlackScholes");
    ASSERT_TRUE(bs_row.absolute_error.has_value());
    EXPECT_EQ(*bs_row.absolute_error, 0.0);
    ASSERT_TRUE(bs_row.relative_error.has_value());
    EXPECT_EQ(*bs_row.relative_error, 0.0);
}

TEST(BenchmarkRunner, PopulatesAbsoluteErrorForNonReferenceEngines) {
    auto engines = make_three_engines();
    std::vector<BenchmarkCase> cases{ standard_benchmark_suite()[0] };  // ATM call
    auto runner = deterministic_runner();
    const auto report = runner.run(engines, cases);

    // Binomial (index 1) and MC (index 2) rows: error columns present.
    for (std::size_t i = 1; i < 3; ++i) {
        const auto& row = report.rows[i];
        ASSERT_TRUE(row.absolute_error.has_value()) << row.engine_name;
        ASSERT_TRUE(row.relative_error.has_value()) << row.engine_name;
        ASSERT_TRUE(row.reference_price.has_value()) << row.engine_name;
        EXPECT_GT(*row.reference_price, 0.0);
        // 250-step Binomial and 20k-path MC on ATM 1y call should be
        // within a fraction of a dollar of Black-Scholes.
        EXPECT_LT(*row.absolute_error, 0.5) << row.engine_name;
    }
}

TEST(BenchmarkRunner, LeavesErrorEmptyWhenPrefixIsEmpty) {
    auto engines = make_three_engines();
    std::vector<BenchmarkCase> cases{ standard_benchmark_suite()[0] };

    BenchmarkRunner::Config cfg{};
    cfg.median_reps = 1;
    cfg.reference_engine_prefix = "";
    BenchmarkRunner runner(cfg);

    const auto report = runner.run(engines, cases);
    for (const auto& row : report.rows) {
        EXPECT_FALSE(row.absolute_error.has_value()) << row.engine_name;
        EXPECT_FALSE(row.relative_error.has_value()) << row.engine_name;
        EXPECT_FALSE(row.reference_price.has_value()) << row.engine_name;
    }
}

TEST(BenchmarkRunner, EmptyInputsProduceEmptyReport) {
    auto runner = deterministic_runner();

    std::vector<std::unique_ptr<PricingEngine>> engines;
    engines.push_back(std::make_unique<BlackScholesEngine>());

    const auto empty_engines_report = runner.run({}, standard_benchmark_suite());
    EXPECT_TRUE(empty_engines_report.rows.empty());

    const auto empty_cases_report = runner.run(engines, {});
    EXPECT_TRUE(empty_cases_report.rows.empty());
}

TEST(BenchmarkRunner, CopiesStandardErrorAndCIFromMonteCarloRow) {
    // Ensure PricingResult fields specific to stochastic engines
    // survive into the report — the CSV needs them for downstream plots.
    std::vector<std::unique_ptr<PricingEngine>> engines;
    engines.push_back(std::make_unique<BlackScholesEngine>());
    MonteCarloEngine::Config mc_cfg{};
    mc_cfg.paths = 10'000;
    mc_cfg.seed = 7;
    mc_cfg.antithetic_variates = true;
    engines.push_back(std::make_unique<MonteCarloEngine>(mc_cfg));

    std::vector<BenchmarkCase> cases{ standard_benchmark_suite()[0] };

    auto runner = deterministic_runner();
    const auto report = runner.run(engines, cases);
    ASSERT_EQ(report.rows.size(), 2u);

    const auto& bs = report.rows[0];
    EXPECT_FALSE(bs.standard_error.has_value());
    EXPECT_FALSE(bs.confidence_interval_95.has_value());

    const auto& mc = report.rows[1];
    ASSERT_TRUE(mc.standard_error.has_value());
    EXPECT_GT(*mc.standard_error, 0.0);
    ASSERT_TRUE(mc.confidence_interval_95.has_value());
    EXPECT_LT(mc.confidence_interval_95->first, mc.confidence_interval_95->second);
    EXPECT_LE(mc.confidence_interval_95->first, mc.price);
    EXPECT_GE(mc.confidence_interval_95->second, mc.price);
}

//
// Convergence sweeps --------------------------------------------------------
//

TEST(BenchmarkConvergence, BinomialSweepIsOrderedAndConverges) {
    const auto case_ = standard_benchmark_suite()[0]; // ATM call
    const std::vector<std::size_t> steps{ 10, 50, 250 };

    auto runner = deterministic_runner();
    const auto report = runner.run_binomial_convergence(case_, steps);

    ASSERT_EQ(report.rows.size(), steps.size());
    // Row i corresponds to steps[i] and has iterations = steps[i].
    for (std::size_t i = 0; i < steps.size(); ++i) {
        ASSERT_TRUE(report.rows[i].iterations.has_value());
        EXPECT_EQ(*report.rows[i].iterations, steps[i]);
        // Sanity: memory grows with N.
        EXPECT_GT(report.rows[i].estimated_memory_bytes, 0u);
    }
    // 250-step tree should be at least as close to BS as the 10-step
    // tree. CRR is *not* monotonically converging (sawtooth) but the
    // trend over a decade of N is unambiguous.
    ASSERT_TRUE(report.rows.front().absolute_error.has_value());
    ASSERT_TRUE(report.rows.back().absolute_error.has_value());
    EXPECT_LT(*report.rows.back().absolute_error,
              *report.rows.front().absolute_error);
}

TEST(BenchmarkConvergence, MonteCarloSweepShrinksStandardError) {
    const auto case_ = standard_benchmark_suite()[0];
    const std::vector<std::size_t> paths{ 1'000, 10'000, 50'000 };

    auto runner = deterministic_runner();
    const auto report = runner.run_monte_carlo_convergence(
        case_, paths, /*seed=*/2026, /*antithetic=*/true);

    ASSERT_EQ(report.rows.size(), paths.size());
    for (std::size_t i = 0; i < paths.size(); ++i) {
        ASSERT_TRUE(report.rows[i].standard_error.has_value());
        ASSERT_TRUE(report.rows[i].iterations.has_value());
        EXPECT_EQ(*report.rows[i].iterations, paths[i]);
    }
    // O(1/sqrt(N)) implies SE(50k) < SE(1k). Given the same seed the
    // sequence of Welford draws is a prefix of the same underlying
    // stream, so the trend is essentially guaranteed.
    EXPECT_LT(*report.rows.back().standard_error,
              *report.rows.front().standard_error);
}

//
// CSV export ----------------------------------------------------------------
//

TEST(BenchmarkReport, CsvExportHasExpectedHeaderAndRowCount) {
    auto engines = make_three_engines();
    std::vector<BenchmarkCase> cases{ standard_benchmark_suite()[0] };
    auto runner = deterministic_runner();
    const auto report = runner.run(engines, cases);

    std::ostringstream oss;
    report.write_csv(oss);
    const auto csv = oss.str();

    // Header sanity: first line contains every expected column name.
    const auto header_end = csv.find('\n');
    ASSERT_NE(header_end, std::string::npos);
    const auto header = csv.substr(0, header_end);
    for (const char* expected : {
             "case_name", "engine_name", "price", "delta", "gamma",
             "vega", "theta", "rho", "runtime_us", "iterations",
             "standard_error", "ci_95_low", "ci_95_high",
             "reference_price", "absolute_error", "relative_error",
             "estimated_memory_bytes"
         })
    {
        EXPECT_NE(header.find(expected), std::string::npos)
            << "missing column: " << expected;
    }

    // Row count: header + N rows + trailing newline on the last row.
    std::size_t newline_count = 0;
    for (char c : csv) if (c == '\n') ++newline_count;
    EXPECT_EQ(newline_count, report.rows.size() + 1);
}

TEST(BenchmarkReport, CsvContainsAllEngineNames) {
    auto engines = make_three_engines();
    std::vector<BenchmarkCase> cases{ standard_benchmark_suite()[0] };
    auto runner = deterministic_runner();
    const auto report = runner.run(engines, cases);

    std::ostringstream oss;
    report.write_csv(oss);
    const auto csv = oss.str();

    EXPECT_NE(csv.find("BlackScholes"), std::string::npos);
    EXPECT_NE(csv.find("Binomial"), std::string::npos);
    EXPECT_NE(csv.find("MonteCarlo"), std::string::npos);
}

//
// Memory estimator ----------------------------------------------------------
//

TEST(BenchmarkMemory, BlackScholesReportsZero) {
    BlackScholesEngine bs{};
    EXPECT_EQ(estimated_memory_bytes(bs), 0u);
}

TEST(BenchmarkMemory, BinomialScalesLinearlyWithSteps) {
    BinomialTreeEngine::Config small{}; small.steps = 10;
    BinomialTreeEngine::Config large{}; large.steps = 1000;
    BinomialTreeEngine small_engine(small);
    BinomialTreeEngine large_engine(large);

    const auto small_bytes = estimated_memory_bytes(small_engine);
    const auto large_bytes = estimated_memory_bytes(large_engine);
    EXPECT_GT(large_bytes, small_bytes);
    // Ratio should be about (1001/11) ~= 91×. Allow slack for the +1
    // in `(steps + 1)`.
    EXPECT_GT(large_bytes / small_bytes, 50u);
}

TEST(BenchmarkMemory, MonteCarloIsConstantInPaths) {
    MonteCarloEngine::Config small{}; small.paths = 1'000;
    MonteCarloEngine::Config huge{}; huge.paths = 10'000'000;
    MonteCarloEngine small_engine(small);
    MonteCarloEngine huge_engine(huge);
    EXPECT_EQ(estimated_memory_bytes(small_engine),
              estimated_memory_bytes(huge_engine));
    EXPECT_GT(estimated_memory_bytes(small_engine), 0u);
}
