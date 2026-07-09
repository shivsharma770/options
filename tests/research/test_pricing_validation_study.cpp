/**
 * @file test_pricing_validation_study.cpp
 * @brief Pricing round-trip residuals should be at solver tolerance
 *        for BS-consistent inputs.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include <ore/research/historical_research_engine.hpp>
#include <ore/research/pricing_validation_study.hpp>

#include "research_test_helpers.hpp"

namespace {

using namespace ore::research;
namespace fs = std::filesystem;

fs::path scratch(std::string_view label) {
    auto p = fs::temp_directory_path() /
        ("ore_pricing_" + std::string{label} + "_" +
         std::to_string(std::hash<std::string_view>{}(label)));
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}

TEST(PricingValidationStudy, RoundTripsToSolverTolerance) {
    auto ds = ore::research::testing::make_synthetic_dataset(3);
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch("residual");
    HistoricalResearchEngine engine{ds, cfg};

    PricingValidationStudy::Config sc{};
    sc.residual_tolerance = 1e-6; // loose: mid quantised at 0.01 already
    PricingValidationStudy s{sc};

    auto report = engine.run(s);
    ASSERT_GT(report.processed_contracts, 0u);
    // Every row must converge and land within the solver's price
    // tolerance.  1e-6 covers even the deep OTM edge cases where the
    // solver drops into bisection.
    EXPECT_EQ(s.stats().total, s.stats().within_tolerance);
    EXPECT_LT(s.stats().mean_abs_residual, 1e-8);
    EXPECT_LT(s.stats().worst_abs_residual, 1e-6);
    EXPECT_NEAR(s.stats().fraction_within_tolerance, 1.0, 1e-12);
}

TEST(PricingValidationStudy, CsvHeaderShape) {
    auto ds = ore::research::testing::make_synthetic_dataset(1);
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch("csv_header");
    HistoricalResearchEngine engine{ds, cfg};
    PricingValidationStudy s{};
    auto report = engine.run(s);
    ASSERT_EQ(report.generated_files.size(), 1u);
    std::ifstream in(report.generated_files[0]);
    std::string header;
    std::getline(in, header);
    EXPECT_EQ(header,
        "date,expiration,strike,option_type,mid_price,"
        "recovered_iv,repriced,residual,converged");
}

TEST(PricingValidationStudy, ParallelEqualsSerial) {
    auto ds = ore::research::testing::make_synthetic_dataset(10);

    HistoricalResearchEngine::Config c_ser{};
    c_ser.threads = 1;
    c_ser.output_dir = scratch("pv_serial");
    HistoricalResearchEngine e_ser{ds, c_ser};
    PricingValidationStudy s_ser{};
    (void)e_ser.run(s_ser);

    HistoricalResearchEngine::Config c_par{};
    c_par.threads = 4;
    c_par.output_dir = scratch("pv_parallel");
    HistoricalResearchEngine e_par{ds, c_par};
    PricingValidationStudy s_par{};
    (void)e_par.run(s_par);

    EXPECT_EQ(s_ser.stats().total, s_par.stats().total);
    EXPECT_NEAR(s_ser.stats().mean_abs_residual,
                s_par.stats().mean_abs_residual, 1e-15);
    EXPECT_NEAR(s_ser.stats().rmse_residual,
                s_par.stats().rmse_residual, 1e-15);
}

} // namespace
