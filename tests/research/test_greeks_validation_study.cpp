/**
 * @file test_greeks_validation_study.cpp
 * @brief Greeks-validation correctness and CSV shape.
 */
#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include <ore/research/greeks_validation_study.hpp>
#include <ore/research/historical_research_engine.hpp>

#include "research_test_helpers.hpp"

namespace {

using namespace ore::research;
namespace fs = std::filesystem;

fs::path scratch(std::string_view label) {
    auto p = fs::temp_directory_path() /
        ("ore_greeks_" + std::string{label} + "_" +
         std::to_string(std::hash<std::string_view>{}(label)));
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}

TEST(GreeksValidationStudy, ProviderMatchesComputedOnSyntheticData) {
    // The synthetic dataset stores provider Greeks that are exactly
    // the BS Greeks converted into the vendor's unit convention, so
    // every error should be zero to floating-point noise.
    auto ds = ore::research::testing::make_synthetic_dataset(2);
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch("zero_error");
    HistoricalResearchEngine engine{ds, cfg};
    GreeksValidationStudy s{};
    auto report = engine.run(s);

    ASSERT_GT(report.processed_contracts, 0u);
    EXPECT_NEAR(s.stats().delta.mae, 0.0, 1e-12);
    EXPECT_NEAR(s.stats().gamma.mae, 0.0, 1e-12);
    EXPECT_NEAR(s.stats().theta.mae, 0.0, 1e-12);
    EXPECT_NEAR(s.stats().vega.mae,  0.0, 1e-12);
    EXPECT_NEAR(s.stats().rho.mae,   0.0, 1e-12);
    EXPECT_NEAR(s.stats().delta.rmse, 0.0, 1e-12);
    EXPECT_NEAR(s.stats().gamma.rmse, 0.0, 1e-12);
}

TEST(GreeksValidationStudy, CsvHeaderShape) {
    auto ds = ore::research::testing::make_synthetic_dataset(1);
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch("csv_header");
    HistoricalResearchEngine engine{ds, cfg};
    GreeksValidationStudy s{};
    auto report = engine.run(s);
    ASSERT_EQ(report.generated_files.size(), 1u);
    std::ifstream in(report.generated_files[0]);
    std::string header;
    std::getline(in, header);
    // 21 columns total: 5 identifiers + 5 Greeks × 3 (provider / computed / error).
    EXPECT_EQ(header,
        "date,expiration,strike,option_type,provider_iv,"
        "provider_delta,computed_delta,delta_error,"
        "provider_gamma,computed_gamma,gamma_error,"
        "provider_theta,computed_theta,theta_error,"
        "provider_vega,computed_vega,vega_error,"
        "provider_rho,computed_rho,rho_error");
}

TEST(GreeksValidationStudy, SkipsRowsWithoutProviderGreeks) {
    auto ds_full = ore::research::testing::make_synthetic_dataset(1);
    std::vector<ore::marketdata::HistoricalSnapshot> stripped;
    for (const auto& snap : ds_full) {
        auto opts = snap.options();
        for (auto& oms : opts) {
            oms.quote.provider_greeks.reset();
        }
        ore::core::OptionChain chain{
            snap.underlying(), snap.market(), std::move(opts),
        };
        stripped.emplace_back(snap.date(), std::move(chain));
    }
    ore::marketdata::HistoricalDataset ds{
        std::string(ds_full.ticker()),
        std::move(stripped),
    };
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch("no_greeks");
    HistoricalResearchEngine engine{ds, cfg};
    GreeksValidationStudy s{};
    auto report = engine.run(s);
    EXPECT_EQ(report.processed_contracts, 0u);
    EXPECT_GT(report.skipped_contracts, 0u);
}

TEST(GreeksValidationStudy, ParallelEqualsSerial) {
    auto ds = ore::research::testing::make_synthetic_dataset(12);
    HistoricalResearchEngine::Config c_ser{};
    c_ser.threads = 1;
    c_ser.output_dir = scratch("greeks_serial");
    HistoricalResearchEngine e_ser{ds, c_ser};
    GreeksValidationStudy s_ser{};
    (void)e_ser.run(s_ser);

    HistoricalResearchEngine::Config c_par{};
    c_par.threads = 4;
    c_par.output_dir = scratch("greeks_parallel");
    HistoricalResearchEngine e_par{ds, c_par};
    GreeksValidationStudy s_par{};
    (void)e_par.run(s_par);

    EXPECT_EQ(s_ser.rows().size(), s_par.rows().size());
    EXPECT_NEAR(s_ser.stats().delta.rmse, s_par.stats().delta.rmse, 1e-15);
    EXPECT_NEAR(s_ser.stats().gamma.rmse, s_par.stats().gamma.rmse, 1e-15);
    EXPECT_NEAR(s_ser.stats().vega.rmse,  s_par.stats().vega.rmse,  1e-15);
    EXPECT_NEAR(s_ser.stats().theta.rmse, s_par.stats().theta.rmse, 1e-15);
    EXPECT_NEAR(s_ser.stats().rho.rmse,   s_par.stats().rho.rmse,   1e-15);
}

} // namespace
