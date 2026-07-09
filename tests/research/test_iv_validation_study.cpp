/**
 * @file test_iv_validation_study.cpp
 * @brief IV validation correctness, CSV export shape, and stats.
 */
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <ore/research/historical_research_engine.hpp>
#include <ore/research/iv_validation_study.hpp>

#include "research_test_helpers.hpp"

namespace {

using namespace ore::research;
namespace fs = std::filesystem;

fs::path scratch(std::string_view label) {
    auto p = fs::temp_directory_path() /
        ("ore_iv_" + std::string{label} + "_" +
         std::to_string(std::hash<std::string_view>{}(label)));
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}

TEST(IVValidationStudy, RecoversProviderIVOnSyntheticData) {
    auto ds = ore::research::testing::make_synthetic_dataset(3, {}, 0.25);
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch("recover");
    HistoricalResearchEngine engine{ds, cfg};

    IVValidationStudy s{};
    auto report = engine.run(s);

    // Every synthetic contract is BS-consistent by construction, so
    // every solve must converge and the recovered IV must equal the
    // provider IV to within the solver's price tolerance times the
    // local sensitivity.  A conservative 1e-3 tolerance on |computed
    // - provider| picks up problems without being sensitive to
    // spread noise on deep OTM strikes.
    ASSERT_EQ(s.stats().solves_attempted, s.rows().size());
    ASSERT_GT(s.stats().solves_attempted, 0u);
    EXPECT_EQ(s.stats().solves_converged, s.stats().solves_attempted);
    EXPECT_LT(s.stats().mean_absolute_error, 1e-3);
    EXPECT_LT(s.stats().worst_error,          1e-2);
    EXPECT_NEAR(s.stats().convergence_rate, 1.0, 1e-12);

    EXPECT_GT(report.processed_contracts, 0u);
    ASSERT_EQ(report.generated_files.size(), 1u);
    EXPECT_TRUE(fs::exists(report.generated_files[0]));
}

TEST(IVValidationStudy, CsvHeaderMatchesSchema) {
    auto ds = ore::research::testing::make_synthetic_dataset(1);
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch("csv");
    HistoricalResearchEngine engine{ds, cfg};
    IVValidationStudy s{};
    auto report = engine.run(s);

    ASSERT_EQ(report.generated_files.size(), 1u);
    std::ifstream in(report.generated_files[0]);
    std::string header;
    std::getline(in, header);
    EXPECT_EQ(header,
        "date,expiration,strike,option_type,mid_price,provider_iv,"
        "computed_iv,absolute_error,relative_error,iterations,"
        "solver_method,converged");
}

TEST(IVValidationStudy, SkipsContractsWithoutProviderIV) {
    // Build a dataset then wipe the provider IV from every quote.
    auto ds_full = ore::research::testing::make_synthetic_dataset(1);
    std::vector<ore::marketdata::HistoricalSnapshot> stripped_snaps;
    for (const auto& snap : ds_full) {
        auto opts = snap.options();
        for (auto& oms : opts) {
            oms.quote.implied_volatility.reset();
        }
        ore::core::OptionChain chain{
            snap.underlying(), snap.market(), std::move(opts),
        };
        stripped_snaps.emplace_back(snap.date(), std::move(chain));
    }
    ore::marketdata::HistoricalDataset ds{
        std::string(ds_full.ticker()),
        std::move(stripped_snaps),
    };

    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch("no_iv");
    HistoricalResearchEngine engine{ds, cfg};
    IVValidationStudy s{};
    auto report = engine.run(s);

    EXPECT_EQ(report.processed_contracts, 0u);
    EXPECT_GT(report.skipped_contracts, 0u);
    EXPECT_TRUE(s.rows().empty());
}

TEST(IVValidationStudy, ParallelEqualsSerial) {
    auto ds = ore::research::testing::make_synthetic_dataset(15);

    HistoricalResearchEngine::Config serial_cfg{};
    serial_cfg.threads = 1;
    serial_cfg.output_dir = scratch("iv_serial");
    HistoricalResearchEngine serial_engine{ds, serial_cfg};
    IVValidationStudy s_serial{};
    auto r_serial = serial_engine.run(s_serial);

    HistoricalResearchEngine::Config par_cfg{};
    par_cfg.threads = 4;
    par_cfg.output_dir = scratch("iv_parallel");
    HistoricalResearchEngine par_engine{ds, par_cfg};
    IVValidationStudy s_parallel{};
    auto r_parallel = par_engine.run(s_parallel);

    EXPECT_EQ(r_serial.processed_contracts, r_parallel.processed_contracts);
    EXPECT_EQ(s_serial.rows().size(),       s_parallel.rows().size());
    // The aggregate stats must be numerically identical when
    // processing the same rows in an order-independent aggregator.
    EXPECT_NEAR(s_serial.stats().mean_absolute_error,
                s_parallel.stats().mean_absolute_error, 1e-15);
    EXPECT_NEAR(s_serial.stats().rmse,
                s_parallel.stats().rmse, 1e-15);
    EXPECT_EQ(s_serial.stats().solves_converged,
              s_parallel.stats().solves_converged);
}

} // namespace
