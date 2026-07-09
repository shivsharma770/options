/**
 * @file test_eod_options_loader.cpp
 * @brief Unit coverage for the OptionsDX / Delta-Neutral EOD parser.
 */
#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include <ore/marketdata/eod_options_loader.hpp>
#include <ore/marketdata/historical_loader.hpp>

namespace {

using namespace ore::marketdata;
namespace fs = std::filesystem;

/// A tiny two-day fixture that exercises the parser's quirks:
///   - bracketed column headers,
///   - whitespace after every comma,
///   - a row with an empty IV field,
///   - two distinct QUOTE_DATE values so grouping is tested.
constexpr std::string_view kSampleFile =
    "[QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE], [QUOTE_TIME_HOURS], "
    "[UNDERLYING_LAST], [EXPIRE_DATE], [EXPIRE_UNIX], [DTE], "
    "[C_DELTA], [C_GAMMA], [C_VEGA], [C_THETA], [C_RHO], [C_IV], [C_VOLUME], "
    "[C_LAST], [C_SIZE], [C_BID], [C_ASK], [STRIKE], "
    "[P_BID], [P_ASK], [P_SIZE], [P_LAST], [P_DELTA], [P_GAMMA], [P_VEGA], "
    "[P_THETA], [P_RHO], [P_IV], [P_VOLUME], "
    "[STRIKE_DISTANCE], [STRIKE_DISTANCE_PCT]\n"
    "1262638800, 2010-01-04 16:00, 2010-01-04, 16.0, 100.0, 2010-01-15, 1263589200, 11.0, "
    "0.55, 0.02, 0.15, -0.01, 0.005, 0.20, 100, 5.0, 10x10, 4.9, 5.1, 100.0, "
    "4.8, 5.0, 10x10, 4.9, -0.45, 0.02, 0.15, -0.01, -0.005, 0.20, 100, 0.0, 0.0\n"
    "1262638800, 2010-01-04 16:00, 2010-01-04, 16.0, 100.0, 2010-01-15, 1263589200, 11.0, "
    "0.30, 0.03, 0.15, -0.01, 0.005, , 50, 2.0, 5x5, 1.9, 2.1, 105.0, "
    "6.8, 7.0, 10x10, 6.9, -0.70, 0.03, 0.15, -0.01, -0.005, 0.30, 50, 5.0, 0.05\n"
    "1262725200, 2010-01-05 16:00, 2010-01-05, 16.0, 101.0, 2010-01-15, 1263589200, 10.0, "
    "0.60, 0.02, 0.15, -0.01, 0.005, 0.22, 120, 5.5, 10x10, 5.4, 5.6, 100.0, "
    "3.9, 4.1, 10x10, 4.0, -0.40, 0.02, 0.15, -0.01, -0.005, 0.22, 120, 1.0, 0.01\n";

TEST(EodOptionsLoader, GroupsRowsByQuoteDate) {
    auto snaps = EodOptionsLoader::parse_string(kSampleFile);
    ASSERT_EQ(snaps.size(), 2u);
    EXPECT_EQ(snaps[0].date(), (std::chrono::year{2010}/std::chrono::January/4));
    EXPECT_EQ(snaps[1].date(), (std::chrono::year{2010}/std::chrono::January/5));

    // Two strikes on 2010-01-04 → four contracts (call + put).
    EXPECT_EQ(snaps[0].size(), 4u);
    // One strike on 2010-01-05 → two contracts.
    EXPECT_EQ(snaps[1].size(), 2u);
}

TEST(EodOptionsLoader, PopulatesProviderGreeksAndSpot) {
    auto snaps = EodOptionsLoader::parse_string(kSampleFile);
    ASSERT_EQ(snaps.size(), 2u);
    const auto& snap = snaps[0];

    // Spot recovered from `UNDERLYING_LAST`.
    EXPECT_DOUBLE_EQ(snap.market().spot, 100.0);
    EXPECT_EQ(snap.market().valuation_date,
              (std::chrono::year{2010}/std::chrono::January/4));

    // First option in the chain is the call at strike 100 (order:
    // strike 100 call, strike 100 put, strike 105 call, strike 105
    // put — mirrors the file order).
    const auto& call100 = snap.options().front();
    EXPECT_EQ(call100.option.type,   ore::core::OptionType::Call);
    EXPECT_DOUBLE_EQ(call100.option.strike, 100.0);
    EXPECT_DOUBLE_EQ(call100.quote.bid, 4.9);
    EXPECT_DOUBLE_EQ(call100.quote.ask, 5.1);
    ASSERT_TRUE(call100.quote.provider_greeks.has_value());
    EXPECT_TRUE(call100.quote.provider_greeks->delta.has_value());
    EXPECT_NEAR(*call100.quote.provider_greeks->delta, 0.55, 1e-12);
    ASSERT_TRUE(call100.quote.implied_volatility.has_value());
    EXPECT_NEAR(*call100.quote.implied_volatility, 0.20, 1e-12);
}

TEST(EodOptionsLoader, EmptyIvSurvivesAsNullopt) {
    auto snaps = EodOptionsLoader::parse_string(kSampleFile);
    // 2010-01-04 has two strikes; the 105-strike call has empty C_IV
    // in the fixture. Third option in the chain (index 2) is the
    // call at strike 105.
    ASSERT_EQ(snaps[0].options().size(), 4u);
    const auto& call105 = snaps[0].options()[2];
    EXPECT_DOUBLE_EQ(call105.option.strike, 105.0);
    EXPECT_EQ(call105.option.type, ore::core::OptionType::Call);
    EXPECT_FALSE(call105.quote.implied_volatility.has_value())
        << "Missing IV must survive as std::nullopt, not zero.";
}

TEST(EodOptionsLoader, LoadFromDirectoryDiscoversAllArchives) {
    // Build a temporary directory with two archives spanning the
    // same two days each so the merge path is exercised too.
    auto root = fs::temp_directory_path() / "ore_eod_test";
    fs::remove_all(root);
    fs::create_directories(root / "spy_eod_2010-part1");
    fs::create_directories(root / "spy_eod_2010-part2");
    {
        std::ofstream(root / "spy_eod_2010-part1" / "spy_eod_201001.txt")
            << kSampleFile;
    }
    {
        std::ofstream(root / "spy_eod_2010-part2" / "spy_eod_201002.txt")
            << kSampleFile;
    }

    auto ds = HistoricalLoader::load_from_eod(root, "SPY");
    ASSERT_EQ(ds.size(), 2u); // two distinct trading dates
    // Each date should have contracts from BOTH archive files, so
    // the option count doubles.
    EXPECT_EQ(ds.snapshots()[0].size(), 8u);
    EXPECT_EQ(ds.snapshots()[1].size(), 4u);
    EXPECT_EQ(ds.ticker(), "SPY");
}

TEST(EodOptionsLoader, MissingRequiredColumnThrows) {
    // Drop `STRIKE` from the header — parsing should refuse to
    // proceed rather than silently misalign columns.
    constexpr std::string_view kBad =
        "[QUOTE_DATE], [UNDERLYING_LAST], [EXPIRE_DATE], [C_BID], [C_ASK], "
        "[P_BID], [P_ASK]\n"
        "2010-01-04, 100.0, 2010-01-15, 4.9, 5.1, 4.8, 5.0\n";
    EXPECT_THROW({
        (void)EodOptionsLoader::parse_string(kBad);
    }, LoaderError);
}

TEST(EodOptionsLoader, DiscoverFilesIgnoresNonArchiveEntries) {
    auto root = fs::temp_directory_path() / "ore_eod_discover_test";
    fs::remove_all(root);
    fs::create_directories(root);
    std::ofstream(root / "spy_eod_201001.txt") << "";
    std::ofstream(root / "spy_eod_201002.csv") << "";
    std::ofstream(root / "README.md")          << "";
    std::ofstream(root / ".DS_Store")          << "";
    auto files = EodOptionsLoader::discover_files(root);
    ASSERT_EQ(files.size(), 2u);
    EXPECT_EQ(files[0].filename().string(), "spy_eod_201001.txt");
    EXPECT_EQ(files[1].filename().string(), "spy_eod_201002.csv");
}

} // namespace
