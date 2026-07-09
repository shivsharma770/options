/**
 * @file test_historical_loader.cpp
 * @brief Tests for `HistoricalLoader`, `HistoricalDataset`, and
 *        `HistoricalSnapshot`.
 *
 * Coverage:
 *   1.  Loading a single day from a fixture directory.
 *   2.  Loading two clean days into a dataset (with the ordering
 *       invariant).
 *   3.  Enumeration: `list_dates` skips non-date directories and
 *       stray files.
 *   4.  Empty ticker directory → empty dataset (no error).
 *   5.  Strict mode: malformed day propagates `LoaderError`.
 *   6.  Lenient mode: malformed day is recorded, other days load.
 *   7.  Date range filtering (`start_date`, `end_date`).
 *   8.  Statistics: counts, averages, first/last dates, missing days.
 *   9.  Range-based-for and `operator[]` yield the same snapshots.
 *  10.  Multiple tickers: two fixtures under different symbols coexist.
 *  11.  Pricing-engine integration: a loaded snapshot's `market()`
 *       flows straight into `PricingEngine::price` without any
 *       adapter code.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <ore/marketdata/historical_dataset.hpp>
#include <ore/marketdata/historical_loader.hpp>
#include <ore/marketdata/historical_snapshot.hpp>
#include <ore/marketdata/yahoo_option_loader.hpp>
#include <ore/pricing/black_scholes_engine.hpp>

#ifndef ORE_TEST_FIXTURES_DIR
#error "ORE_TEST_FIXTURES_DIR must be defined by the build system"
#endif

using ore::marketdata::HistoricalDataset;
using ore::marketdata::HistoricalLoader;
using ore::marketdata::HistoricalSnapshot;
using ore::marketdata::LoaderError;
using ore::pricing::BlackScholesEngine;

namespace fs = std::filesystem;

namespace {

/** Root of the historical fixture tree, `.../historical/`. */
fs::path historical_root() {
    return fs::path{ORE_TEST_FIXTURES_DIR} / "marketdata" / "historical";
}

std::chrono::year_month_day ymd(int y, unsigned m, unsigned d) {
    return {std::chrono::year{y},
            std::chrono::month{m},
            std::chrono::day{d}};
}

} // namespace

// -----------------------------------------------------------------------------
// list_dates
// -----------------------------------------------------------------------------

TEST(HistoricalLoader, ListDatesReturnsSortedTradingDates) {
    const auto dates = HistoricalLoader::list_dates("SPY", historical_root());

    // The fixture has 2024-01-02, 2024-01-03, 2024-01-04, plus a
    // README.txt file and a `not_a_date_dir/` directory that must be
    // filtered out.
    ASSERT_EQ(dates.size(), 3u);
    EXPECT_EQ(dates[0], ymd(2024, 1, 2));
    EXPECT_EQ(dates[1], ymd(2024, 1, 3));
    EXPECT_EQ(dates[2], ymd(2024, 1, 4));
}

TEST(HistoricalLoader, ListDatesForUnknownTickerReturnsEmpty) {
    const auto dates = HistoricalLoader::list_dates("DOES_NOT_EXIST",
                                                    historical_root());
    EXPECT_TRUE(dates.empty());
}

// -----------------------------------------------------------------------------
// load_day
// -----------------------------------------------------------------------------

TEST(HistoricalLoader, LoadDaySuccess) {
    const auto snap = HistoricalLoader::load_day(
        "SPY", ymd(2024, 1, 2), historical_root());

    EXPECT_EQ(snap.date(), ymd(2024, 1, 2));
    EXPECT_EQ(snap.underlying().symbol, "SPY");
    EXPECT_DOUBLE_EQ(snap.market().spot, 470.10);
    // 3 calls + 3 puts.
    EXPECT_EQ(snap.size(), 6u);
    EXPECT_FALSE(snap.empty());
    EXPECT_EQ(snap.market().valuation_date, snap.date());
}

TEST(HistoricalLoader, LoadDayForMissingDateThrows) {
    EXPECT_THROW(
        HistoricalLoader::load_day("SPY", ymd(2020, 1, 1), historical_root()),
        LoaderError);
}

// -----------------------------------------------------------------------------
// load(ticker, Options)
// -----------------------------------------------------------------------------

TEST(HistoricalLoader, LoadRangeStrictSucceedsOnCleanDays) {
    HistoricalLoader::Options opts{};
    opts.root       = historical_root();
    opts.start_date = ymd(2024, 1, 2);
    opts.end_date   = ymd(2024, 1, 3);
    opts.strict     = true;

    const auto result = HistoricalLoader::load("SPY", opts);
    ASSERT_EQ(result.dataset.size(), 2u);
    EXPECT_TRUE(result.failed_days.empty());

    EXPECT_EQ(result.dataset[0].date(), ymd(2024, 1, 2));
    EXPECT_EQ(result.dataset[1].date(), ymd(2024, 1, 3));
}

TEST(HistoricalLoader, LoadRangeStrictPropagatesLoaderError) {
    HistoricalLoader::Options opts{};
    opts.root       = historical_root();
    opts.start_date = ymd(2024, 1, 2);
    opts.end_date   = ymd(2024, 1, 4);  // Includes the malformed day.
    opts.strict     = true;

    EXPECT_THROW(HistoricalLoader::load("SPY", opts), LoaderError);
}

TEST(HistoricalLoader, LoadRangeLenientRecordsFailuresAndKeepsGoing) {
    HistoricalLoader::Options opts{};
    opts.root       = historical_root();
    opts.start_date = ymd(2024, 1, 2);
    opts.end_date   = ymd(2024, 1, 4);
    opts.strict     = false;

    const auto result = HistoricalLoader::load("SPY", opts);

    // Two clean days load; the malformed day is recorded.
    ASSERT_EQ(result.dataset.size(), 2u);
    ASSERT_EQ(result.failed_days.size(), 1u);
    EXPECT_EQ(result.failed_days[0].first, ymd(2024, 1, 4));
    EXPECT_FALSE(result.failed_days[0].second.empty());
}

TEST(HistoricalLoader, EmptyRangeYieldsEmptyDataset) {
    HistoricalLoader::Options opts{};
    opts.root       = historical_root();
    // End before start on purpose.
    opts.start_date = ymd(2024, 2, 1);
    opts.end_date   = ymd(2024, 1, 15);
    opts.strict     = true;

    const auto result = HistoricalLoader::load("SPY", opts);
    EXPECT_TRUE(result.dataset.empty());
    EXPECT_TRUE(result.failed_days.empty());
}

TEST(HistoricalLoader, UnknownTickerYieldsEmptyDataset) {
    HistoricalLoader::Options opts{};
    opts.root = historical_root();
    const auto result = HistoricalLoader::load("NOSUCH", opts);

    EXPECT_TRUE(result.dataset.empty());
    EXPECT_EQ(result.dataset.ticker(), "NOSUCH");
    EXPECT_TRUE(result.failed_days.empty());
}

TEST(HistoricalLoader, ConvenienceLoadDelegatesToStrict) {
    // Convenience API with default root — should throw on the
    // malformed day if the default root points at our fixture root.
    // We reroute via an explicit Options{} call to be root-safe.
    HistoricalLoader::Options opts{};
    opts.root   = historical_root();
    opts.strict = true;
    // Skip the malformed day for this API-shape check.
    opts.end_date = ymd(2024, 1, 3);
    const auto result = HistoricalLoader::load("SPY", opts);
    EXPECT_EQ(result.dataset.size(), 2u);
    EXPECT_EQ(result.dataset.ticker(), "SPY");
}

// -----------------------------------------------------------------------------
// Dataset ordering & access
// -----------------------------------------------------------------------------

TEST(HistoricalDataset, IterationAndIndexingAgree) {
    HistoricalLoader::Options opts{};
    opts.root     = historical_root();
    opts.end_date = ymd(2024, 1, 3);
    const auto result = HistoricalLoader::load("SPY", opts);
    const auto& dataset = result.dataset;

    ASSERT_EQ(dataset.size(), 2u);

    std::size_t i = 0;
    for (const auto& snap : dataset) {
        EXPECT_EQ(snap.date(), dataset[i].date());
        ++i;
    }
    EXPECT_EQ(i, dataset.size());
}

TEST(HistoricalDataset, EnforcesAscendingDateOrder) {
    // Build a dataset with two snapshots supplied in *reverse* order;
    // the container's constructor must resort them into ascending
    // order.
    auto s1 = HistoricalLoader::load_day("SPY", ymd(2024, 1, 3), historical_root());
    auto s0 = HistoricalLoader::load_day("SPY", ymd(2024, 1, 2), historical_root());
    std::vector<HistoricalSnapshot> reversed;
    reversed.push_back(std::move(s1));
    reversed.push_back(std::move(s0));

    HistoricalDataset dataset("SPY", std::move(reversed));
    ASSERT_EQ(dataset.size(), 2u);
    EXPECT_EQ(dataset[0].date(), ymd(2024, 1, 2));
    EXPECT_EQ(dataset[1].date(), ymd(2024, 1, 3));
    EXPECT_EQ(dataset.first_date(), ymd(2024, 1, 2));
    EXPECT_EQ(dataset.last_date(),  ymd(2024, 1, 3));
}

TEST(HistoricalDataset, AtBoundsCheckedThrows) {
    HistoricalDataset empty("SPY", {});
    EXPECT_THROW(empty.at(0), std::out_of_range);
}

// -----------------------------------------------------------------------------
// Statistics
// -----------------------------------------------------------------------------

TEST(HistoricalDataset, StatsMatchLoadedCounts) {
    HistoricalLoader::Options opts{};
    opts.root     = historical_root();
    opts.end_date = ymd(2024, 1, 3);
    const auto result = HistoricalLoader::load("SPY", opts);
    const auto stats = result.dataset.stats();

    EXPECT_EQ(stats.ticker, "SPY");
    EXPECT_EQ(stats.trading_days, 2u);
    // Both days ship 3 calls + 3 puts.
    EXPECT_EQ(stats.contracts_loaded, 12u);
    EXPECT_DOUBLE_EQ(stats.average_contracts_per_day, 6.0);
    ASSERT_TRUE(stats.first_date.has_value());
    EXPECT_EQ(*stats.first_date, ymd(2024, 1, 2));
    EXPECT_EQ(*stats.last_date,  ymd(2024, 1, 3));
    EXPECT_EQ(stats.parse_failures, 0u);
}

TEST(HistoricalDataset, StatsOfEmptyDatasetAreSafe) {
    HistoricalDataset empty("SPY", {});
    const auto stats = empty.stats();

    EXPECT_EQ(stats.trading_days, 0u);
    EXPECT_EQ(stats.contracts_loaded, 0u);
    EXPECT_DOUBLE_EQ(stats.average_contracts_per_day, 0.0);
    EXPECT_FALSE(stats.first_date.has_value());
    EXPECT_FALSE(stats.last_date.has_value());
}

TEST(HistoricalLoader, MissingDatesReflectWeekendGaps) {
    // 2024-01-02 (Tue), 2024-01-03 (Wed), 2024-01-04 (Thu) are all
    // weekdays. A window ending on 2024-01-08 (Mon) should mark
    // 2024-01-05 (Fri) and 2024-01-08 (Mon) as missing business days
    // — the intervening 6/7 are weekends and are excluded by design.
    HistoricalLoader::Options opts{};
    opts.root       = historical_root();
    opts.start_date = ymd(2024, 1, 2);
    opts.end_date   = ymd(2024, 1, 8);
    opts.strict     = false;
    const auto result = HistoricalLoader::load("SPY", opts);

    // 2024-01-04 is the malformed day; it counts as "present" (a
    // failure), not "missing", per the loader spec.
    ASSERT_GE(result.missing_dates.size(), 2u);
    // The gap should include Fri 2024-01-05 and Mon 2024-01-08.
    const bool has_friday = std::any_of(
        result.missing_dates.begin(), result.missing_dates.end(),
        [](auto d) { return d == ymd(2024, 1, 5); });
    const bool has_monday = std::any_of(
        result.missing_dates.begin(), result.missing_dates.end(),
        [](auto d) { return d == ymd(2024, 1, 8); });
    EXPECT_TRUE(has_friday);
    EXPECT_TRUE(has_monday);
    // Neither Saturday nor Sunday should appear.
    for (const auto& d : result.missing_dates) {
        const auto wd = std::chrono::weekday{std::chrono::sys_days{d}};
        EXPECT_LT(wd.iso_encoding(), 6u) << "Weekend in missing_dates";
    }
}

// -----------------------------------------------------------------------------
// Multiple tickers coexisting
// -----------------------------------------------------------------------------

TEST(HistoricalLoader, MultipleTickersCoexist) {
    // Create a scratch directory with two ticker sub-trees so we can
    // verify the loader treats them as independent datasets. We reuse
    // the shared SPY fixture bytes but rewrite `underlying_symbol` so
    // the copied metadata is internally consistent with each ticker's
    // directory.
    const auto scratch = fs::temp_directory_path() / "ore_historical_multi_test";
    std::error_code ec;
    fs::remove_all(scratch, ec);   // ok if the directory did not exist
    fs::create_directories(scratch);
    struct Cleanup {
        fs::path root;
        ~Cleanup() {
            std::error_code ec2;
            fs::remove_all(root, ec2);
        }
    } cleanup{scratch};

    auto write_ticker = [&](const std::string& tk) {
        const auto src = historical_root() / "SPY" / "2024-01-02";
        const auto dst = scratch / tk / "2024-01-02";
        fs::create_directories(dst);
        fs::copy_file(src / "calls.csv", dst / "calls.csv");
        fs::copy_file(src / "puts.csv",  dst / "puts.csv");

        // Rewrite metadata.csv with the new underlying_symbol.
        std::ifstream in(src / "metadata.csv");
        std::string header, data;
        std::getline(in, header);
        std::getline(in, data);
        // Replace the second CSV column ("SPY") with `tk`.
        // Simplistic: works because tk is uppercase alnum only.
        const auto pos_start = data.find(',') + 1;
        const auto pos_end   = data.find(',', pos_start);
        data = data.substr(0, pos_start) + tk + data.substr(pos_end);

        std::ofstream out(dst / "metadata.csv");
        out << header << '\n' << data << '\n';
    };
    write_ticker("SPY");
    write_ticker("QQQ");

    HistoricalLoader::Options opts{};
    opts.root = scratch;

    const auto spy = HistoricalLoader::load("SPY", opts);
    const auto qqq = HistoricalLoader::load("QQQ", opts);
    EXPECT_EQ(spy.dataset.size(), 1u);
    EXPECT_EQ(qqq.dataset.size(), 1u);
    EXPECT_EQ(spy.dataset[0].underlying().symbol, "SPY");
    EXPECT_EQ(qqq.dataset[0].underlying().symbol, "QQQ");
}

// -----------------------------------------------------------------------------
// End-to-end: pricing on a historical snapshot
// -----------------------------------------------------------------------------

TEST(HistoricalIntegration, SnapshotFeedsBlackScholesUnchanged) {
    // The whole point of the API design: a loaded snapshot must flow
    // into `PricingEngine::price` without any adapter layer.
    const auto snap = HistoricalLoader::load_day(
        "SPY", ymd(2024, 1, 2), historical_root());
    ASSERT_FALSE(snap.empty());
    const auto& option = snap.options().front().option;

    BlackScholesEngine bs{};
    const auto result = bs.price(option, snap.market());
    EXPECT_GT(result.price, 0.0);
}
