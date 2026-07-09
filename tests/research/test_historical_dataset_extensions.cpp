/**
 * @file test_historical_dataset_extensions.cpp
 * @brief Coverage for `HistoricalDataset`'s new accessors
 *        (`front`, `back`, `contains`, `at(date)`, `between`,
 *        `filter`, `for_each`).
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <ore/marketdata/historical_dataset.hpp>
#include <ore/marketdata/historical_snapshot.hpp>

#include "research_test_helpers.hpp"

namespace {

using namespace ore::marketdata;
using namespace std::chrono;

TEST(HistoricalDatasetExtensions, FrontAndBackReturnEndpoints) {
    auto ds = ore::research::testing::make_synthetic_dataset(5);
    ASSERT_EQ(ds.size(), 5u);
    EXPECT_EQ(ds.front().date(), ds.snapshots().front().date());
    EXPECT_EQ(ds.back().date(),  ds.snapshots().back().date());
}

TEST(HistoricalDatasetExtensions, ContainsAndFindAreExact) {
    auto ds = ore::research::testing::make_synthetic_dataset(5);
    const auto d = ds.snapshots()[2].date();

    EXPECT_TRUE(ds.contains(d));
    // A date guaranteed to not be in the set — one day before the
    // start of the range.
    const auto before = year_month_day{sys_days{ds.first_date().value()} - days{1}};
    EXPECT_FALSE(ds.contains(before));

    const auto* p = ds.find(d);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->date(), d);
    EXPECT_EQ(ds.find(before), nullptr);
}

TEST(HistoricalDatasetExtensions, AtByDateThrowsForMissing) {
    auto ds = ore::research::testing::make_synthetic_dataset(3);
    const auto d = ds.snapshots()[1].date();
    EXPECT_NO_THROW({ (void)ds.at(d); });

    const auto missing = year_month_day{sys_days{ds.last_date().value()} + days{1}};
    EXPECT_THROW({ (void)ds.at(missing); }, std::out_of_range);
}

TEST(HistoricalDatasetExtensions, BetweenReturnsContiguousSlice) {
    auto ds = ore::research::testing::make_synthetic_dataset(7);
    const auto first = ds.snapshots()[1].date();
    const auto last  = ds.snapshots()[4].date();

    std::span<const HistoricalSnapshot> view = ds.between(first, last);
    ASSERT_EQ(view.size(), 4u);
    EXPECT_EQ(view.front().date(), first);
    EXPECT_EQ(view.back().date(),  last);

    // The span must alias the underlying vector (zero-copy).
    EXPECT_EQ(view.data(), ds.snapshots().data() + 1);
}

TEST(HistoricalDatasetExtensions, BetweenHandlesEmptyAndInvertedRanges) {
    auto ds = ore::research::testing::make_synthetic_dataset(4);
    const auto first = ds.first_date().value();
    const auto last  = ds.last_date().value();

    // Inverted range → empty span.
    EXPECT_TRUE(ds.between(last, first).empty());

    // Range entirely before the dataset → empty span.
    const auto before1 = year_month_day{sys_days{first} - days{10}};
    const auto before2 = year_month_day{sys_days{first} - days{5}};
    EXPECT_TRUE(ds.between(before1, before2).empty());

    // Range entirely after the dataset → empty span.
    const auto after1 = year_month_day{sys_days{last} + days{5}};
    const auto after2 = year_month_day{sys_days{last} + days{10}};
    EXPECT_TRUE(ds.between(after1, after2).empty());

    // Range covering everything → whole vector.
    auto all = ds.between(before1, after2);
    EXPECT_EQ(all.size(), ds.size());
}

TEST(HistoricalDatasetExtensions, FilterKeepsMatchingSnapshotsOnly) {
    auto ds = ore::research::testing::make_synthetic_dataset(6);
    const auto midpoint = ds.snapshots()[3].date();
    auto later = ds.filter([&](const HistoricalSnapshot& s) {
        return sys_days{s.date()} >= sys_days{midpoint};
    });
    EXPECT_EQ(later.size(), 3u);
    EXPECT_EQ(later.front().date(), midpoint);
    EXPECT_EQ(later.ticker(), ds.ticker());
}

TEST(HistoricalDatasetExtensions, ForEachVisitsInOrder) {
    auto ds = ore::research::testing::make_synthetic_dataset(5);
    std::vector<year_month_day> seen;
    ds.for_each([&](const HistoricalSnapshot& s) { seen.push_back(s.date()); });
    ASSERT_EQ(seen.size(), 5u);
    // Since the dataset sorts on construction, `seen` must be
    // monotone-increasing.
    EXPECT_TRUE(std::is_sorted(seen.begin(), seen.end(),
        [](year_month_day a, year_month_day b) {
            return sys_days{a} < sys_days{b};
        }));
}

TEST(HistoricalDatasetExtensions, EmptyDatasetIsSafe) {
    HistoricalDataset ds{"SPY", {}};
    EXPECT_TRUE(ds.empty());
    EXPECT_FALSE(ds.contains(2024y/1/1));
    EXPECT_EQ(ds.find(2024y/1/1), nullptr);
    EXPECT_TRUE(ds.between(2024y/1/1, 2024y/1/31).empty());
    EXPECT_THROW({ (void)ds.at(2024y/1/1); }, std::out_of_range);

    auto filtered = ds.filter([](const HistoricalSnapshot&) { return true; });
    EXPECT_TRUE(filtered.empty());
}

} // namespace
