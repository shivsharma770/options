#include <gtest/gtest.h>

#include <ore/core/option.hpp>
#include <ore/core/option_market_snapshot.hpp>
#include <ore/core/quote.hpp>
#include <ore/core/underlying.hpp>

#include <chrono>

using ore::core::AssetType;
using ore::core::ExerciseStyle;
using ore::core::Option;
using ore::core::OptionMarketSnapshot;
using ore::core::OptionType;
using ore::core::Quote;
using ore::core::Underlying;

TEST(OptionMarketSnapshotTest, ComposesOptionAndQuote) {
    const Option opt{
        .underlying = Underlying{.symbol = "SPY", .exchange = "NYSE", .asset_type = AssetType::ETF},
        .strike     = 425.0,
        .expiration = std::chrono::year{2026} / std::chrono::June / std::chrono::day{19},
        .type       = OptionType::Put,
        .exercise   = ExerciseStyle::European,
    };
    const Quote q{
        .bid           = 12.10,
        .ask           = 12.30,
        .last          = 12.20,
        .volume        = 542.0,
        .open_interest = 1200.0,
        .timestamp     = std::chrono::system_clock::now(),
    };

    const OptionMarketSnapshot snap{opt, q};

    EXPECT_EQ(snap.option, opt);
    EXPECT_EQ(snap.quote, q);
    EXPECT_DOUBLE_EQ(snap.quote.mid(), 12.20);
}

TEST(QuoteTest, MidIsAverageOfBidAsk) {
    const Quote q{.bid = 1.00, .ask = 1.10};
    EXPECT_DOUBLE_EQ(q.mid(), 1.05);
}
