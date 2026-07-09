#include <gtest/gtest.h>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/option_chain.hpp>
#include <ore/core/option_market_snapshot.hpp>
#include <ore/core/underlying.hpp>

#include <chrono>
#include <vector>

using ore::core::AssetType;
using ore::core::ExerciseStyle;
using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionChain;
using ore::core::OptionMarketSnapshot;
using ore::core::OptionType;
using ore::core::Quote;
using ore::core::Underlying;

namespace {

Underlying make_underlying() {
    return {"SPY", "NYSEARCA", AssetType::ETF};
}

MarketSnapshot make_market() {
    return {472.15,
            0.0525,
            0.0145,
            std::chrono::year{2026} / std::chrono::July / std::chrono::day{8}};
}

OptionMarketSnapshot make_snap(double strike, OptionType type) {
    Option opt{
        .underlying = make_underlying(),
        .strike     = strike,
        .expiration = std::chrono::year{2026} / std::chrono::August / std::chrono::day{8},
        .type       = type,
        .exercise   = ExerciseStyle::European,
    };
    Quote q{
        .bid  = 1.0,
        .ask  = 1.1,
        .last = 1.05,
    };
    return {std::move(opt), std::move(q), "SYMBOL_STUB"};
}

} // namespace

TEST(OptionChainTest, StoresConstituents) {
    std::vector<OptionMarketSnapshot> snaps;
    snaps.push_back(make_snap(470.0, OptionType::Call));
    snaps.push_back(make_snap(475.0, OptionType::Call));
    snaps.push_back(make_snap(470.0, OptionType::Put));

    const OptionChain chain{make_underlying(), make_market(), std::move(snaps)};

    EXPECT_EQ(chain.size(), 3U);
    EXPECT_FALSE(chain.empty());
    EXPECT_EQ(chain.underlying().symbol, "SPY");
    EXPECT_DOUBLE_EQ(chain.market().spot, 472.15);
    EXPECT_EQ(chain.options().size(), 3U);
}

TEST(OptionChainTest, IteratesDirectlyAndViaOptions) {
    std::vector<OptionMarketSnapshot> snaps;
    snaps.push_back(make_snap(470.0, OptionType::Call));
    snaps.push_back(make_snap(475.0, OptionType::Call));

    const OptionChain chain{make_underlying(), make_market(), std::move(snaps)};

    // Direct range-for on the chain itself.
    std::size_t counted = 0;
    for (const auto& s : chain) {
        EXPECT_EQ(s.option.underlying.symbol, "SPY");
        ++counted;
    }
    EXPECT_EQ(counted, 2U);

    // Explicit .options() access, matching the user-facing example.
    std::size_t counted_via_accessor = 0;
    for (const auto& s : chain.options()) {
        (void)s;
        ++counted_via_accessor;
    }
    EXPECT_EQ(counted_via_accessor, 2U);
}

TEST(OptionChainTest, EmptyChainIsValid) {
    const OptionChain chain{make_underlying(), make_market(), {}};
    EXPECT_TRUE(chain.empty());
    EXPECT_EQ(chain.size(), 0U);
    EXPECT_EQ(chain.begin(), chain.end());
}
