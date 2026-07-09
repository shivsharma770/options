#include <gtest/gtest.h>

#include <ore/core/underlying.hpp>

using ore::core::AssetType;
using ore::core::Underlying;

TEST(UnderlyingTest, DefaultsToEmptyEquity) {
    const Underlying u{};
    EXPECT_TRUE(u.symbol.empty());
    EXPECT_TRUE(u.exchange.empty());
    EXPECT_EQ(u.asset_type, AssetType::Equity);
}

TEST(UnderlyingTest, StoresFields) {
    const Underlying u{.symbol = "AAPL", .exchange = "NASDAQ", .asset_type = AssetType::Equity};
    EXPECT_EQ(u.symbol, "AAPL");
    EXPECT_EQ(u.exchange, "NASDAQ");
    EXPECT_EQ(u.asset_type, AssetType::Equity);
}

TEST(UnderlyingTest, StructuralEquality) {
    const Underlying a{.symbol = "AAPL", .exchange = "NASDAQ", .asset_type = AssetType::Equity};
    Underlying       b = a;
    EXPECT_EQ(a, b);

    b        = a;
    b.symbol = "MSFT";
    EXPECT_NE(a, b);

    b            = a;
    b.exchange   = "OTHER";
    EXPECT_NE(a, b);

    b            = a;
    b.asset_type = AssetType::Index;
    EXPECT_NE(a, b);
}
