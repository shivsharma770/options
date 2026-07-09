#include <gtest/gtest.h>

#include <ore/core/option.hpp>
#include <ore/core/underlying.hpp>

#include <chrono>

using ore::core::AssetType;
using ore::core::ExerciseStyle;
using ore::core::Option;
using ore::core::OptionType;
using ore::core::to_string;
using ore::core::Underlying;

namespace {

Option make_sample_call() {
    return Option{
        .underlying = Underlying{.symbol = "AAPL",
                                 .exchange = "NASDAQ",
                                 .asset_type = AssetType::Equity},
        .strike     = 100.0,
        .expiration = std::chrono::year{2026} / std::chrono::December / std::chrono::day{18},
        .type       = OptionType::Call,
        .exercise   = ExerciseStyle::European,
    };
}

} // namespace

TEST(OptionTest, StoresContractFields) {
    const Option opt = make_sample_call();
    EXPECT_EQ(opt.underlying.symbol, "AAPL");
    EXPECT_EQ(opt.underlying.exchange, "NASDAQ");
    EXPECT_EQ(opt.underlying.asset_type, AssetType::Equity);
    EXPECT_DOUBLE_EQ(opt.strike, 100.0);
    EXPECT_EQ(opt.type, OptionType::Call);
    EXPECT_EQ(opt.exercise, ExerciseStyle::European);
    EXPECT_EQ(opt.expiration.year(), std::chrono::year{2026});
    EXPECT_EQ(opt.expiration.month(), std::chrono::December);
    EXPECT_EQ(opt.expiration.day(), std::chrono::day{18});
}

TEST(OptionTest, StructuralEquality) {
    const Option a = make_sample_call();
    Option       b = a;
    EXPECT_EQ(a, b);

    b        = a;
    b.strike = 110.0;
    EXPECT_NE(a, b);

    b      = a;
    b.type = OptionType::Put;
    EXPECT_NE(a, b);

    b          = a;
    b.exercise = ExerciseStyle::American;
    EXPECT_NE(a, b);

    b                    = a;
    b.underlying.symbol  = "MSFT";
    EXPECT_NE(a, b);
}

TEST(OptionTest, EnumStringification) {
    EXPECT_EQ(to_string(OptionType::Call), "Call");
    EXPECT_EQ(to_string(OptionType::Put), "Put");
    EXPECT_EQ(to_string(ExerciseStyle::European), "European");
    EXPECT_EQ(to_string(ExerciseStyle::American), "American");
}

TEST(OptionTest, TimeToMaturityIsDerivedNotStored) {
    // Sanity check that we can compute year fraction from expiration -
    // valuation date without any dedicated field on `Option`. Real day-count
    // logic will live in ore::utils::day_count later; this test exists to
    // pin the design choice.
    const auto valuation  = std::chrono::sys_days{std::chrono::year{2026} / std::chrono::June / std::chrono::day{18}};
    const auto expiration = std::chrono::sys_days{std::chrono::year{2026} / std::chrono::December / std::chrono::day{18}};
    const auto days       = (expiration - valuation).count();
    EXPECT_GT(days, 0);
    const double year_fraction_act365f = static_cast<double>(days) / 365.0;
    EXPECT_GT(year_fraction_act365f, 0.0);
    EXPECT_LT(year_fraction_act365f, 1.0);
}
