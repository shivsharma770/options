#include <gtest/gtest.h>

#include <ore/core/option_chain.hpp>
#include <ore/io/csv_reader.hpp>
#include <ore/marketdata/yahoo_option_loader.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <string_view>

#ifndef ORE_TEST_FIXTURES_DIR
#error "ORE_TEST_FIXTURES_DIR must be defined by the build system"
#endif

using ore::core::AssetType;
using ore::core::OptionChain;
using ore::core::OptionType;
using ore::core::Underlying;
using ore::io::CsvTable;
using ore::marketdata::LoaderError;
using ore::marketdata::YahooOptionLoader;

namespace {

std::filesystem::path fixture_snapshot() {
    return std::filesystem::path{ORE_TEST_FIXTURES_DIR}
        / "marketdata" / "SPY" / "options" / "2026-07-08";
}

// A small helper for building in-memory CSV tables in tests.
CsvTable make_table(std::string_view text) {
    return CsvTable::parse_string(text, {}, "<test>");
}

constexpr std::string_view kValidMetadata =
    "valuation_date,underlying_symbol,exchange,asset_type,spot,dividend_yield,risk_free_rate\n"
    "2026-07-08,SPY,NYSEARCA,ETF,472.15,0.0145,0.0525\n";

constexpr std::string_view kValidCallsHeader =
    "contract_symbol,expiration,strike,option_type,bid,ask,last,volume,"
    "open_interest,implied_volatility,in_the_money,last_trade_date\n";

std::string valid_calls_body() {
    return "SPY260808C00470000,2026-08-08,470.0,call,7.10,7.25,7.18,4321,9876,0.1180,true,2026-07-08\n"
           "SPY260808C00475000,2026-08-08,475.0,call,3.55,3.65,3.60,2100,4500,0.1152,false,2026-07-08\n";
}

} // namespace

// -----------------------------------------------------------------------------
// End-to-end load() from disk fixture.
// -----------------------------------------------------------------------------

TEST(YahooOptionLoaderTest, LoadsFullSnapshotFromDisk) {
    const OptionChain chain = YahooOptionLoader::load(fixture_snapshot());

    EXPECT_EQ(chain.underlying().symbol, "SPY");
    EXPECT_EQ(chain.underlying().exchange, "NYSEARCA");
    EXPECT_EQ(chain.underlying().asset_type, AssetType::ETF);

    EXPECT_DOUBLE_EQ(chain.market().spot, 472.15);
    EXPECT_DOUBLE_EQ(chain.market().dividend_yield, 0.0145);
    EXPECT_DOUBLE_EQ(chain.market().risk_free_rate, 0.0525);
    EXPECT_EQ(chain.market().valuation_date,
              std::chrono::year{2026} / std::chrono::July / std::chrono::day{8});

    // 6 calls + 5 puts in the fixture files.
    EXPECT_EQ(chain.size(), 11U);

    const auto calls = std::count_if(chain.begin(), chain.end(),
        [](const auto& s) { return s.option.type == OptionType::Call; });
    const auto puts = std::count_if(chain.begin(), chain.end(),
        [](const auto& s) { return s.option.type == OptionType::Put; });
    EXPECT_EQ(calls, 6);
    EXPECT_EQ(puts, 5);

    // Every contract carries the underlying identity.
    for (const auto& s : chain) {
        EXPECT_EQ(s.option.underlying.symbol, "SPY");
        EXPECT_GT(s.option.strike, 0.0);
        EXPECT_FALSE(s.contract_symbol.empty());
    }
}

TEST(YahooOptionLoaderTest, LoadThrowsOnMissingDirectory) {
    EXPECT_THROW(YahooOptionLoader::load("no_such_directory_exists_here_hopefully"),
                 LoaderError);
}

TEST(YahooOptionLoaderTest, ExposesRequiredColumnNames) {
    const auto cols = YahooOptionLoader::columns();
    EXPECT_FALSE(cols.metadata.empty());
    EXPECT_FALSE(cols.contract.empty());
    EXPECT_NE(std::find(cols.metadata.begin(), cols.metadata.end(),
                        std::string{"valuation_date"}),
              cols.metadata.end());
    EXPECT_NE(std::find(cols.contract.begin(), cols.contract.end(),
                        std::string{"contract_symbol"}),
              cols.contract.end());
}

// -----------------------------------------------------------------------------
// parse_metadata — happy path & every hard-validation rule.
// -----------------------------------------------------------------------------

TEST(YahooOptionLoaderTest, ParseMetadataHappyPath) {
    const auto table = make_table(kValidMetadata);
    const auto meta = YahooOptionLoader::parse_metadata(table);
    EXPECT_EQ(meta.underlying.symbol, "SPY");
    EXPECT_EQ(meta.underlying.exchange, "NYSEARCA");
    EXPECT_EQ(meta.underlying.asset_type, AssetType::ETF);
    EXPECT_DOUBLE_EQ(meta.market.spot, 472.15);
}

TEST(YahooOptionLoaderTest, ParseMetadataRejectsEmpty) {
    const auto table = make_table(
        "valuation_date,underlying_symbol,spot,dividend_yield,risk_free_rate\n");
    EXPECT_THROW(YahooOptionLoader::parse_metadata(table), LoaderError);
}

TEST(YahooOptionLoaderTest, ParseMetadataRejectsMissingColumn) {
    const auto table = make_table(
        "valuation_date,underlying_symbol,dividend_yield,risk_free_rate\n"
        "2026-07-08,SPY,0.01,0.05\n");
    EXPECT_THROW(YahooOptionLoader::parse_metadata(table), LoaderError);
}

TEST(YahooOptionLoaderTest, ParseMetadataRejectsNegativeSpot) {
    const auto table = make_table(
        "valuation_date,underlying_symbol,spot,dividend_yield,risk_free_rate\n"
        "2026-07-08,SPY,-1.0,0.01,0.05\n");
    EXPECT_THROW(YahooOptionLoader::parse_metadata(table), LoaderError);
}

TEST(YahooOptionLoaderTest, ParseMetadataRejectsBadDate) {
    const auto table = make_table(
        "valuation_date,underlying_symbol,spot,dividend_yield,risk_free_rate\n"
        "2026/07/08,SPY,472.0,0.01,0.05\n");
    EXPECT_THROW(YahooOptionLoader::parse_metadata(table), LoaderError);
}

// -----------------------------------------------------------------------------
// parse_contracts — happy path + every hard-validation rule.
// -----------------------------------------------------------------------------

TEST(YahooOptionLoaderTest, ParseContractsHappyPath) {
    const auto table = make_table(std::string(kValidCallsHeader) + valid_calls_body());
    const Underlying u{.symbol = "SPY"};
    const auto valuation = std::chrono::year{2026}/7/8;
    const auto out = YahooOptionLoader::parse_contracts(table, OptionType::Call, u, valuation);

    ASSERT_EQ(out.size(), 2U);
    EXPECT_EQ(out[0].contract_symbol, "SPY260808C00470000");
    EXPECT_DOUBLE_EQ(out[0].option.strike, 470.0);
    EXPECT_EQ(out[0].option.type, OptionType::Call);
    EXPECT_EQ(out[0].option.underlying.symbol, "SPY");
    EXPECT_DOUBLE_EQ(out[0].quote.bid, 7.10);
    EXPECT_DOUBLE_EQ(out[0].quote.ask, 7.25);
    // Provider IV is now stored on the Quote (added in M6 so the
    // analytics module can compare against our computed IV).
    ASSERT_TRUE(out[0].quote.implied_volatility.has_value());
    EXPECT_DOUBLE_EQ(*out[0].quote.implied_volatility, 0.1180);
}

TEST(YahooOptionLoaderTest, ParseContractsRejectsWrongOptionType) {
    // File body is calls, but we ask parse_contracts to expect Puts. This
    // simulates loading calls.csv into the put slot by mistake.
    const auto table = make_table(std::string(kValidCallsHeader) + valid_calls_body());
    const Underlying u{.symbol = "SPY"};
    const auto valuation = std::chrono::year{2026}/7/8;
    EXPECT_THROW(YahooOptionLoader::parse_contracts(table, OptionType::Put, u, valuation),
                 LoaderError);
}

TEST(YahooOptionLoaderTest, ParseContractsRejectsExpiredContract) {
    // Expiration 2026-01-01 is before valuation 2026-07-08.
    const auto text = std::string(kValidCallsHeader) +
        "SPY260101C00470000,2026-01-01,470.0,call,7.10,7.25,7.18,4321,9876,0.1180,true,2026-01-01\n";
    const auto table = make_table(text);
    const Underlying u{.symbol = "SPY"};
    const auto valuation = std::chrono::year{2026}/7/8;
    EXPECT_THROW(YahooOptionLoader::parse_contracts(table, OptionType::Call, u, valuation),
                 LoaderError);
}

TEST(YahooOptionLoaderTest, ParseContractsRejectsCrossedMarket) {
    // bid=8.00 > ask=7.00 and both positive => crossed market.
    const auto text = std::string(kValidCallsHeader) +
        "SPY260808C00470000,2026-08-08,470.0,call,8.00,7.00,7.50,100,200,0.10,true,2026-07-08\n";
    const auto table = make_table(text);
    const Underlying u{.symbol = "SPY"};
    const auto valuation = std::chrono::year{2026}/7/8;
    EXPECT_THROW(YahooOptionLoader::parse_contracts(table, OptionType::Call, u, valuation),
                 LoaderError);
}

TEST(YahooOptionLoaderTest, ParseContractsAllowsZeroBidZeroAsk) {
    // "No market" (bid=ask=0) is legal — deep OTM contracts often show this.
    const auto text = std::string(kValidCallsHeader) +
        "SPY260808C00700000,2026-08-08,700.0,call,0.00,0.00,0.00,0,0,0.0,false,2026-07-08\n";
    const auto table = make_table(text);
    const Underlying u{.symbol = "SPY"};
    const auto valuation = std::chrono::year{2026}/7/8;
    const auto out = YahooOptionLoader::parse_contracts(table, OptionType::Call, u, valuation);
    ASSERT_EQ(out.size(), 1U);
    EXPECT_DOUBLE_EQ(out[0].quote.bid, 0.0);
    EXPECT_DOUBLE_EQ(out[0].quote.ask, 0.0);
}

TEST(YahooOptionLoaderTest, ParseContractsRejectsNegativeStrike) {
    const auto text = std::string(kValidCallsHeader) +
        "SPY260808C_NEG,2026-08-08,-1.0,call,1.0,1.1,1.05,10,20,0.10,false,2026-07-08\n";
    const auto table = make_table(text);
    EXPECT_THROW(YahooOptionLoader::parse_contracts(table, OptionType::Call,
                                                   Underlying{.symbol = "SPY"},
                                                   std::chrono::year{2026}/7/8),
                 LoaderError);
}

TEST(YahooOptionLoaderTest, ParseContractsRejectsNegativeBid) {
    const auto text = std::string(kValidCallsHeader) +
        "SPY260808C00470000,2026-08-08,470.0,call,-0.01,1.10,1.05,10,20,0.10,true,2026-07-08\n";
    const auto table = make_table(text);
    EXPECT_THROW(YahooOptionLoader::parse_contracts(table, OptionType::Call,
                                                   Underlying{.symbol = "SPY"},
                                                   std::chrono::year{2026}/7/8),
                 LoaderError);
}

TEST(YahooOptionLoaderTest, ParseContractsRejectsDuplicateSymbol) {
    const auto text = std::string(kValidCallsHeader) +
        "SPY260808C00470000,2026-08-08,470.0,call,1.0,1.1,1.05,10,20,0.10,true,2026-07-08\n"
        "SPY260808C00470000,2026-08-08,470.0,call,1.0,1.1,1.05,10,20,0.10,true,2026-07-08\n";
    const auto table = make_table(text);
    EXPECT_THROW(YahooOptionLoader::parse_contracts(table, OptionType::Call,
                                                   Underlying{.symbol = "SPY"},
                                                   std::chrono::year{2026}/7/8),
                 LoaderError);
}

TEST(YahooOptionLoaderTest, ParseContractsRejectsMissingColumns) {
    const auto text =
        "contract_symbol,strike,bid,ask\n"
        "SPY,470.0,1.0,1.1\n";
    const auto table = make_table(text);
    EXPECT_THROW(YahooOptionLoader::parse_contracts(table, OptionType::Call,
                                                   Underlying{.symbol = "SPY"},
                                                   std::chrono::year{2026}/7/8),
                 LoaderError);
}

TEST(YahooOptionLoaderTest, ParseContractsRejectsInvalidOptionType) {
    const auto text = std::string(kValidCallsHeader) +
        "SPY260808X00470000,2026-08-08,470.0,banana,1.0,1.1,1.05,10,20,0.10,true,2026-07-08\n";
    const auto table = make_table(text);
    EXPECT_THROW(YahooOptionLoader::parse_contracts(table, OptionType::Call,
                                                   Underlying{.symbol = "SPY"},
                                                   std::chrono::year{2026}/7/8),
                 LoaderError);
}

TEST(YahooOptionLoaderTest, ParseContractsRejectsInvalidDate) {
    const auto text = std::string(kValidCallsHeader) +
        "SPY260808C00470000,2026-13-40,470.0,call,1.0,1.1,1.05,10,20,0.10,true,2026-07-08\n";
    const auto table = make_table(text);
    EXPECT_THROW(YahooOptionLoader::parse_contracts(table, OptionType::Call,
                                                   Underlying{.symbol = "SPY"},
                                                   std::chrono::year{2026}/7/8),
                 LoaderError);
}
