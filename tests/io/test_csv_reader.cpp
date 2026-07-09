#include <gtest/gtest.h>

#include <ore/io/csv_reader.hpp>

#include <string>
#include <string_view>

using ore::io::CsvOptions;
using ore::io::CsvParseError;
using ore::io::CsvTable;

namespace {

CsvTable parse(std::string_view text, CsvOptions opts = {}) {
    return CsvTable::parse_string(text, opts, "<test>");
}

} // namespace

TEST(CsvReaderTest, ParsesHeaderAndRows) {
    const auto table = parse(
        "a,b,c\n"
        "1,2,3\n"
        "4,5,6\n");
    ASSERT_EQ(table.num_rows(), 2U);
    ASSERT_EQ(table.columns().size(), 3U);
    EXPECT_EQ(table.columns()[0], "a");
    EXPECT_EQ(table.columns()[1], "b");
    EXPECT_EQ(table.columns()[2], "c");

    EXPECT_EQ(table.row(0).field("a"), "1");
    EXPECT_EQ(table.row(0).field("b"), "2");
    EXPECT_EQ(table.row(0).field("c"), "3");
    EXPECT_EQ(table.row(1).field("a"), "4");
}

TEST(CsvReaderTest, TypedAccessors) {
    const auto table = parse(
        "name,price,qty,flag\n"
        "AAPL,192.75,1000,true\n"
        "MSFT, 415.10 , 250 , FALSE\n");
    EXPECT_DOUBLE_EQ(table.row(0).as_double("price"), 192.75);
    EXPECT_DOUBLE_EQ(table.row(1).as_double("price"), 415.10);
    EXPECT_EQ(table.row(0).as_int("qty"), 1000);
    EXPECT_EQ(table.row(1).as_int("qty"), 250);
    EXPECT_TRUE(table.row(0).as_bool("flag"));
    EXPECT_FALSE(table.row(1).as_bool("flag"));
}

TEST(CsvReaderTest, QuotedFieldsWithEmbeddedDelimiterAndEscapedQuotes) {
    const auto table = parse(
        "name,description\n"
        "AAPL,\"a, comma-separated, phrase\"\n"
        "MSFT,\"quoth: \"\"hello\"\"\"\n");
    EXPECT_EQ(table.row(0).field("description"), "a, comma-separated, phrase");
    EXPECT_EQ(table.row(1).field("description"), "quoth: \"hello\"");
}

TEST(CsvReaderTest, HandlesCrlfLineEndings) {
    const auto table = parse("a,b\r\n1,2\r\n3,4\r\n");
    EXPECT_EQ(table.num_rows(), 2U);
    EXPECT_EQ(table.row(0).field("a"), "1");
    EXPECT_EQ(table.row(1).field("b"), "4");
}

TEST(CsvReaderTest, HandlesUtf8Bom) {
    // \xEF\xBB\xBF is the UTF-8 byte-order mark. Without stripping it the
    // first column name would be "\xEFBOMa" and lookups would fail.
    const std::string bom = "\xEF\xBB\xBF";
    const auto table = parse(bom + "a,b\n1,2\n");
    EXPECT_TRUE(table.has_column("a"));
    EXPECT_EQ(table.row(0).field("a"), "1");
}

TEST(CsvReaderTest, SkipsBlankLines) {
    const auto table = parse("a,b\n\n1,2\n\n3,4\n\n");
    EXPECT_EQ(table.num_rows(), 2U);
}

TEST(CsvReaderTest, MissingFieldsPadToHeaderWidth) {
    const auto table = parse("a,b,c\n1,2\n");
    EXPECT_EQ(table.row(0).field("a"), "1");
    EXPECT_EQ(table.row(0).field("b"), "2");
    EXPECT_EQ(table.row(0).field("c"), "");
    EXPECT_FALSE(table.row(0).has("c"));
}

TEST(CsvReaderTest, ThrowsOnExtraFields) {
    EXPECT_THROW(parse("a,b\n1,2,3\n"), CsvParseError);
}

TEST(CsvReaderTest, ThrowsOnDuplicateHeader) {
    EXPECT_THROW(parse("a,b,a\n1,2,3\n"), CsvParseError);
}

TEST(CsvReaderTest, ThrowsOnBlankHeader) {
    EXPECT_THROW(parse("a,,c\n1,2,3\n"), CsvParseError);
}

TEST(CsvReaderTest, ThrowsOnUnterminatedQuote) {
    EXPECT_THROW(parse("a,b\n\"unterminated,2\n"), CsvParseError);
}

TEST(CsvReaderTest, ThrowsOnEmptyFile) {
    EXPECT_THROW(parse(""), CsvParseError);
}

TEST(CsvReaderTest, ThrowsOnUnknownColumnLookup) {
    const auto table = parse("a,b\n1,2\n");
    EXPECT_THROW(table.row(0).field("c"), CsvParseError);
}

TEST(CsvReaderTest, ThrowsOnNonNumericAsDouble) {
    const auto table = parse("x\nhello\n");
    EXPECT_THROW(table.row(0).as_double("x"), CsvParseError);
}

TEST(CsvReaderTest, ThrowsOnEmptyFieldAsDouble) {
    const auto table = parse("x,y\n,2\n");
    EXPECT_THROW(table.row(0).as_double("x"), CsvParseError);
}

TEST(CsvReaderTest, ThrowsOnNonBooleanAsBool) {
    const auto table = parse("x\nmaybe\n");
    EXPECT_THROW(table.row(0).as_bool("x"), CsvParseError);
}

TEST(CsvReaderTest, PreservesRowLineNumbers) {
    // Header on line 1, blank line 2, data on lines 3 and 4.
    const auto table = parse("a\n\n1\n2\n");
    ASSERT_EQ(table.num_rows(), 2U);
    EXPECT_EQ(table.row(0).line_number(), 3U);
    EXPECT_EQ(table.row(1).line_number(), 4U);
}

TEST(CsvReaderTest, HasCheckReturnsFalseForEmptyField) {
    const auto table = parse("a,b\n1,\n");
    EXPECT_TRUE(table.row(0).has("a"));
    EXPECT_FALSE(table.row(0).has("b"));
    EXPECT_FALSE(table.row(0).has("nonexistent"));
}

TEST(CsvReaderTest, IteratorSupport) {
    const auto table = parse("v\n10\n20\n30\n");
    long long total = 0;
    for (const auto& row : table) {
        total += row.as_int("v");
    }
    EXPECT_EQ(total, 60);
}
