#include <ore/marketdata/yahoo_option_loader.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string_view>
#include <unordered_set>

namespace ore::marketdata {

namespace {

using ore::core::AssetType;
using ore::core::MarketSnapshot;
using ore::core::Option;
using ore::core::OptionChain;
using ore::core::OptionMarketSnapshot;
using ore::core::OptionType;
using ore::core::Quote;
using ore::core::Underlying;
using ore::core::ExerciseStyle;
using ore::io::CsvParseError;
using ore::io::CsvRow;
using ore::io::CsvTable;

// ---- Required column lists ------------------------------------------------
const std::vector<std::string> kMetadataColumns = {
    "valuation_date",
    "underlying_symbol",
    "spot",
    "dividend_yield",
    "risk_free_rate",
};

const std::vector<std::string> kContractColumns = {
    "contract_symbol",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "implied_volatility",
    "in_the_money",
    "last_trade_date",
};

// ---- Small helpers --------------------------------------------------------

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

[[noreturn]] void die(const std::string& message,
                      const std::string& source = {},
                      std::size_t line = 0,
                      const std::string& column = {}) {
    std::ostringstream oss;
    oss << message;
    if (!source.empty())  oss << " [source=" << source << "]";
    if (line != 0)        oss << " [line=" << line << "]";
    if (!column.empty())  oss << " [column=" << column << "]";
    throw LoaderError(oss.str());
}

void require_columns(const CsvTable& t, const std::vector<std::string>& cols) {
    for (const auto& c : cols) {
        if (!t.has_column(c)) {
            die("Missing required column: " + c, t.source(), 0, c);
        }
    }
}

std::chrono::year_month_day parse_date(std::string_view raw,
                                       const std::string& source,
                                       std::size_t line,
                                       const std::string& column) {
    // Accept ISO 8601 dates. If a full timestamp is provided
    // ("2026-07-08T15:59:00Z" or "2026-07-08 15:59:00"), we keep only the
    // date portion.
    if (raw.size() < 10) {
        die("Date is too short (expected YYYY-MM-DD)", source, line, column);
    }
    const std::string_view date_part = raw.substr(0, 10);
    if (date_part[4] != '-' || date_part[7] != '-') {
        die("Date is not in YYYY-MM-DD format: " + std::string(raw),
            source, line, column);
    }
    const auto to_int = [&](std::string_view piece) {
        int v{};
        for (char c : piece) {
            if (c < '0' || c > '9') {
                die("Non-numeric character in date: " + std::string(raw),
                    source, line, column);
            }
            v = v * 10 + (c - '0');
        }
        return v;
    };
    const int y = to_int(date_part.substr(0, 4));
    const int m = to_int(date_part.substr(5, 2));
    const int d = to_int(date_part.substr(8, 2));

    std::chrono::year_month_day ymd{std::chrono::year{y},
                                    std::chrono::month{static_cast<unsigned>(m)},
                                    std::chrono::day{static_cast<unsigned>(d)}};
    if (!ymd.ok()) {
        die("Date is not a valid calendar date: " + std::string(raw),
            source, line, column);
    }
    return ymd;
}

std::chrono::system_clock::time_point parse_timestamp(std::string_view raw,
                                                      const std::string& source,
                                                      std::size_t line,
                                                      const std::string& column) {
    // We only need the date portion for `Quote::timestamp` at this stage;
    // the C++ side never sub-day-schedules on option quote times. Truncate
    // to midnight UTC on the parsed date so downstream consumers get a
    // deterministic time_point.
    const auto ymd = parse_date(raw, source, line, column);
    const auto sd  = std::chrono::sys_days{ymd};
    return std::chrono::system_clock::time_point{sd.time_since_epoch()};
}

OptionType parse_option_type(std::string_view raw,
                             const std::string& source,
                             std::size_t line,
                             const std::string& column) {
    const std::string t = to_lower(std::string(raw));
    if (t == "call" || t == "c") return OptionType::Call;
    if (t == "put"  || t == "p") return OptionType::Put;
    die("Invalid option_type (expected 'call' or 'put'): " + std::string(raw),
        source, line, column);
}

AssetType parse_asset_type(std::string_view raw) {
    const std::string t = to_lower(std::string(raw));
    if (t.empty() || t == "equity" || t == "stock") return AssetType::Equity;
    if (t == "index")     return AssetType::Index;
    if (t == "etf")       return AssetType::ETF;
    if (t == "future")    return AssetType::Future;
    if (t == "fx")        return AssetType::FX;
    return AssetType::Other;
}

// Wrap CsvParseError into LoaderError to give callers a single exception
// hierarchy to catch.
double csv_double(const CsvRow& row, std::string_view column, const std::string& source) {
    try {
        return row.as_double(column);
    } catch (const CsvParseError& e) {
        die(e.what(), source, row.line_number(), std::string(column));
    }
}

[[maybe_unused]] long long csv_int(const CsvRow& row, std::string_view column, const std::string& source) {
    try {
        return row.as_int(column);
    } catch (const CsvParseError& e) {
        die(e.what(), source, row.line_number(), std::string(column));
    }
}

bool csv_bool(const CsvRow& row, std::string_view column, const std::string& source) {
    try {
        return row.as_bool(column);
    } catch (const CsvParseError& e) {
        die(e.what(), source, row.line_number(), std::string(column));
    }
}

CsvTable read_or_die(const std::filesystem::path& path) {
    try {
        return CsvTable::read_file(path);
    } catch (const CsvParseError& e) {
        die(e.what());
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

YahooOptionLoader::Columns YahooOptionLoader::columns() {
    return {kMetadataColumns, kContractColumns};
}

YahooOptionLoader::SnapshotMetadata YahooOptionLoader::parse_metadata(const CsvTable& table) {
    require_columns(table, kMetadataColumns);
    if (table.num_rows() != 1) {
        die("metadata.csv must contain exactly one data row (found "
            + std::to_string(table.num_rows()) + ")",
            table.source(), 0, "");
    }
    const CsvRow row = table.row(0);

    Underlying u;
    u.symbol   = std::string(row.field("underlying_symbol"));
    if (u.symbol.empty()) {
        die("Empty underlying_symbol", table.source(), row.line_number(), "underlying_symbol");
    }
    if (row.has("exchange")) {
        u.exchange = std::string(row.field("exchange"));
    }
    if (row.has("asset_type")) {
        u.asset_type = parse_asset_type(row.field("asset_type"));
    }

    MarketSnapshot m;
    m.valuation_date  = parse_date(row.field("valuation_date"),
                                   table.source(), row.line_number(), "valuation_date");
    m.spot            = csv_double(row, "spot", table.source());
    if (!std::isfinite(m.spot) || m.spot <= 0.0) {
        die("spot must be a positive finite number, got " + std::to_string(m.spot),
            table.source(), row.line_number(), "spot");
    }
    m.dividend_yield  = csv_double(row, "dividend_yield", table.source());
    if (!std::isfinite(m.dividend_yield)) {
        die("dividend_yield is not finite", table.source(), row.line_number(), "dividend_yield");
    }
    m.risk_free_rate  = csv_double(row, "risk_free_rate", table.source());
    if (!std::isfinite(m.risk_free_rate)) {
        die("risk_free_rate is not finite", table.source(), row.line_number(), "risk_free_rate");
    }

    return SnapshotMetadata{u, m};
}

std::vector<OptionMarketSnapshot> YahooOptionLoader::parse_contracts(
    const CsvTable& table,
    OptionType expected_type,
    const Underlying& underlying,
    std::chrono::year_month_day valuation_date) {

    require_columns(table, kContractColumns);

    std::vector<OptionMarketSnapshot> out;
    out.reserve(table.num_rows());
    std::unordered_set<std::string> seen_symbols;
    seen_symbols.reserve(table.num_rows());

    for (const CsvRow row : table) {
        OptionMarketSnapshot snap;

        // Contract identifier -----------------------------------------------
        snap.contract_symbol = std::string(row.field("contract_symbol"));
        if (snap.contract_symbol.empty()) {
            die("Empty contract_symbol", table.source(), row.line_number(), "contract_symbol");
        }
        if (!seen_symbols.insert(snap.contract_symbol).second) {
            die("Duplicate contract_symbol: " + snap.contract_symbol,
                table.source(), row.line_number(), "contract_symbol");
        }

        // Contract terms ----------------------------------------------------
        Option& opt = snap.option;
        opt.underlying = underlying;
        opt.strike     = csv_double(row, "strike", table.source());
        if (!std::isfinite(opt.strike) || opt.strike <= 0.0) {
            die("Strike must be a positive finite number, got " + std::to_string(opt.strike),
                table.source(), row.line_number(), "strike");
        }
        opt.expiration = parse_date(row.field("expiration"),
                                    table.source(), row.line_number(), "expiration");
        if (std::chrono::sys_days{opt.expiration} < std::chrono::sys_days{valuation_date}) {
            die("Expiration " + std::string(row.field("expiration"))
                + " is before the valuation date",
                table.source(), row.line_number(), "expiration");
        }
        const OptionType stated_type = parse_option_type(row.field("option_type"),
                                                        table.source(),
                                                        row.line_number(),
                                                        "option_type");
        if (stated_type != expected_type) {
            die("Row's option_type disagrees with the file it was found in",
                table.source(), row.line_number(), "option_type");
        }
        opt.type     = stated_type;
        // ExerciseStyle: Yahoo's option chain export does not carry this
        // field, and equity options (SPY, AAPL, ...) are actually
        // American-style while index options (SPX) are European. Defaulting
        // to European here matches the assumption used by Black-Scholes;
        // once binomial/American pricing lands we will accept an optional
        // `exercise_style` column in `metadata.csv` and honour it here.
        opt.exercise = ExerciseStyle::European;

        // Quote -------------------------------------------------------------
        Quote& q = snap.quote;
        q.bid = csv_double(row, "bid", table.source());
        q.ask = csv_double(row, "ask", table.source());
        q.last = csv_double(row, "last", table.source());
        if (!std::isfinite(q.bid) || q.bid < 0.0) {
            die("bid must be finite and non-negative, got " + std::to_string(q.bid),
                table.source(), row.line_number(), "bid");
        }
        if (!std::isfinite(q.ask) || q.ask < 0.0) {
            die("ask must be finite and non-negative, got " + std::to_string(q.ask),
                table.source(), row.line_number(), "ask");
        }
        // Reject a crossed market. Both-zero is allowed (means "no market").
        if (q.bid > q.ask && q.bid > 0.0) {
            die("Crossed market: bid > ask", table.source(), row.line_number(), "bid");
        }

        // Volume and open-interest are conceptually integers but stored as
        // double on `Quote` so that NaN-for-missing round-trips cleanly.
        // Reading them as double tolerates values like "0.0" (some pandas
        // exports emit that) without loss of information.
        q.volume        = csv_double(row, "volume", table.source());
        q.open_interest = csv_double(row, "open_interest", table.source());
        if (!std::isfinite(q.volume) || q.volume < 0.0) {
            die("Volume must be a finite non-negative number, got " + std::to_string(q.volume),
                table.source(), row.line_number(), "volume");
        }
        if (!std::isfinite(q.open_interest) || q.open_interest < 0.0) {
            die("Open interest must be a finite non-negative number, got " + std::to_string(q.open_interest),
                table.source(), row.line_number(), "open_interest");
        }

        q.timestamp = parse_timestamp(row.field("last_trade_date"),
                                      table.source(), row.line_number(), "last_trade_date");

        // Provider-published implied volatility. Tolerant parsing:
        //   * column missing entirely  -> Quote::implied_volatility = nullopt
        //   * cell empty / NaN         -> nullopt (Yahoo emits this for deep OTM)
        //   * finite non-negative      -> stored on the Quote
        //   * finite negative          -> hard error (physically impossible)
        // A value of 0.0 is genuinely stored — some providers publish 0
        // when they cannot compute IV, and consumers should be able to
        // distinguish "provider says 0" from "provider did not publish".
        if (row.has("implied_volatility")) {
            const double iv = csv_double(row, "implied_volatility", table.source());
            if (std::isfinite(iv)) {
                if (iv < 0.0) {
                    die("Negative implied_volatility", table.source(), row.line_number(), "implied_volatility");
                }
                q.implied_volatility = iv;
            }
        }
        // in_the_money is validated for parseability but not stored on the
        // domain object; the pricing/analytics layer computes moneyness
        // from strike vs. spot itself.
        (void)csv_bool(row, "in_the_money", table.source());

        out.push_back(std::move(snap));
    }

    return out;
}

OptionChain YahooOptionLoader::load(const std::filesystem::path& snapshot_directory) {
    namespace fs = std::filesystem;

    if (!fs::exists(snapshot_directory) || !fs::is_directory(snapshot_directory)) {
        die("Snapshot path does not exist or is not a directory: "
            + snapshot_directory.string());
    }

    const fs::path meta_path  = snapshot_directory / "metadata.csv";
    const fs::path calls_path = snapshot_directory / "calls.csv";
    const fs::path puts_path  = snapshot_directory / "puts.csv";

    for (const auto& p : {meta_path, calls_path, puts_path}) {
        if (!fs::exists(p)) {
            die("Required file is missing from snapshot: " + p.string());
        }
    }

    CsvTable metadata_table = read_or_die(meta_path);
    CsvTable calls_table    = read_or_die(calls_path);
    CsvTable puts_table     = read_or_die(puts_path);

    SnapshotMetadata meta = parse_metadata(metadata_table);

    // Cross-check the directory structure against the metadata. This catches
    // wrong-file-in-wrong-folder mistakes at load time rather than at
    // pricing time. Two levels of check:
    //   1. Leaf directory name must equal the valuation date (always
    //      enforced; tests must name their temp dirs after the date).
    //   2. Underlying symbol must match the grandparent folder, but only
    //      when the canonical <SYMBOL>/options/<date>/ layout is detected.
    //      This lets tests use arbitrary temp roots without triggering the
    //      check.
    {
        const std::string dir_name = snapshot_directory.filename().string();
        std::ostringstream ymd_str;
        ymd_str << static_cast<int>(meta.market.valuation_date.year()) << "-"
                << std::setw(2) << std::setfill('0')
                << static_cast<unsigned>(meta.market.valuation_date.month()) << "-"
                << std::setw(2) << std::setfill('0')
                << static_cast<unsigned>(meta.market.valuation_date.day());
        if (dir_name != ymd_str.str()) {
            die("Directory name '" + dir_name +
                "' does not match metadata valuation_date '" + ymd_str.str() + "'");
        }

        const fs::path options_dir = snapshot_directory.parent_path();
        if (options_dir.filename() == "options") {
            const std::string parent_symbol = options_dir.parent_path().filename().string();
            if (!parent_symbol.empty() && parent_symbol != meta.underlying.symbol) {
                die("Snapshot parent directory '" + parent_symbol +
                    "' does not match metadata underlying_symbol '"
                    + meta.underlying.symbol + "'");
            }
        }
    }

    std::vector<OptionMarketSnapshot> options = parse_contracts(
        calls_table, OptionType::Call, meta.underlying, meta.market.valuation_date);
    std::vector<OptionMarketSnapshot> puts = parse_contracts(
        puts_table, OptionType::Put, meta.underlying, meta.market.valuation_date);
    options.insert(options.end(),
                   std::make_move_iterator(puts.begin()),
                   std::make_move_iterator(puts.end()));

    return OptionChain{std::move(meta.underlying),
                       std::move(meta.market),
                       std::move(options)};
}

} // namespace ore::marketdata
