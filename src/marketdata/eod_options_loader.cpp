#include <ore/marketdata/eod_options_loader.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/option_chain.hpp>
#include <ore/core/option_market_snapshot.hpp>
#include <ore/core/provider_greeks.hpp>
#include <ore/core/quote.hpp>
#include <ore/core/types.hpp>
#include <ore/core/underlying.hpp>
#include <ore/marketdata/historical_snapshot.hpp>
#include <ore/marketdata/yahoo_option_loader.hpp>

namespace ore::marketdata {

namespace {

namespace fs = std::filesystem;

/// Trim ASCII whitespace and the OptionsDX bracket wrappers from a
/// token. The archives quote column names as `[QUOTE_DATE]` and pad
/// every data field with a leading space (`, 0.883040`). Both are
/// removed at parse time so downstream code can look up columns and
/// parse numerics without further ceremony.
[[nodiscard]] std::string_view trim_field(std::string_view s) noexcept {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t'
                          || s.front() == '\r' || s.front() == '\n')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back()  == ' ' || s.back()  == '\t'
                          || s.back()  == '\r' || s.back()  == '\n')) {
        s.remove_suffix(1);
    }
    if (!s.empty() && s.front() == '[' && s.back() == ']') {
        s.remove_prefix(1);
        s.remove_suffix(1);
    }
    return s;
}

/// Split a single CSV line into field views. Non-allocating: the
/// returned string_views point into `line`, which the caller must keep
/// alive. Quoted fields are not expected in EOD archives so we do not
/// attempt to support them here.
void split_line(std::string_view line,
                std::vector<std::string_view>& out) {
    out.clear();
    std::size_t start = 0;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == ',') {
            out.push_back(line.substr(start, i - start));
            start = i + 1;
        }
    }
    out.push_back(line.substr(start));
}

/// Parse a `YYYY-MM-DD` calendar date. Returns `nullopt` on any
/// structural failure. Deliberately non-throwing so the caller can
/// skip malformed rows in lenient mode.
[[nodiscard]] std::optional<std::chrono::year_month_day>
parse_iso_date(std::string_view s) {
    s = trim_field(s);
    if (s.size() != 10 || s[4] != '-' || s[7] != '-') return std::nullopt;
    auto to_int = [](std::string_view piece) -> std::optional<int> {
        int v = 0;
        for (char c : piece) {
            if (c < '0' || c > '9') return std::nullopt;
            v = v * 10 + (c - '0');
        }
        return v;
    };
    const auto y = to_int(s.substr(0, 4));
    const auto m = to_int(s.substr(5, 2));
    const auto d = to_int(s.substr(8, 2));
    if (!y || !m || !d) return std::nullopt;
    std::chrono::year_month_day ymd{
        std::chrono::year{*y},
        std::chrono::month{static_cast<unsigned>(*m)},
        std::chrono::day  {static_cast<unsigned>(*d)},
    };
    if (!ymd.ok()) return std::nullopt;
    return ymd;
}

/// Parse an optional double. Empty / whitespace-only fields return
/// `nullopt`; non-numeric text returns `nullopt` in lenient mode
/// (the parser is called on huge archives where a stray dash in the
/// vendor's export is not worth failing over). `std::from_chars` is
/// used to keep this call allocation-free.
[[nodiscard]] std::optional<double> parse_optional_double(std::string_view s) {
    s = trim_field(s);
    if (s.empty()) return std::nullopt;
    double value{};
    const char* first = s.data();
    const char* last  = s.data() + s.size();
    const auto res = std::from_chars(first, last, value);
    if (res.ec != std::errc{} || res.ptr != last) return std::nullopt;
    return value;
}

/// Parse a required double. Missing / malformed fields throw when
/// the caller runs in strict mode; lenient mode returns `nullopt`
/// and the row is skipped upstream.
[[nodiscard]] std::optional<double> parse_double(std::string_view s) {
    return parse_optional_double(s);
}

/// Column-index lookup built from the header line. Case-insensitive
/// on the column name because upstream archives are inconsistent
/// about capitalisation across years (`[C_IV]` vs `[c_iv]`).
struct HeaderIndex {
    std::unordered_map<std::string, std::size_t> map;

    [[nodiscard]] std::optional<std::size_t> lookup(std::string_view name) const {
        std::string key(name);
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char c) { return std::toupper(c); });
        const auto it = map.find(key);
        if (it == map.end()) return std::nullopt;
        return it->second;
    }

    [[nodiscard]] bool has(std::string_view name) const {
        return lookup(name).has_value();
    }
};

/// Build a `HeaderIndex` from a header line. Columns whose name is
/// missing (empty after trimming) are silently ignored — some archives
/// have a trailing empty column produced by a stray comma at end-of-
/// line. The `required` list is the minimum set the parser needs; any
/// missing name throws `LoaderError`.
[[nodiscard]] HeaderIndex build_header_index(
    std::string_view header_line,
    const std::vector<std::string>& required,
    std::string_view source)
{
    HeaderIndex idx;
    std::vector<std::string_view> fields;
    split_line(header_line, fields);
    for (std::size_t i = 0; i < fields.size(); ++i) {
        std::string key(trim_field(fields[i]));
        if (key.empty()) continue;
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char c) { return std::toupper(c); });
        idx.map.emplace(std::move(key), i);
    }
    for (const auto& r : required) {
        if (!idx.has(r)) {
            std::ostringstream oss;
            oss << "EOD archive '" << source
                << "' is missing required column '" << r << "'";
            throw LoaderError(oss.str());
        }
    }
    return idx;
}

/// Column-name catalogue for the OptionsDX schema. Kept in one place
/// so tests and the loader agree; renamed columns can be tolerated by
/// updating this table without touching the parsing loop.
struct EodColumns {
    static constexpr const char* quote_date      = "QUOTE_DATE";
    static constexpr const char* underlying_last = "UNDERLYING_LAST";
    static constexpr const char* expire_date     = "EXPIRE_DATE";
    static constexpr const char* strike          = "STRIKE";

    static constexpr const char* c_bid   = "C_BID";
    static constexpr const char* c_ask   = "C_ASK";
    static constexpr const char* c_last  = "C_LAST";
    static constexpr const char* c_iv    = "C_IV";
    static constexpr const char* c_vol   = "C_VOLUME";
    static constexpr const char* c_delta = "C_DELTA";
    static constexpr const char* c_gamma = "C_GAMMA";
    static constexpr const char* c_theta = "C_THETA";
    static constexpr const char* c_vega  = "C_VEGA";
    static constexpr const char* c_rho   = "C_RHO";

    static constexpr const char* p_bid   = "P_BID";
    static constexpr const char* p_ask   = "P_ASK";
    static constexpr const char* p_last  = "P_LAST";
    static constexpr const char* p_iv    = "P_IV";
    static constexpr const char* p_vol   = "P_VOLUME";
    static constexpr const char* p_delta = "P_DELTA";
    static constexpr const char* p_gamma = "P_GAMMA";
    static constexpr const char* p_theta = "P_THETA";
    static constexpr const char* p_vega  = "P_VEGA";
    static constexpr const char* p_rho   = "P_RHO";
};

/// Look up a numeric field by column name. Returns `nullopt` when the
/// column is present but the field is empty; the caller decides
/// whether that is fatal.
[[nodiscard]] std::optional<double>
field_opt(const HeaderIndex& idx,
          const std::vector<std::string_view>& row,
          const char* column)
{
    const auto pos = idx.lookup(column);
    if (!pos.has_value() || *pos >= row.size()) return std::nullopt;
    return parse_optional_double(row[*pos]);
}

/// Look up a string field. Returns an empty view if the column is
/// absent — mirrors `field_opt`'s "silent-if-missing" semantics.
[[nodiscard]] std::string_view
field_str(const HeaderIndex& idx,
          const std::vector<std::string_view>& row,
          const char* column)
{
    const auto pos = idx.lookup(column);
    if (!pos.has_value() || *pos >= row.size()) return {};
    return trim_field(row[*pos]);
}

/// Build a single-side `OptionMarketSnapshot` (call or put) from a
/// parsed EOD row. Returns `nullopt` if the row is malformed in a way
/// the caller asked to be skipped (see `Options`).
[[nodiscard]] std::optional<ore::core::OptionMarketSnapshot>
build_side(const HeaderIndex& idx,
           const std::vector<std::string_view>& row,
           ore::core::OptionType side,
           const ore::core::Underlying& underlying,
           std::chrono::year_month_day quote_date,
           std::chrono::year_month_day expire_date,
           double strike,
           const EodOptionsLoader::Options& options)
{
    const bool call = (side == ore::core::OptionType::Call);
    const auto bid   = field_opt(idx, row, call ? EodColumns::c_bid   : EodColumns::p_bid);
    const auto ask   = field_opt(idx, row, call ? EodColumns::c_ask   : EodColumns::p_ask);
    const auto last  = field_opt(idx, row, call ? EodColumns::c_last  : EodColumns::p_last);
    const auto vol   = field_opt(idx, row, call ? EodColumns::c_vol   : EodColumns::p_vol);
    const auto iv    = field_opt(idx, row, call ? EodColumns::c_iv    : EodColumns::p_iv);

    ore::core::Quote quote{};
    quote.bid    = bid.value_or(0.0);
    quote.ask    = ask.value_or(0.0);
    quote.last   = last.value_or(0.0);
    quote.volume = vol.value_or(0.0);
    if (iv.has_value() && *iv > 0.0) quote.implied_volatility = *iv;

    if (options.skip_zero_quotes && quote.bid == 0.0 && quote.ask == 0.0) {
        return std::nullopt;
    }

    ore::core::ProviderGreeks pg{};
    if (auto d = field_opt(idx, row, call ? EodColumns::c_delta : EodColumns::p_delta)) pg.delta = *d;
    if (auto g = field_opt(idx, row, call ? EodColumns::c_gamma : EodColumns::p_gamma)) pg.gamma = *g;
    if (auto t = field_opt(idx, row, call ? EodColumns::c_theta : EodColumns::p_theta)) pg.theta = *t;
    if (auto v = field_opt(idx, row, call ? EodColumns::c_vega  : EodColumns::p_vega )) pg.vega  = *v;
    if (auto r = field_opt(idx, row, call ? EodColumns::c_rho   : EodColumns::p_rho  )) pg.rho   = *r;
    if (pg.any()) quote.provider_greeks = pg;

    ore::core::Option opt{};
    opt.underlying = underlying;
    opt.strike     = strike;
    opt.expiration = expire_date;
    opt.type       = side;
    opt.exercise   = ore::core::ExerciseStyle::European;

    ore::core::OptionMarketSnapshot oms{};
    oms.option          = opt;
    oms.quote           = quote;
    // Vendor archives do not publish an OCC symbol per row. Leaving
    // it empty is documented behaviour of `OptionMarketSnapshot`.
    oms.contract_symbol = {};

    (void)quote_date; // reserved for future validation / diagnostics
    return oms;
}

/// Accumulator that groups rows by trading date. Kept as a
/// `std::map` (not `unordered_map`) so the final iteration produces
/// snapshots in chronological order at negligible cost — the number
/// of distinct dates per file is at most ~25.
struct SnapshotBuilder {
    ore::core::Underlying underlying;
    std::map<std::chrono::year_month_day,
             std::pair<double /* spot */,
                       std::vector<ore::core::OptionMarketSnapshot>>> by_date;

    void add(std::chrono::year_month_day date,
             double spot,
             ore::core::OptionMarketSnapshot oms)
    {
        auto& [existing_spot, options] = by_date[date];
        // Spot is common across every row of a given day; keep the
        // last-seen value (they should all agree). Tests exercise
        // this by parsing multi-row fixtures and checking spot on
        // the resulting snapshot.
        existing_spot = spot;
        options.push_back(std::move(oms));
    }

    [[nodiscard]] std::vector<HistoricalSnapshot> finalize(
        const EodOptionsLoader::Options& options) const
    {
        std::vector<HistoricalSnapshot> out;
        out.reserve(by_date.size());
        for (const auto& [date, pair] : by_date) {
            ore::core::MarketSnapshot market{};
            market.spot           = pair.first;
            market.risk_free_rate = options.risk_free_rate;
            market.dividend_yield = options.dividend_yield;
            // Fallback vol — pricing engines that use the market
            // snapshot's `volatility` field will need a positive
            // value. Research studies always override with per-
            // contract IVs recovered from the quote, so 0.20 is a
            // conventional placeholder rather than a real prior.
            market.volatility     = 0.20;
            market.valuation_date = date;

            ore::core::OptionChain chain{
                underlying,
                market,
                pair.second,
            };
            out.emplace_back(date, std::move(chain));
        }
        return out;
    }
};

/// Column set the parser needs at minimum. Missing any of these
/// throws `LoaderError` regardless of strict/lenient mode — without
/// them we cannot even identify the strike or the trading day.
std::vector<std::string> required_columns() {
    return {
        EodColumns::quote_date,
        EodColumns::underlying_last,
        EodColumns::expire_date,
        EodColumns::strike,
        EodColumns::c_bid, EodColumns::c_ask,
        EodColumns::p_bid, EodColumns::p_ask,
    };
}

/// Determine whether `p` looks like an OptionsDX archive we should
/// try to parse. `discover_files` uses this to skip README-style
/// siblings without recursing into their contents.
[[nodiscard]] bool looks_like_archive(const fs::path& p) {
    if (!p.has_extension()) return false;
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext != ".txt" && ext != ".csv") return false;
    const auto fn = p.filename().string();
    if (!fn.empty() && fn.front() == '.') return false; // .DS_Store, swap files
    return true;
}

/// Streamed file-to-string reader. `std::ifstream` + `rdbuf` would do
/// the same in one line but this version is a little kinder on peak
/// memory for the multi-hundred-MB archives (the reserve avoids
/// reallocations without materialising the whole file twice).
[[nodiscard]] std::string slurp_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::ostringstream oss;
        oss << "cannot open EOD archive '" << path.string() << "'";
        throw LoaderError(oss.str());
    }
    std::error_code ec;
    const auto sz = fs::file_size(path, ec);
    std::string data;
    if (!ec) data.reserve(static_cast<std::size_t>(sz));
    data.assign(std::istreambuf_iterator<char>(in),
                std::istreambuf_iterator<char>());
    return data;
}

/// Line splitter that copes with `\n` and `\r\n`. Returns
/// `string_view`s into `text`, so `text` must outlive the returned
/// vector — the archive path signs up for that by keeping the
/// entire file in memory.
[[nodiscard]] std::vector<std::string_view> split_lines(std::string_view text) {
    std::vector<std::string_view> lines;
    std::size_t start = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\n') {
            std::size_t end = i;
            if (end > start && text[end - 1] == '\r') --end;
            lines.emplace_back(text.substr(start, end - start));
            start = i + 1;
        }
    }
    if (start < text.size()) {
        lines.emplace_back(text.substr(start));
    }
    return lines;
}

/// Parse the fully-loaded text into a snapshot list. Split out from
/// `load_file` / `parse_string` so both entry points share the loop.
[[nodiscard]] std::vector<HistoricalSnapshot> parse_body(
    std::string_view text,
    const EodOptionsLoader::Options& options,
    std::string_view source)
{
    const auto lines = split_lines(text);
    if (lines.empty()) return {};

    // Find the first non-empty line; that's the header.
    std::size_t header_line_idx = 0;
    while (header_line_idx < lines.size()
           && trim_field(lines[header_line_idx]).empty()) {
        ++header_line_idx;
    }
    if (header_line_idx >= lines.size()) return {};

    const auto header_line = lines[header_line_idx];
    const auto required    = required_columns();
    const auto idx         = build_header_index(header_line, required, source);

    SnapshotBuilder builder{};
    builder.underlying = ore::core::Underlying{options.ticker};

    std::vector<std::string_view> row_fields;
    row_fields.reserve(idx.map.size());

    for (std::size_t li = header_line_idx + 1; li < lines.size(); ++li) {
        const auto line = trim_field(lines[li]);
        if (line.empty()) continue;

        split_line(line, row_fields);

        const auto date  = parse_iso_date(field_str(idx, row_fields, EodColumns::quote_date));
        const auto exp   = parse_iso_date(field_str(idx, row_fields, EodColumns::expire_date));
        const auto spot  = field_opt(idx, row_fields, EodColumns::underlying_last);
        const auto strk  = field_opt(idx, row_fields, EodColumns::strike);

        if (!date || !exp || !spot || !strk || *strk <= 0.0) {
            if (options.strict) {
                std::ostringstream oss;
                oss << "EOD archive '" << source << "' line "
                    << (li + 1) << ": malformed row (missing date, "
                    << "expiration, spot, or strike)";
                throw LoaderError(oss.str());
            }
            continue;
        }

        if (options.skip_expired) {
            const auto q_sd = std::chrono::sys_days{*date};
            const auto e_sd = std::chrono::sys_days{*exp};
            if (e_sd < q_sd) continue;
        }

        // Emit both the call and put sides. `build_side` may drop
        // one or both if `skip_zero_quotes` is on and the strike is
        // effectively unquoted on that side.
        for (auto side : {ore::core::OptionType::Call, ore::core::OptionType::Put}) {
            auto oms = build_side(idx, row_fields, side,
                                  builder.underlying,
                                  *date, *exp, *strk, options);
            if (oms.has_value()) {
                builder.add(*date, *spot, std::move(*oms));
            }
        }
    }

    return builder.finalize(options);
}

} // namespace

//
// load_file -------------------------------------------------------------------
//

std::vector<HistoricalSnapshot>
EodOptionsLoader::load_file(const fs::path& file, const Options& options) {
    const auto text = slurp_file(file);
    return parse_body(text, options, file.string());
}

//
// parse_string ----------------------------------------------------------------
//

std::vector<HistoricalSnapshot>
EodOptionsLoader::parse_string(std::string_view text,
                               const Options& options,
                               std::string source)
{
    return parse_body(text, options, source);
}

//
// discover_files --------------------------------------------------------------
//

std::vector<fs::path>
EodOptionsLoader::discover_files(const fs::path& root) {
    std::vector<fs::path> out;
    std::error_code ec;
    if (!fs::exists(root, ec) || !fs::is_directory(root, ec)) return out;

    for (auto it = fs::recursive_directory_iterator(root, ec);
         it != fs::recursive_directory_iterator();
         it.increment(ec))
    {
        if (ec) break;
        const auto& entry = *it;
        if (!entry.is_regular_file(ec)) continue;
        if (looks_like_archive(entry.path())) {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

//
// load_directory --------------------------------------------------------------
//

std::vector<HistoricalSnapshot>
EodOptionsLoader::load_directory(const fs::path& root, const Options& options) {
    const auto files = discover_files(root);

    // Group all rows across all files by trading date, then finalise
    // once. This handles the (uncommon) case of a trading day
    // straddling two files (e.g. archives that split at month
    // boundaries).
    std::map<std::chrono::year_month_day,
             std::vector<ore::core::OptionMarketSnapshot>> merged_options;
    std::map<std::chrono::year_month_day, double> merged_spot;

    ore::core::Underlying underlying{options.ticker};

    for (const auto& f : files) {
        auto per_file = load_file(f, options);
        for (auto& snap : per_file) {
            const auto d = snap.date();
            auto& bucket = merged_options[d];
            bucket.reserve(bucket.size() + snap.options().size());
            for (const auto& oms : snap.options()) {
                bucket.push_back(oms);
            }
            merged_spot[d] = snap.market().spot;
        }
    }

    std::vector<HistoricalSnapshot> out;
    out.reserve(merged_options.size());
    for (auto& [date, opts] : merged_options) {
        ore::core::MarketSnapshot market{};
        market.spot           = merged_spot[date];
        market.risk_free_rate = options.risk_free_rate;
        market.dividend_yield = options.dividend_yield;
        market.volatility     = 0.20;
        market.valuation_date = date;

        ore::core::OptionChain chain{
            underlying,
            market,
            std::move(opts),
        };
        out.emplace_back(date, std::move(chain));
    }
    return out;
}

} // namespace ore::marketdata
