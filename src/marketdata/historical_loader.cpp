#include <ore/marketdata/historical_loader.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <ore/core/option_chain.hpp>
#include <ore/marketdata/eod_options_loader.hpp>
#include <ore/marketdata/historical_dataset.hpp>
#include <ore/marketdata/historical_snapshot.hpp>
#include <ore/marketdata/yahoo_option_loader.hpp>

namespace ore::marketdata {

namespace {

namespace fs = std::filesystem;

/**
 * @brief Parse a `YYYY-MM-DD` directory name into a `year_month_day`.
 *        Returns `nullopt` on any parse failure (extra characters,
 *        non-digits, invalid calendar date).
 *
 * We do the parse by hand rather than reuse `YahooOptionLoader`'s
 * `parse_date` because that one throws on failure — the caller here
 * needs a non-fatal "does this name look like a date?" primitive so
 * `list_dates` can silently skip README.md, `.DS_Store`, etc.
 */
std::optional<std::chrono::year_month_day>
try_parse_date_dir(std::string_view name)
{
    if (name.size() != 10) return std::nullopt;
    if (name[4] != '-' || name[7] != '-') return std::nullopt;

    auto to_int = [](std::string_view piece) -> std::optional<int> {
        int v = 0;
        for (char c : piece) {
            if (c < '0' || c > '9') return std::nullopt;
            v = v * 10 + (c - '0');
        }
        return v;
    };
    const auto y = to_int(name.substr(0, 4));
    const auto m = to_int(name.substr(5, 2));
    const auto d = to_int(name.substr(8, 2));
    if (!y || !m || !d) return std::nullopt;

    std::chrono::year_month_day ymd{
        std::chrono::year{*y},
        std::chrono::month{static_cast<unsigned>(*m)},
        std::chrono::day{static_cast<unsigned>(*d)},
    };
    if (!ymd.ok()) return std::nullopt;
    return ymd;
}

/** Format a `year_month_day` back to `YYYY-MM-DD`. */
std::string format_date(std::chrono::year_month_day ymd) {
    std::ostringstream oss;
    oss << static_cast<int>(ymd.year()) << '-';
    unsigned m = static_cast<unsigned>(ymd.month());
    unsigned d = static_cast<unsigned>(ymd.day());
    if (m < 10) oss << '0';
    oss << m << '-';
    if (d < 10) oss << '0';
    oss << d;
    return oss.str();
}

/**
 * @brief Build the disk path a given (root, ticker, date) triple
 *        resolves to. Kept in one place so tests and the loader agree
 *        on the layout even when the format is tweaked.
 */
fs::path day_directory(const fs::path& root,
                       std::string_view ticker,
                       std::chrono::year_month_day date)
{
    return root / std::string(ticker) / format_date(date);
}

/**
 * @brief List all business days (Mon-Fri) in `[start, end]`. Used to
 *        compute `LoadResult::missing_dates`.
 *
 * We treat any weekday as a "potentially expected" trading day; real
 * market holidays (e.g. NYSE) are not filtered because the loader has
 * no calendar dependency at this milestone. Callers who want a strict
 * NYSE calendar can post-filter the returned list against a holiday
 * table. `docs/HISTORICAL_DATA.md` documents this limitation.
 */
std::vector<std::chrono::year_month_day>
enumerate_business_days(std::chrono::year_month_day start,
                        std::chrono::year_month_day end)
{
    using days = std::chrono::days;
    std::vector<std::chrono::year_month_day> out;

    const auto start_sd = std::chrono::sys_days{start};
    const auto end_sd   = std::chrono::sys_days{end};
    if (end_sd < start_sd) return out;

    for (auto sd = start_sd; sd <= end_sd; sd += days{1}) {
        const auto wd = std::chrono::weekday{sd};
        // Skip Saturday (6) and Sunday (0 in iso_encoding()).
        const auto iso = wd.iso_encoding();
        if (iso == 6u || iso == 7u) continue;
        out.push_back(std::chrono::year_month_day{sd});
    }
    return out;
}

/**
 * @brief Given a sorted list of on-disk dates and a sorted list of
 *        business days in the same range, return the business days
 *        that are *not* present on disk.
 *
 * O(n+m) merge-scan. Both inputs are already sorted so we don't need
 * a set or a binary-search-per-element pass.
 */
std::vector<std::chrono::year_month_day>
missing_business_days(
    const std::vector<std::chrono::year_month_day>& business,
    const std::vector<std::chrono::year_month_day>& present)
{
    std::vector<std::chrono::year_month_day> out;
    std::size_t i = 0, j = 0;
    while (i < business.size() && j < present.size()) {
        const auto b = std::chrono::sys_days{business[i]};
        const auto p = std::chrono::sys_days{present[j]};
        if (b < p) {
            out.push_back(business[i]);
            ++i;
        } else if (b > p) {
            ++j;
        } else {
            ++i;
            ++j;
        }
    }
    while (i < business.size()) {
        out.push_back(business[i]);
        ++i;
    }
    return out;
}

} // namespace

//
// list_dates ------------------------------------------------------------------
//

std::vector<std::chrono::year_month_day>
HistoricalLoader::list_dates(std::string_view ticker,
                             const fs::path& root)
{
    const fs::path ticker_dir = root / std::string(ticker);

    std::vector<std::chrono::year_month_day> dates;
    std::error_code ec;
    if (!fs::exists(ticker_dir, ec) || !fs::is_directory(ticker_dir, ec)) {
        // A missing ticker directory returns an empty list rather than
        // throwing — matches "no data" more cleanly than "error", and
        // lets `load(ticker)` produce a well-defined empty dataset.
        return dates;
    }

    for (const auto& entry : fs::directory_iterator(ticker_dir, ec)) {
        if (!entry.is_directory(ec)) continue;
        const auto name = entry.path().filename().string();
        if (auto parsed = try_parse_date_dir(name)) {
            dates.push_back(*parsed);
        }
        // else: silently skip. `underlying_history.csv`, README.md,
        // `.DS_Store`, and other non-date entries land here.
    }
    std::sort(dates.begin(), dates.end(),
              [](auto a, auto b) {
                  return std::chrono::sys_days{a} < std::chrono::sys_days{b};
              });
    return dates;
}

//
// load_day --------------------------------------------------------------------
//

HistoricalSnapshot
HistoricalLoader::load_day(std::string_view ticker,
                           std::chrono::year_month_day date,
                           const fs::path& root)
{
    const auto dir = day_directory(root, ticker, date);
    // Delegates to the existing Yahoo loader — no duplicated parsing.
    // YahooOptionLoader::load already validates the directory name
    // against metadata.valuation_date and skips the "options/" parent
    // check when the parent is a ticker rather than `options`.
    ore::core::OptionChain chain = YahooOptionLoader::load(dir);
    return HistoricalSnapshot{date, std::move(chain)};
}

//
// load(ticker) ----------------------------------------------------------------
//

HistoricalDataset HistoricalLoader::load(std::string_view ticker) {
    Options opts{};
    auto result = load(ticker, opts);
    return std::move(result.dataset);
}

//
// load(ticker, Options) -------------------------------------------------------
//

HistoricalLoader::LoadResult
HistoricalLoader::load(std::string_view ticker, const Options& options)
{
    LoadResult result{
        .dataset       = HistoricalDataset{std::string(ticker), {}},
        .failed_days   = {},
        .missing_dates = {},
    };

    auto all_dates = list_dates(ticker, options.root);

    // Filter to the requested [start, end] window (inclusive on both
    // ends). Kept simple with std::remove_if — the total number of
    // trading days on disk is typically a few thousand, which is not
    // large enough to bother with a bisection.
    if (options.start_date.has_value()) {
        const auto start_sd = std::chrono::sys_days{*options.start_date};
        all_dates.erase(
            std::remove_if(all_dates.begin(), all_dates.end(),
                           [start_sd](auto d) {
                               return std::chrono::sys_days{d} < start_sd;
                           }),
            all_dates.end());
    }
    if (options.end_date.has_value()) {
        const auto end_sd = std::chrono::sys_days{*options.end_date};
        all_dates.erase(
            std::remove_if(all_dates.begin(), all_dates.end(),
                           [end_sd](auto d) {
                               return std::chrono::sys_days{d} > end_sd;
                           }),
            all_dates.end());
    }

    std::vector<HistoricalSnapshot> snapshots;
    snapshots.reserve(all_dates.size());

    for (auto d : all_dates) {
        try {
            snapshots.push_back(load_day(ticker, d, options.root));
        } catch (const LoaderError& e) {
            if (options.strict) {
                // In strict mode, propagate the first failure —
                // callers can catch `LoaderError` at the top level.
                throw;
            }
            result.failed_days.emplace_back(d, e.what());
        }
    }

    result.dataset = HistoricalDataset{std::string(ticker), std::move(snapshots)};

    // Compute missing dates against the *loaded* range so weekends
    // outside [first, last] don't produce a huge irrelevant list.
    // Also: if the caller supplied an explicit window, honour it —
    // that lets the loader report "these dates were expected but
    // absent" against the user's request, not against the disk.
    std::optional<std::chrono::year_month_day> range_start = result.dataset.first_date();
    std::optional<std::chrono::year_month_day> range_end   = result.dataset.last_date();
    if (options.start_date.has_value()) range_start = *options.start_date;
    if (options.end_date.has_value())   range_end   = *options.end_date;

    if (range_start.has_value() && range_end.has_value()) {
        const auto business = enumerate_business_days(*range_start, *range_end);

        // `present` = loaded dates ∪ failed dates. A failed day was
        // still "on disk" — it should not be counted as missing.
        std::vector<std::chrono::year_month_day> present;
        present.reserve(result.dataset.size() + result.failed_days.size());
        for (const auto& snap : result.dataset) present.push_back(snap.date());
        for (const auto& [d, _] : result.failed_days) present.push_back(d);
        std::sort(present.begin(), present.end(),
                  [](auto a, auto b) {
                      return std::chrono::sys_days{a} < std::chrono::sys_days{b};
                  });

        result.missing_dates = missing_business_days(business, present);
    }

    return result;
}

//
// load_from_eod ---------------------------------------------------------------
//

HistoricalDataset
HistoricalLoader::load_from_eod(const fs::path& root,
                                std::string_view ticker,
                                EodOptionsLoader::Options options)
{
    // Ensure the ticker passed here is what winds up on every derived
    // `Underlying` even if the caller left `options.ticker` at the
    // default "SPY". This keeps the ticker convention consistent
    // between the dataset's `ticker()` accessor and each snapshot's
    // `underlying().symbol()`.
    options.ticker = std::string(ticker);
    auto snapshots = EodOptionsLoader::load_directory(root, options);
    return HistoricalDataset{std::string(ticker), std::move(snapshots)};
}

} // namespace ore::marketdata
