/**
 * @file eod_options_loader.hpp
 * @brief Parser for end-of-day option-chain archives distributed by
 *        OptionsDX / Delta-Neutral and similar vendors.
 *
 * ### Wire format
 *
 * A single "file" is a plain-text CSV with:
 *
 *   - a header line whose column names are wrapped in square brackets,
 *     e.g. `[QUOTE_UNIXTIME], [QUOTE_DATE], [C_DELTA], ...`;
 *   - data lines separated by commas with **whitespace after each
 *     delimiter** (`, ` rather than `,`) — an idiosyncrasy of the
 *     upstream feed that a plain RFC 4180 parser rejects;
 *   - **one row per strike** carrying both the call quote/Greeks and
 *     the put quote/Greeks side by side (`C_DELTA, C_GAMMA, ..., C_BID,
 *     C_ASK, STRIKE, P_BID, P_ASK, ..., P_DELTA, ...`);
 *   - multiple trading days per file (the archives are typically packed
 *     one calendar month per file). Rows for the same trading day share
 *     `QUOTE_DATE`. Rows are grouped by `QUOTE_DATE` at load time.
 *
 * Empty fields represent missing data (a common case for illiquid
 * strikes with no published IV or no Greeks) and are preserved as
 * `std::nullopt` on the returned `Quote` / `ProviderGreeks`.
 *
 * ### Public API
 *
 * `EodOptionsLoader` is a small collection of static functions. Two
 * entry points cover every use case:
 *
 *   - `load_file()` — parse a single archive file and return the list
 *     of `HistoricalSnapshot`s (one per trading day) it contains.
 *   - `load_directory()` — recursively descend a directory tree, load
 *     every `.txt`/`.csv` file in it, and return the merged
 *     `std::vector<HistoricalSnapshot>` sorted by date.
 *
 * `HistoricalLoader::load_from_eod` wraps `load_directory` and boxes
 * the result into a `HistoricalDataset` with the ticker attached.
 *
 * ### Error handling
 *
 * The loader is deliberately **lenient**: real-world EOD archives
 * contain thousands of degenerate rows (zero bid/ask, missing IV,
 * expired contracts in the "expiring today" batch). Silently skipping
 * these keeps the loader usable on the raw feed. Callers who want a
 * strict view can post-filter the loaded chains.
 *
 * Structural failures (unreadable file, corrupt header, missing
 * required columns) still throw `LoaderError`.
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <ore/core/option_chain.hpp>
#include <ore/marketdata/historical_snapshot.hpp>
#include <ore/marketdata/yahoo_option_loader.hpp>

namespace ore::marketdata {

/**
 * @class EodOptionsLoader
 * @brief Loader for OptionsDX / Delta-Neutral end-of-day archives.
 */
class EodOptionsLoader {
public:
    /**
     * @brief Options controlling `load_file` / `load_directory`.
     *
     * Defaults are chosen so that vanilla archives load without any
     * configuration.
     */
    struct Options {
        /**
         * Ticker to attach to every loaded `Underlying`. Delta-Neutral
         * archive filenames encode the ticker (`spy_eod_...`) but the
         * per-row payload does not repeat it; callers must supply it
         * once at load time. Defaults to "SPY".
         */
        std::string ticker{"SPY"};

        /**
         * If a row has zero-bid **and** zero-ask, the strike is
         * effectively unquoted (no market maker responded). Setting
         * this to `true` drops those rows. `false` keeps them — useful
         * when the study wants to look at the distribution of
         * unquoted strikes.
         */
        bool skip_zero_quotes{false};

        /**
         * If a row's expiration date is before its quote date, the
         * contract has already expired. Vendors sometimes include the
         * expiring-today entries with negative DTE due to timezone
         * quirks. Setting this to `true` drops them.
         */
        bool skip_expired{true};

        /**
         * If the parser encounters a row it cannot decode (missing
         * `STRIKE`, malformed date, non-numeric field where a number
         * was required), setting this to `true` throws immediately.
         * The default swallows the failure and continues — real
         * archives contain a small fraction of unparseable rows and
         * halting the whole load on the first one is rarely what a
         * research script wants.
         */
        bool strict{false};

        /**
         * Continuous risk-free rate to attach to every derived
         * `MarketSnapshot`. EOD archives do not carry rates so the
         * value must be supplied by the caller. Zero is a common
         * default for equity-index research where the effect on IV
         * recovery is small; supply an actual rate curve for
         * production-grade validation.
         */
        double risk_free_rate{0.0};

        /**
         * Continuous dividend yield to attach to every derived
         * `MarketSnapshot`. Same rationale as `risk_free_rate`.
         */
        double dividend_yield{0.0};
    };

    /**
     * @brief The default `Options` used when a caller omits the argument.
     *
     * Returned by const reference to a function-local static. The load
     * entry points below default to `= default_options()` rather than
     * `= {}` because GCC rejects an in-class `Options{}` braced-init
     * default argument — it would evaluate `Options`'s default member
     * initializers before this enclosing class is complete. Routing
     * through a function call defers that to the definition point.
     */
    [[nodiscard]] static const Options& default_options();

    /**
     * @brief Parse a single OptionsDX archive file.
     *
     * @param file    Path to the `.txt`/`.csv` archive.
     * @param options Loader configuration (see `Options`).
     *
     * @return One `HistoricalSnapshot` per distinct `QUOTE_DATE`
     *         encountered in the file, sorted ascending.
     *
     * @throws LoaderError on structural failures (missing file,
     *         malformed header, missing required columns) or,
     *         when `Options::strict` is `true`, on the first
     *         unparseable data row.
     */
    [[nodiscard]] static std::vector<HistoricalSnapshot> load_file(
        const std::filesystem::path& file,
        const Options& options = default_options());

    /**
     * @brief Recursively load every archive file under `root`.
     *
     * The traversal considers any file with a `.txt` or `.csv`
     * extension. Files whose name begins with `.` are ignored (macOS
     * resource forks, editor swap files). Snapshots from all files
     * are merged; if two files carry rows for the same trading date
     * they are combined into one snapshot (contracts appended).
     *
     * @param root    Directory to search. May be a single archive
     *                directory (`spy_eod_2010-XXXXX/`) or the parent
     *                (`data/historical/spy/`) — the loader recurses
     *                in both cases.
     * @param options Loader configuration.
     *
     * @return Snapshots sorted ascending by date. Empty vector if
     *         `root` is missing, not a directory, or contains no
     *         parseable rows.
     */
    [[nodiscard]] static std::vector<HistoricalSnapshot> load_directory(
        const std::filesystem::path& root,
        const Options& options = default_options());

    /**
     * @brief List every archive file that `load_directory` would
     *        visit, in ascending path order.
     *
     * Useful for progress reporting and for tests that want to
     * exercise the enumeration logic without actually parsing
     * anything.
     */
    [[nodiscard]] static std::vector<std::filesystem::path> discover_files(
        const std::filesystem::path& root);

    /**
     * @brief Parse an EOD archive from an in-memory string. Handy for
     *        unit tests and for callers that fetch data from a
     *        non-filesystem source. Errors reference `source` in
     *        their messages.
     */
    [[nodiscard]] static std::vector<HistoricalSnapshot> parse_string(
        std::string_view text,
        const Options& options = default_options(),
        std::string source = "<memory>");
};

} // namespace ore::marketdata
