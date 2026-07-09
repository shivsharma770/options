/**
 * @file historical_loader.hpp
 * @brief `HistoricalLoader` — turns a filesystem tree of per-day
 *        option-chain snapshots into a `HistoricalDataset`.
 *
 * ### Directory layout
 *
 * ```
 * <root>/<TICKER>/<YYYY-MM-DD>/
 *     metadata.csv    # one row of market-wide state
 *     calls.csv       # per-contract data, all expirations
 *     puts.csv        # per-contract data, all expirations
 * ```
 *
 * The schema of each CSV is identical to the "current-snapshot" layout
 * consumed by `YahooOptionLoader` (which is what `HistoricalLoader`
 * delegates to under the hood). The only difference is the enclosing
 * directory tree: historical data lives under `data/historical/raw/`
 * with **no** intermediate `options/` folder. See
 * `docs/HISTORICAL_DATA.md` for the rationale.
 *
 * ### Behaviour
 *
 * The loader is a small set of static functions — no state, trivially
 * thread-safe. Each call performs a fresh filesystem traversal; there
 * is deliberately **no caching** at this milestone (see the milestone
 * spec).
 *
 * The strict-mode default rejects the first parse failure with
 * `LoaderError` (inherited from `YahooOptionLoader`). Lenient mode
 * (`Options::strict = false`) drops broken days and records them in
 * the returned `LoadResult`.
 *
 * ### Extensibility
 *
 * `HistoricalLoader` currently understands only the Yahoo-schema
 * per-day layout. When we add Polygon/Databento adapters, we will
 * introduce a `HistoricalLoaderBackend` polymorphic interface here and
 * have the current implementation become the default `YahooBackend`.
 * The public API of this class will not change.
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ore/marketdata/eod_options_loader.hpp>
#include <ore/marketdata/historical_dataset.hpp>
#include <ore/marketdata/historical_snapshot.hpp>
#include <ore/marketdata/yahoo_option_loader.hpp>

namespace ore::marketdata {

/**
 * @class HistoricalLoader
 * @brief Static filesystem-loader for historical option-chain data.
 */
class HistoricalLoader {
public:
    /**
     * @brief Options controlling `load(ticker, Options)`.
     *
     * Every field has a sensible default so `Options{}` matches the
     * behaviour of the parameterless `load(ticker)` shortcut.
     */
    struct Options {
        /** Root directory containing per-ticker folders. */
        std::filesystem::path root{"data/historical/raw"};
        /** Inclusive lower bound on trading dates to load. Nullopt =
         *  "since the earliest date on disk". */
        std::optional<std::chrono::year_month_day> start_date{};
        /** Inclusive upper bound on trading dates to load. Nullopt =
         *  "until the latest date on disk". */
        std::optional<std::chrono::year_month_day> end_date{};
        /** Strict mode: rethrow the first `LoaderError` and abort. In
         *  lenient mode, per-day failures are recorded in
         *  `LoadResult::failed_days` and loading proceeds. */
        bool strict{true};
    };

    /**
     * @brief Bundle returned by `load(ticker, Options)`.
     *
     * The `dataset` field is always populated (possibly empty). The
     * diagnostic fields let lenient-mode callers see what did *not*
     * load.
     */
    struct LoadResult {
        HistoricalDataset dataset;
        /** One entry per failed day: `(date, exception::what())`. */
        std::vector<std::pair<std::chrono::year_month_day, std::string>>
            failed_days{};
        /** Business days within `[first_date, last_date]` that neither
         *  loaded nor failed — weekends and calendar holes. */
        std::vector<std::chrono::year_month_day> missing_dates{};
    };

    /**
     * @brief Convenience: load every day under
     *        `data/historical/raw/<ticker>/` strictly.
     *
     * Throws on the first parse failure. Suitable for research
     * scripts that want to assume the data on disk is clean.
     */
    [[nodiscard]] static HistoricalDataset load(std::string_view ticker);

    /**
     * @brief Full-power overload: pass `Options` for root, date range,
     *        and strict/lenient behaviour.
     *
     * @throws LoaderError if `options.strict == true` and any day
     *         fails to parse.
     * @return `LoadResult` with `dataset` populated by the days that
     *         loaded, and diagnostic vectors describing anything that
     *         did not.
     */
    [[nodiscard]] static LoadResult load(std::string_view ticker,
                                         const Options&   options);

    /**
     * @brief Enumerate every directory under `<root>/<ticker>/` whose
     *        name parses as `YYYY-MM-DD`, sorted ascending.
     *
     * Directories whose names don't parse are silently ignored — the
     * function's job is enumeration, not validation. Missing
     * `<root>/<ticker>` returns an empty vector.
     */
    [[nodiscard]] static std::vector<std::chrono::year_month_day> list_dates(
        std::string_view ticker,
        const std::filesystem::path& root = "data/historical/raw");

    /**
     * @brief Load exactly one day's snapshot. Delegates to
     *        `YahooOptionLoader::load` after computing the on-disk
     *        directory path.
     *
     * @throws LoaderError if the directory or any of its CSVs are
     *         missing or malformed.
     */
    [[nodiscard]] static HistoricalSnapshot load_day(
        std::string_view ticker,
        std::chrono::year_month_day date,
        const std::filesystem::path& root = "data/historical/raw");

    /**
     * @brief Load a large multi-year archive in the OptionsDX / Delta-
     *        Neutral end-of-day format.
     *
     * Recursively descends `root`, discovering every archive file
     * whose extension is `.txt` or `.csv`. Directory names starting
     * with `spy_eod_` are the intended container (that's what the
     * upstream distributions ship as) but the loader itself does not
     * pattern-match on the directory name — any parseable archive
     * anywhere beneath `root` is consumed. This keeps future
     * archives (e.g. `qqq_eod_2022-*`) working without code changes.
     *
     * @param root    Directory to search. Typical values:
     *                `data/historical/spy` — parent of every
     *                `spy_eod_YYYY-*` subdirectory — or a single
     *                archive directory.
     * @param ticker  Ticker symbol attached to the returned dataset
     *                and every derived `Underlying`. Not validated
     *                against the file contents.
     * @param options Additional `EodOptionsLoader::Options`
     *                (skip-expired, risk-free rate, ...). The
     *                `ticker` field is overwritten with this
     *                function's `ticker` argument.
     *
     * @return `HistoricalDataset` containing one snapshot per
     *         trading date discovered, sorted ascending.
     */
    [[nodiscard]] static HistoricalDataset load_from_eod(
        const std::filesystem::path& root,
        std::string_view ticker,
        EodOptionsLoader::Options options = {});
};

} // namespace ore::marketdata
