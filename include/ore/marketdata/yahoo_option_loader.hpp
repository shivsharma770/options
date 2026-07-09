/**
 * @file yahoo_option_loader.hpp
 * @brief Loads a full option chain snapshot written by the Python
 *        `download_option_chain.py` script (which uses `yfinance`).
 *
 * A snapshot on disk consists of three sibling CSV files inside a directory
 * named for the valuation date:
 *
 * @code
 *   data/raw/SPY/options/2026-07-08/
 *       metadata.csv    # one row: market-wide state at the valuation moment
 *       calls.csv       # per-contract data for all calls, all expirations
 *       puts.csv        # per-contract data for all puts, all expirations
 * @endcode
 *
 * The loader is a small set of static functions rather than a class with
 * state: no configuration, no side effects, trivially thread-safe. Adding a
 * new data provider (Polygon, Databento, ...) means adding a sibling class
 * with the same shape — no polymorphism needed because every loader
 * produces the same `ore::core::OptionChain`.
 *
 * Errors: all failures throw `ore::marketdata::LoaderError`. Underlying CSV
 * parse errors are wrapped and re-thrown as `LoaderError` so callers only
 * need to catch one exception type.
 *
 * Strictness: the loader is strict by default. It refuses to construct an
 * `OptionChain` from a snapshot with any of the anomalies enumerated in
 * `Rules`. A `lenient` mode can be added later; keeping it strict now
 * forces data problems to surface immediately.
 */
#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <ore/core/market_snapshot.hpp>
#include <ore/core/option.hpp>
#include <ore/core/option_chain.hpp>
#include <ore/core/option_market_snapshot.hpp>
#include <ore/core/underlying.hpp>
#include <ore/io/csv_reader.hpp>

namespace ore::marketdata {

/**
 * Thrown when a Yahoo snapshot cannot be loaded or validated. Message names
 * the file, the row, and the offending column where relevant.
 */
class LoaderError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * Loader for Yahoo-Finance-style option chain snapshots.
 *
 * Validation rules enforced (strict mode):
 *   - required columns present in every file (see `columns()` below)
 *   - `strike > 0`
 *   - `bid >= 0`, `ask >= 0`
 *   - `ask >= bid` when both are strictly positive (allows `bid = ask = 0`
 *     which signals "no market" rather than a crossed quote)
 *   - `implied_volatility >= 0` when present (NaN / empty is tolerated —
 *     Yahoo often reports zero or missing IV for deep OTM contracts)
 *   - `volume >= 0`, `open_interest >= 0`
 *   - `option_type` parses to "call" or "put" (case-insensitive)
 *   - `expiration`, `valuation_date`, and `last_trade_date` parse as
 *     ISO 8601 (`YYYY-MM-DD` / `YYYY-MM-DDTHH:MM:SS`)
 *   - `expiration >= valuation_date` (rejects stale/expired contracts)
 *   - `contract_symbol` non-empty and unique within a single file
 *   - metadata `underlying_symbol` matches the parent directory name
 */
class YahooOptionLoader {
public:
    /**
     * Load a full option chain from a snapshot directory.
     *
     * @param snapshot_directory Directory containing metadata.csv,
     *                           calls.csv, and puts.csv.
     * @throws LoaderError if any file is missing, malformed, or fails
     *                     validation.
     */
    [[nodiscard]] static ore::core::OptionChain load(
        const std::filesystem::path& snapshot_directory);

    /**
     * Parsed contents of `metadata.csv`. Exposed as an intermediate step so
     * unit tests can exercise metadata parsing without touching the disk.
     */
    struct SnapshotMetadata {
        ore::core::Underlying     underlying;
        ore::core::MarketSnapshot market;
    };

    /** Parse a `CsvTable` produced from `metadata.csv`. Requires exactly one row. */
    [[nodiscard]] static SnapshotMetadata parse_metadata(const ore::io::CsvTable& table);

    /**
     * Parse a `CsvTable` produced from `calls.csv` or `puts.csv` into
     * `OptionMarketSnapshot`s. `expected_type` is used both to populate
     * `Option::type` on rows where the CSV omits an `option_type` column
     * and to reject rows whose stated `option_type` disagrees. The
     * `underlying` is attached to every returned option.
     *
     * `valuation_date` is used to reject expired contracts.
     */
    [[nodiscard]] static std::vector<ore::core::OptionMarketSnapshot> parse_contracts(
        const ore::io::CsvTable&              table,
        ore::core::OptionType                 expected_type,
        const ore::core::Underlying&          underlying,
        std::chrono::year_month_day           valuation_date);

    /** Column names required by each file, exposed for docs and tests. */
    struct Columns {
        std::vector<std::string> metadata;
        std::vector<std::string> contract;
    };
    [[nodiscard]] static Columns columns();
};

} // namespace ore::marketdata
