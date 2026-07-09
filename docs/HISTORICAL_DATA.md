# Historical Market Data

This document describes the `ore::marketdata` extension that turns the
single-day option-chain snapshot into a time series. The completed
module lets research code iterate over hundreds or thousands of
trading days without changing the pricing or analytics APIs.

The milestone is **infrastructure only**: no backtesting, realised
volatility, PnL, or model validation is implemented here. Those layers
sit on top of the objects introduced below.

---

## 1. Scope

Given a directory tree of per-day snapshots, the library should
support:

```cpp
HistoricalDataset dataset = HistoricalLoader::load("SPY");
for (const auto& snapshot : dataset) {
    PricingResult r = engine.price(option, snapshot.market());
}
```

The design must extend naturally to multiple tickers (AAPL, QQQ, TSLA)
and to date-range queries.

---

## 2. Directory layout

```
data/
    historical/
        raw/
            SPY/
                2024-01-02/
                    metadata.csv
                    calls.csv
                    puts.csv
                2024-01-03/
                    metadata.csv
                    calls.csv
                    puts.csv
                ...
                underlying_history.csv   # optional, whole-range spot batch
            AAPL/
                2024-01-02/
                    ...
        processed/                        # reserved for future milestones
```

Two things to notice:

1. **No intermediate `options/` directory.** The historical layout
   puts date-directories immediately under the ticker. This keeps the
   tree short and lets one directory represent one day.
2. **Same per-day file schema as the existing `data/raw/<T>/options/`
   layout.** The C++ loader reuses `YahooOptionLoader::load` verbatim —
   see §4.

---

## 3. C++ API

### 3.1 Data types

| Type | Header | Purpose |
|---|---|---|
| `HistoricalSnapshot` | `ore/marketdata/historical_snapshot.hpp` | One `(date, OptionChain)`. |
| `HistoricalDataset` | `ore/marketdata/historical_dataset.hpp` | Ordered vector of snapshots for one ticker. |
| `DatasetStatistics` | `ore/marketdata/historical_dataset.hpp` | Summary of a loaded dataset (counts, date range, missing days, parse failures). |

`HistoricalSnapshot` is a thin wrapper around `OptionChain`. `market()`,
`options()`, and `underlying()` accessors delegate to the wrapped
chain so callers never need to reach into it. The `date()` field is
exposed as a first-class accessor for chronological ordering and
lookup.

`HistoricalDataset` enforces its ordering invariant (ascending
`date()`) at construction — consumers can rely on
`dataset[i].date() < dataset[i+1].date()` without re-sorting.
Duplicate dates are permitted; the loader never produces them but the
container does not judge.

### 3.2 Loader

| Function | Purpose |
|---|---|
| `HistoricalLoader::load(ticker)` | Strict, default root, full range. Throws `LoaderError` on any parse failure. |
| `HistoricalLoader::load(ticker, Options)` | Full-control overload. Returns `LoadResult` with the dataset plus diagnostic vectors. |
| `HistoricalLoader::list_dates(ticker, root)` | Enumerate the trading dates on disk for a ticker. Silently skips names that don't parse as `YYYY-MM-DD`. |
| `HistoricalLoader::load_day(ticker, date, root)` | Load exactly one day's snapshot. |

`Options` controls:

```cpp
struct Options {
    std::filesystem::path root{"data/historical/raw"};
    std::optional<std::chrono::year_month_day> start_date{};
    std::optional<std::chrono::year_month_day> end_date{};
    bool strict{true};
};
```

`LoadResult` bundles the outputs:

```cpp
struct LoadResult {
    HistoricalDataset dataset;
    std::vector<std::pair<year_month_day, std::string>> failed_days;
    std::vector<year_month_day> missing_dates;
};
```

`failed_days` is populated only in lenient mode (`strict = false`) or
never non-empty in strict mode (the first failure throws). Every entry
is a `(date, exception::what())` pair so downstream code can render
a diagnostic without re-parsing anything.

`missing_dates` is computed against the *business-day* calendar (Mon-Fri)
in the loaded range, minus the union of `dataset` dates and
`failed_days` dates. NYSE holidays are **not** filtered — see §7.

### 3.3 Parsing

The loader does **not** duplicate parsing logic. `load_day` computes
the on-disk directory path (`<root>/<ticker>/<YYYY-MM-DD>/`) and
forwards to `YahooOptionLoader::load`. That function's cross-check
between the directory name and `metadata.valuation_date` is still
applied. Its ancillary check between the grandparent directory name
and the metadata `underlying_symbol` is silently skipped for the
historical layout (the grandparent is the ticker, not `options/`),
which is the correct behaviour for the new tree.

Adding a Polygon or Databento backend later means implementing a
sibling class with the same shape as `YahooOptionLoader` and having
`HistoricalLoader` dispatch to it based on a runtime option. The
public API of this class will not change.

---

## 4. Data-integrity guarantees

Every day loaded by `HistoricalLoader` inherits the validation done
by `YahooOptionLoader` in strict mode:

- Required columns are present.
- `strike > 0`, `bid >= 0`, `ask >= 0`.
- `bid > ask && bid > 0` — crossed market — is rejected.
- `implied_volatility >= 0` when present (NaN / empty is tolerated).
- `volume >= 0`, `open_interest >= 0`.
- `option_type` parses to `"call"` or `"put"`.
- `expiration`, `valuation_date`, `last_trade_date` parse as ISO 8601.
- `expiration >= valuation_date` (rejects stale contracts).
- `contract_symbol` non-empty and unique within one file.
- Directory name matches `metadata.valuation_date`.

`HistoricalLoader` adds:

- Directory-name parses as `YYYY-MM-DD`. Malformed names are silently
  skipped by `list_dates`.
- Non-directory entries (`README.md`, `.DS_Store`, `underlying_history.csv`)
  are ignored.
- Missing tickers (no such directory) yield an empty dataset without
  raising — that keeps the "have data for this ticker?" question a
  well-formed query.

---

## 5. Statistics

`HistoricalDataset::stats()` returns:

| Field | Semantics |
|---|---|
| `ticker` | As constructed. |
| `trading_days` | `dataset.size()`. |
| `contracts_loaded` | Sum of `snapshot.size()` across the dataset. |
| `average_contracts_per_day` | `contracts_loaded / trading_days`, or 0 for empty. |
| `first_date`, `last_date` | Earliest / latest observed date; nullopt for empty. |
| `parse_failures` | Loader-only; left 0 by `stats()`. |
| `missing_dates` | Loader-only; left empty by `stats()`. |

The loader-only fields are filled in on `LoadResult`, not on
`stats()`. Callers merging both are expected to overwrite the loader
fields:

```cpp
auto stats = result.dataset.stats();
stats.parse_failures = result.failed_days.size();
stats.missing_dates  = result.missing_dates;
stats.print(std::cout);
```

The `DatasetStatistics::print` helper renders a compact multi-line
diagnostic — used by `examples/load_historical_dataset.cpp` and by any
CI check that wants a canonical human-readable dump.

---

## 6. Acquisition — `python/download_historical_data.py`

`yfinance` does not publish historical option-chain data. Its
`Ticker.option_chain(expiration)` returns *only* the current-moment
snapshot. Anything older than the current market close is unavailable
through the free API. This is a fundamental provider limitation.

The downloader therefore does two things per invocation:

1. **Batch-downloads the historical spot** of the underlying for the
   full `[start, end]` range in one `yfinance.download(...)` call and
   writes it to `data/historical/raw/<TICKER>/underlying_history.csv`.
   One HTTP round-trip covers years of daily prices.
2. **Fetches the current option chain** once and writes it into
   `data/historical/raw/<TICKER>/<today>/{metadata.csv, calls.csv, puts.csv}`.

Historical option-chain coverage grows organically: the daily
downloader is intended to be run every trading day (via cron / GitHub
Actions / Task Scheduler), each run adding one directory. Users with a
paid provider can drop their own per-day directories into the same
tree and the C++ loader will pick them up without any code changes.

### Usage

```
python download_historical_data.py \
    --ticker SPY \
    --start 2024-01-01 \
    --end 2024-12-31
```

`--start` and `--end` are optional. Without them the underlying-history
batch defaults to the ticker's full available range and the current
option chain is still saved.

The command is safe to re-run: existing per-day directories are
never overwritten, and `underlying_history.csv` merges with the
existing file (never overwrites historical rows). Same safety
semantics as `download_option_chain.py` and `download_stock_history.py`.

---

## 7. Limitations

- **NYSE holidays.** `missing_dates` filters weekends but not
  half-days or full-day exchange holidays. A `docs/nyse_holidays.csv`
  and a holiday-aware filter will land alongside a future backtest
  milestone; the current design keeps the C++ side calendar-free.
- **Provider-only option-chain history.** As noted above, yfinance
  cannot backfill historical option chains. Populating a multi-year
  dataset requires either a paid provider or a cron job that has been
  running long enough to have accumulated the data.
- **Cross-day consistency.** Two adjacent days may report subtly
  different `dividend_yield` or `risk_free_rate`, because
  `download_option_chain.py` computes those from Yahoo's `Ticker.info`
  at run time. This is by design: the loader stores whatever was
  written, and downstream code is responsible for smoothing.
- **No caching.** Every load is a fresh filesystem traversal. Adding
  an in-memory cache is a mechanical change (`HistoricalDataset` is
  moveable and cache-friendly) but is deliberately deferred until a
  research workload identifies it as a bottleneck.

---

## 8. Future extensions

- A `HistoricalLoaderBackend` polymorphic interface with pluggable
  provider adapters (`YahooBackend`, `PolygonBackend`,
  `DatabentoBackend`). The public `HistoricalLoader::load` API stays
  identical.
- A `HistoricalCache` layer that memoises per-day snapshots keyed by
  `(root, ticker, date)`.
- Multi-ticker loaders — `load_all(std::vector<ticker>)` returning a
  `std::unordered_map<std::string, HistoricalDataset>`.
- A trading-calendar dependency for holiday-aware missing-day
  reporting.

None of these change the shape of `HistoricalSnapshot` or
`HistoricalDataset`, so downstream research code written today will
still compile against the extended API.
