# CSV Schema (Milestone 2)

Authoritative definition of every CSV file the market-data pipeline
produces and consumes. Column names are **snake_case**. Column order in
the files is stable but the C++ loader is column-name-driven — new columns
can be appended by producers without breaking existing readers.

**Encoding**: UTF-8. An optional leading BOM is tolerated. Line endings
`\n` or `\r\n`.
**Numeric convention**: dot-decimal, no thousands separators, no currency
symbols. Missing numerics are written as `0` or an empty field where
documented; `NaN` / `inf` are **not** valid.
**Date convention**: ISO 8601 (`YYYY-MM-DD`). Timestamps: extended ISO
8601 (`YYYY-MM-DDTHH:MM:SSZ`).
**Booleans**: `true` / `false` (case-insensitive; `1`/`0` and `yes`/`no`
are also accepted by the reader).

---

## 1. `data/raw/<TICKER>/stock/history.csv`

Daily OHLCV history for a single ticker. Continuous time series; append-only.

| Column       | Type    | Description                                  |
| ------------ | ------- | -------------------------------------------- |
| `Date`       | date    | Trading date (ISO 8601)                      |
| `Open`       | float   | Opening price                                |
| `High`       | float   | Session high                                 |
| `Low`        | float   | Session low                                  |
| `Close`      | float   | Closing price                                |
| `Adj Close`  | float   | Split- and dividend-adjusted close           |
| `Volume`     | int     | Traded volume                                |

**Historical mutability**: existing rows are never overwritten by
`download_stock_history.py`. Only *new* dates are appended.

Column names retain Yahoo's capitalisation (`Adj Close` with a space) so
that a raw pandas read gives the same layout Yahoo produces.

---

## 2. `data/raw/<TICKER>/options/<YYYY-MM-DD>/metadata.csv`

Exactly **one** data row describing market state at the snapshot moment.

| Column               | Type    | Required | Description                                                                     |
| -------------------- | ------- | -------- | ------------------------------------------------------------------------------- |
| `valuation_date`     | date    | yes      | Snapshot date (must equal the parent directory's name)                          |
| `underlying_symbol`  | string  | yes      | Ticker (must equal the grandparent directory when the canonical layout is used) |
| `exchange`           | string  | no       | Listing exchange, e.g. `NYSEARCA`, `NASDAQ`. Empty if unknown.                   |
| `asset_type`         | string  | no       | One of `Equity`, `Index`, `ETF`, `Future`, `FX`, `Other`. Default `Equity`.     |
| `spot`               | float   | yes      | Underlying spot price at valuation. **Must be > 0 and finite.**                 |
| `dividend_yield`     | float   | yes      | Continuous dividend yield (decimal, ACT/365F). Must be finite.                  |
| `risk_free_rate`     | float   | yes      | Continuous risk-free rate (decimal, ACT/365F). Must be finite. Yahoo does not publish this; Python writes `0.0` and the user is expected to fill it in. |
| `currency`           | string  | no       | Currency of `spot`. Empty defaults to `USD` on read.                            |
| `timezone`           | string  | no       | IANA name of the exchange timezone.                                             |

---

## 3. `data/raw/<TICKER>/options/<YYYY-MM-DD>/calls.csv` and `puts.csv`

One row per listed contract at the snapshot moment. Both files share the
same schema; `option_type` is redundant with the filename but kept so each
file is self-describing.

| Column                | Type    | Required | Description                                                                       |
| --------------------- | ------- | -------- | --------------------------------------------------------------------------------- |
| `contract_symbol`     | string  | yes      | Provider identifier, e.g. `SPY260808C00470000`. **Non-empty, unique within file.**|
| `expiration`          | date    | yes      | Contract expiration date. **Must be ≥ `valuation_date` in metadata.**             |
| `strike`              | float   | yes      | **Must be > 0 and finite.**                                                        |
| `option_type`         | string  | yes      | `call` or `put` (case-insensitive). Must agree with the filename.                 |
| `bid`                 | float   | yes      | Best bid. **Finite and ≥ 0.**                                                      |
| `ask`                 | float   | yes      | Best ask. **Finite and ≥ 0. If `bid > 0`, then `ask ≥ bid`.**                      |
| `last`                | float   | yes      | Last traded price.                                                                |
| `volume`              | float   | yes      | Session traded volume. **Finite and ≥ 0.** (Stored as float to tolerate NaN-for-missing round-trips at the pandas layer; the CSV should contain a non-negative number.) |
| `open_interest`       | float   | yes      | Open interest. **Finite and ≥ 0.**                                                 |
| `implied_volatility`  | float   | yes      | Yahoo-reported IV. May be `0.0` for illiquid contracts. **Negative values are rejected.** |
| `in_the_money`        | bool    | yes      | Provider-reported moneyness flag. Validated for parseability; not stored on the domain object (moneyness is derived from `strike` vs. `spot`). |
| `last_trade_date`     | timestamp | yes    | Last trade time (ISO 8601). Truncated to date on the C++ side.                    |
| `change`              | float   | no       | Daily price change (Yahoo).                                                       |
| `percent_change`      | float   | no       | Daily percentage change (Yahoo).                                                  |
| `currency`            | string  | no       | Contract currency (Yahoo).                                                        |
| `contract_size`       | string  | no       | Contract size string, typically `REGULAR` (Yahoo).                                |

---

## Validation rules enforced by `ore::marketdata::YahooOptionLoader`

The loader is **strict** in Milestone 2: any violation throws
`ore::marketdata::LoaderError`. The exhaustive list:

1. Required column is absent from the header.
2. `valuation_date`, `expiration`, or `last_trade_date` is not ISO 8601.
3. `spot`, `strike`, `bid`, `ask`, `dividend_yield`, `risk_free_rate`,
   `volume`, `open_interest`, or `implied_volatility` is not a finite
   floating-point number.
4. `strike ≤ 0`, `spot ≤ 0`, or any of `bid`, `ask`, `volume`,
   `open_interest`, `implied_volatility` is negative.
5. `bid > ask` while both are strictly positive (crossed market). The
   pair `(0.0, 0.0)` — "no market" — is allowed.
6. `option_type` is not `call` or `put`, or disagrees with the file it
   was loaded from.
7. `expiration < valuation_date` (stale contract in the snapshot).
8. `contract_symbol` is empty, or a duplicate value appears in the same
   file.
9. `metadata.csv` contains zero or more than one data row.
10. Directory-name checks:
    - Snapshot directory basename must equal `valuation_date`.
    - When the canonical `<SYMBOL>/options/<date>/` layout is used, the
      grandparent directory's name must equal `underlying_symbol`.

## Future schema evolution

- `exercise_style` will become an optional metadata column once
  American-style pricing (binomial / FD) lands. Absent → default European.
- A `risk_free_rate_source` string column may be added when a FRED
  downloader is integrated, so the origin of the rate is auditable.
