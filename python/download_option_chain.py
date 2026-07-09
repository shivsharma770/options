"""
Download a full option chain snapshot via yfinance.

Output layout
-------------
For each ticker `T`, writes into a fresh, dated directory:

    <repo_root>/data/raw/<T>/options/<YYYY-MM-DD>/
        metadata.csv    # one row of market-wide state
        calls.csv       # all call contracts, all expirations
        puts.csv        # all put contracts, all expirations

`<YYYY-MM-DD>` is the valuation date (defaults to today). Existing
snapshots are **never** overwritten: if the directory already exists, the
ticker is skipped with a warning. This preserves historical snapshots for
later replay analysis.

Schema
------
See docs/csv_schema.md for the authoritative column definitions.
Column names are snake_case and match the C++ loader.

Usage
-----
Edit `TICKERS` at the top of this file, then run:

    python download_option_chain.py

No CLI arguments.

Design notes
------------
- Yahoo does **not** publish a risk-free rate. This script writes 0.0 by
  default; downstream code (or a future FRED downloader) is responsible
  for filling it in. The C++ loader will still accept the snapshot and
  emit a warning if you attempt to price without setting a real rate.
- `dividend_yield` comes from yfinance's `Ticker.info.get("dividendYield")`.
  Yahoo reports this in decimal form for equity tickers.
- Multi-ticker execution: failures on one ticker do not abort the others.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---- Configuration ---------------------------------------------------------

TICKERS: list[str] = [
    "SPY",
]

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_ROOT: Path = REPO_ROOT / "data" / "raw"

# ---- Output CSV schema -----------------------------------------------------
METADATA_COLUMNS = [
    "valuation_date",
    "underlying_symbol",
    "exchange",
    "asset_type",
    "spot",
    "dividend_yield",
    "risk_free_rate",
    "currency",
    "timezone",
]

CONTRACT_COLUMNS = [
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
    "change",
    "percent_change",
    "currency",
    "contract_size",
]

# ---- yfinance -> our schema field mapping ---------------------------------
# yfinance emits camelCase; we normalise to snake_case. Fields not listed
# here are dropped after the rename.
YFINANCE_RENAME = {
    "contractSymbol":     "contract_symbol",
    "lastTradeDate":      "last_trade_date",
    "strike":             "strike",
    "lastPrice":          "last",
    "bid":                "bid",
    "ask":                "ask",
    "change":             "change",
    "percentChange":      "percent_change",
    "volume":             "volume",
    "openInterest":       "open_interest",
    "impliedVolatility":  "implied_volatility",
    "inTheMoney":         "in_the_money",
    "contractSize":       "contract_size",
    "currency":           "currency",
}


def guess_asset_type(info: dict) -> str:
    """Very light-weight classification. Downstream C++ accepts the string as-is."""
    quote_type = str(info.get("quoteType", "")).lower()
    if quote_type in {"etf"}:
        return "ETF"
    if quote_type in {"index", "indexfund"}:
        return "Index"
    if quote_type in {"future"}:
        return "Future"
    if quote_type in {"equity"}:
        return "Equity"
    return "Other"


def _safe_float(value, default: float = 0.0) -> float:
    """Return a finite float or `default`. yfinance sometimes emits None / NaN."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def build_metadata_row(ticker: str, ticker_obj: yf.Ticker, valuation: date) -> pd.DataFrame:
    info = ticker_obj.info or {}

    spot = _safe_float(info.get("regularMarketPrice")) or _safe_float(
        info.get("previousClose"))
    if spot <= 0.0:
        raise RuntimeError(f"Could not determine a positive spot for {ticker!r}")

    row = {
        "valuation_date":     valuation.isoformat(),
        "underlying_symbol":  ticker,
        "exchange":           info.get("exchange", "") or "",
        "asset_type":         guess_asset_type(info),
        "spot":               spot,
        "dividend_yield":     _safe_float(info.get("dividendYield")),
        "risk_free_rate":     0.0,  # Not published by Yahoo; user must fill in.
        "currency":           info.get("currency", "") or "USD",
        "timezone":           info.get("exchangeTimezoneName", "") or "",
    }
    return pd.DataFrame([row], columns=METADATA_COLUMNS)


def _normalise_contract_frame(df: pd.DataFrame,
                              expiration: str,
                              option_type: str) -> pd.DataFrame:
    """Turn a raw yfinance option DataFrame into our canonical schema."""
    df = df.rename(columns=YFINANCE_RENAME).copy()
    df["expiration"] = expiration
    df["option_type"] = option_type

    # last_trade_date arrives as a pandas Timestamp. Serialize to ISO 8601
    # (UTC where possible) so the C++ parser can accept it uniformly.
    if "last_trade_date" in df.columns:
        df["last_trade_date"] = pd.to_datetime(df["last_trade_date"], utc=True, errors="coerce")
        df["last_trade_date"] = df["last_trade_date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Missing-value handling: yfinance emits NaN for missing volume /
    # openInterest. We fill with 0 to keep the C++ parser (which reads them
    # as required non-negative floats) happy; downstream code should treat
    # 0 as "unknown / no trades" rather than as ground truth.
    for numeric in ("volume", "open_interest"):
        if numeric in df.columns:
            df[numeric] = pd.to_numeric(df[numeric], errors="coerce").fillna(0).astype(int)

    for float_col in ("bid", "ask", "last", "change", "percent_change", "implied_volatility"):
        if float_col in df.columns:
            df[float_col] = pd.to_numeric(df[float_col], errors="coerce").fillna(0.0)

    # in_the_money can be Python bool. Emit as lowercase "true"/"false"
    # to match the CSV boolean grammar accepted by CsvRow::as_bool.
    if "in_the_money" in df.columns:
        df["in_the_money"] = df["in_the_money"].astype(bool).map({True: "true", False: "false"})

    # Ensure every schema column exists; add empty ones if Yahoo didn't
    # provide them (rare but possible).
    for c in CONTRACT_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    return df[CONTRACT_COLUMNS].copy()


def download_ticker(ticker: str, valuation: date) -> None:
    output_dir = OUTPUT_ROOT / ticker / "options" / valuation.isoformat()
    if output_dir.exists():
        print(f"[options] {ticker}: snapshot {output_dir.name} already exists — skipping "
              "(delete the directory manually to re-download)")
        return

    ticker_obj = yf.Ticker(ticker)
    expirations = list(ticker_obj.options or [])
    if not expirations:
        raise RuntimeError(f"No option expirations returned for {ticker!r}")

    metadata = build_metadata_row(ticker, ticker_obj, valuation)

    calls_frames: list[pd.DataFrame] = []
    puts_frames: list[pd.DataFrame] = []
    for expiration in expirations:
        chain = ticker_obj.option_chain(expiration)
        calls_frames.append(_normalise_contract_frame(chain.calls, expiration, "call"))
        puts_frames.append(_normalise_contract_frame(chain.puts, expiration, "put"))

    calls_df = pd.concat(calls_frames, ignore_index=True) if calls_frames else pd.DataFrame(columns=CONTRACT_COLUMNS)
    puts_df  = pd.concat(puts_frames,  ignore_index=True) if puts_frames  else pd.DataFrame(columns=CONTRACT_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=False)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    calls_df.to_csv(output_dir / "calls.csv",   index=False)
    puts_df.to_csv(output_dir / "puts.csv",    index=False)

    print(f"[options] {ticker}: wrote {len(calls_df)} calls, {len(puts_df)} puts across "
          f"{len(expirations)} expirations -> {output_dir}")


def main() -> int:
    valuation = date.today()
    failures = 0
    for ticker in TICKERS:
        try:
            download_ticker(ticker, valuation)
        except Exception as exc:                                                 # noqa: BLE001
            print(f"[options] FAILED {ticker}: {exc}", file=sys.stderr)
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
