"""
Download historical daily stock prices via yfinance.

Output layout
-------------
For each ticker `T`, produces / updates:

    <repo_root>/data/raw/<T>/stock/history.csv

with columns:

    Date, Open, High, Low, Close, Adj Close, Volume

If the file already exists, existing dates are preserved and only newer rows
are appended (never overwrites historical values). If the file does not
exist, the entire available history is downloaded.

Usage
-----
Edit the `TICKERS` list below, then run:

    python download_stock_history.py

No CLI arguments — the script is intentionally a small, editable Python
file rather than a general-purpose command-line tool.

Design notes
------------
- The C++ side of the engine reads these files. Nothing here does analysis.
- Failing on one ticker does not abort the others.
- No logging framework; failures print a single line to stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---- Configuration ---------------------------------------------------------

TICKERS: list[str] = [
    "SPY",
]

# All output goes under this directory. Path is resolved relative to this
# script so the downloader runs correctly from any working directory.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_ROOT: Path = REPO_ROOT / "data" / "raw"

# ---- Output CSV schema -----------------------------------------------------
# These column names match what the C++ engine will parse. yfinance's
# DataFrame index is the date; we materialize it into a `Date` column.
HISTORY_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def fetch_history(ticker: str) -> pd.DataFrame:
    """Fetch the full available daily history for `ticker`."""
    df = yf.download(
        tickers=ticker,
        period="max",
        interval="1d",
        auto_adjust=False,      # keep Close and Adj Close as separate columns
        progress=False,
        actions=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no history for {ticker!r}")

    # yfinance may return a MultiIndex column layout when multiple tickers
    # are requested; single-ticker calls return a flat layout. Normalise
    # either way to the canonical column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index().rename(columns={"index": "Date"})
    missing = [c for c in HISTORY_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"yfinance history for {ticker!r} is missing columns: {missing}"
        )
    df = df[HISTORY_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df


def merge_with_existing(new_df: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    """Append rows in `new_df` that are not already present in `existing_path`."""
    if not existing_path.exists():
        return new_df

    existing = pd.read_csv(existing_path, dtype={"Date": str})
    combined = pd.concat([existing, new_df], ignore_index=True)
    # Keep first occurrence of each date to avoid overwriting historical rows.
    combined = combined.drop_duplicates(subset=["Date"], keep="first")
    combined = combined.sort_values("Date").reset_index(drop=True)
    return combined


def write_history(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def download_ticker(ticker: str) -> None:
    output_path = OUTPUT_ROOT / ticker / "stock" / "history.csv"
    new_df = fetch_history(ticker)
    merged = merge_with_existing(new_df, output_path)
    write_history(merged, output_path)
    print(f"[stock/history] {ticker}: wrote {len(merged)} rows -> {output_path}")


def main() -> int:
    failures = 0
    for ticker in TICKERS:
        try:
            download_ticker(ticker)
        except Exception as exc:                                                 # noqa: BLE001
            print(f"[stock/history] FAILED {ticker}: {exc}", file=sys.stderr)
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
