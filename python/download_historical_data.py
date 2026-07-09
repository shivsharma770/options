"""
Download historical market data for a single ticker.

Rationale
---------

`yfinance` does **not** expose historical option chains. Its
`Ticker.option_chain(expiration)` returns the current-moment snapshot
only. Historical option data can only be accumulated by running this
downloader daily (e.g. via cron / GitHub Actions / Task Scheduler) or
by bringing in a paid provider (Polygon, Databento, ...).

What this script *can* do in one invocation:

1. **Batch-download historical spot** of the underlying for the full
   requested date range in a single yfinance call and write it to
   ``data/historical/raw/<TICKER>/underlying_history.csv``. Merges
   with any existing file (never overwrites historical rows).
2. **Fetch the current option chain** and write it to
   ``data/historical/raw/<TICKER>/<today>/{metadata.csv, calls.csv,
   puts.csv}``. Skips silently if today's directory already exists.

Batching (rather than one HTTP request per day) is a deliberate
trade-off: yfinance requests are slow, and a multi-year download in a
single call is orders of magnitude faster than per-day fetching.

Output layout
-------------

::

    <repo_root>/data/historical/raw/<TICKER>/
        underlying_history.csv        # batch-downloaded, whole range
        <YYYY-MM-DD>/                 # created for today only
            metadata.csv
            calls.csv
            puts.csv

Once the tree has enough data, load it from C++ via::

    HistoricalDataset ds = HistoricalLoader::load("SPY");

Usage
-----

::

    python download_historical_data.py --ticker SPY
    python download_historical_data.py --ticker SPY \
                                       --start 2024-01-01 \
                                       --end 2024-12-31

Both ``--start`` and ``--end`` are optional. Without them the
underlying-history batch runs with ``period="max"`` and the current
option chain is still saved.

Dependencies: pandas, yfinance.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# Reuse the existing option-chain schema + normaliser so the CSVs
# written by this script are pixel-identical to what
# `download_option_chain.py` produces. Importing rather than
# duplicating keeps parity guaranteed.
try:
    from download_option_chain import (  # type: ignore[import-not-found]
        CONTRACT_COLUMNS,
        _normalise_contract_frame,
        build_metadata_row,
    )
except ImportError:
    # When run outside the `python/` directory the sibling import may
    # need help finding itself; fall back to path injection.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from download_option_chain import (  # type: ignore[import-not-found]  # noqa: E402
        CONTRACT_COLUMNS,
        _normalise_contract_frame,
        build_metadata_row,
    )


REPO_ROOT: Path = Path(__file__).resolve().parent.parent
HISTORICAL_ROOT: Path = REPO_ROOT / "data" / "historical" / "raw"

# Columns for the batch-downloaded underlying history file. Matches
# yfinance's own history() output and the existing
# `download_stock_history.py` script; the historical dataset can share
# tooling with that script if we want.
HISTORY_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True,
                        help="Ticker symbol (e.g. SPY, AAPL, QQQ).")
    parser.add_argument("--start", type=str, default=None,
                        help="Inclusive start date (YYYY-MM-DD). "
                             "Omit for `period=max`.")
    parser.add_argument("--end", type=str, default=None,
                        help="Inclusive end date (YYYY-MM-DD). "
                             "Omit for `period=max`.")
    parser.add_argument("--root", type=Path, default=HISTORICAL_ROOT,
                        help="Historical raw-data root. Defaults to "
                             "<repo>/data/historical/raw.")
    parser.add_argument("--skip-option-chain", action="store_true",
                        help="Only refresh the underlying history; do "
                             "not touch today's option chain.")
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# Underlying history
# -----------------------------------------------------------------------------

def _iso_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    # Validate it parses; propagate the ISO representation exactly.
    datetime.strptime(value, "%Y-%m-%d")
    return value


def fetch_underlying_history(ticker: str,
                             start: str | None,
                             end: str | None) -> pd.DataFrame:
    """Batch-download daily spot for the whole requested range."""
    kwargs = {
        "tickers":    ticker,
        "interval":   "1d",
        "auto_adjust": False,
        "progress":   False,
        "actions":    False,
        "threads":    False,
    }
    if start is None and end is None:
        kwargs["period"] = "max"
    else:
        kwargs["start"] = start
        kwargs["end"]   = end
    df = yf.download(**kwargs)
    if df is None or df.empty:
        raise RuntimeError(
            f"yfinance returned no history for {ticker!r} "
            f"(start={start}, end={end})")

    # yfinance may return a MultiIndex for multi-ticker calls; single-
    # ticker calls return flat columns. Normalise defensively.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index().rename(columns={"index": "Date"})
    missing = [c for c in HISTORY_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"yfinance history for {ticker!r} is missing columns: {missing}")
    df = df[HISTORY_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df


def merge_history(new_df: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    """Append rows in `new_df` that are not already in `existing_path`."""
    if not existing_path.exists():
        return new_df
    existing = pd.read_csv(existing_path, dtype={"Date": str})
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date"], keep="first")
    combined = combined.sort_values("Date").reset_index(drop=True)
    return combined


def save_underlying_history(ticker: str,
                            start: str | None,
                            end: str | None,
                            root: Path) -> Path:
    df = fetch_underlying_history(ticker, start, end)
    output_path = root / ticker / "underlying_history.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = merge_history(df, output_path)
    merged.to_csv(output_path, index=False)
    print(f"[historical] {ticker}: underlying_history.csv now has "
          f"{len(merged)} rows -> {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# Today's option chain
# -----------------------------------------------------------------------------

def save_todays_option_chain(ticker: str, root: Path) -> Path | None:
    """Fetch and save today's option chain into <root>/<ticker>/<today>/."""
    today = date.today()
    output_dir = root / ticker / today.isoformat()
    if output_dir.exists():
        print(f"[historical] {ticker}: option snapshot for {today} already "
              f"exists — skipping (delete the directory to re-download)")
        return None

    ticker_obj = yf.Ticker(ticker)
    expirations = list(ticker_obj.options or [])
    if not expirations:
        raise RuntimeError(f"No option expirations returned for {ticker!r}")

    metadata = build_metadata_row(ticker, ticker_obj, today)

    calls_frames: list[pd.DataFrame] = []
    puts_frames:  list[pd.DataFrame] = []
    for expiration in expirations:
        chain = ticker_obj.option_chain(expiration)
        calls_frames.append(_normalise_contract_frame(chain.calls, expiration, "call"))
        puts_frames.append(_normalise_contract_frame(chain.puts,  expiration, "put"))

    calls_df = (pd.concat(calls_frames, ignore_index=True)
                if calls_frames else pd.DataFrame(columns=CONTRACT_COLUMNS))
    puts_df  = (pd.concat(puts_frames,  ignore_index=True)
                if puts_frames  else pd.DataFrame(columns=CONTRACT_COLUMNS))

    output_dir.mkdir(parents=True, exist_ok=False)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    calls_df.to_csv(output_dir / "calls.csv",   index=False)
    puts_df.to_csv(output_dir / "puts.csv",    index=False)
    print(f"[historical] {ticker}: saved {len(calls_df)} calls, "
          f"{len(puts_df)} puts across {len(expirations)} expirations "
          f"-> {output_dir}")
    return output_dir


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = _iso_or_none(args.start)
    end   = _iso_or_none(args.end)

    try:
        save_underlying_history(args.ticker, start, end, args.root)
    except Exception as exc:  # noqa: BLE001
        print(f"[historical] FAILED underlying history for {args.ticker}: {exc}",
              file=sys.stderr)
        return 1

    if not args.skip_option_chain:
        try:
            save_todays_option_chain(args.ticker, args.root)
        except Exception as exc:  # noqa: BLE001
            print(f"[historical] FAILED option chain for {args.ticker}: {exc}",
                  file=sys.stderr)
            return 1
    else:
        print("[historical] --skip-option-chain: not fetching option chain.")

    # A note the user should see so they don't wonder why the tree isn't
    # populated with years of options data.
    print("[historical] Note: yfinance does not expose historical option "
          "chains. Run this daily (cron/CI/Task Scheduler) to accumulate "
          "coverage, or drop per-day directories from a paid provider "
          "into the same tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
