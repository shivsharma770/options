# Python (Milestone 2 — market-data downloaders)

Python is responsible for **downloading market data only**. It writes CSV
files that the C++ engine reads. There is no runtime communication between
Python and C++.

## Scripts

| Script                       | Output                                                                                   |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| `download_stock_history.py`  | `../data/raw/<TICKER>/stock/history.csv` — full daily OHLCV, append-only                 |
| `download_option_chain.py`   | `../data/raw/<TICKER>/options/<YYYY-MM-DD>/{metadata,calls,puts}.csv` — daily snapshot   |

Both scripts have a `TICKERS = [...]` list at the top. Edit that list and
run the script — there are no CLI arguments by design.

## Install

Create a fresh virtual environment (recommended so `yfinance` and `pandas`
don't collide with anything else on your system):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
cd python
python download_stock_history.py
python download_option_chain.py
```

Each snapshot lands in its own dated directory. **Existing snapshots are
never overwritten** — delete the snapshot directory manually if you want to
re-download for the same date. Stock history is merged non-destructively:
existing rows are preserved, new rows are appended.

## What Python is *not* allowed to do

- No pricing, no Greeks, no analytics — the C++ engine owns all numerics.
- No plotting inside these scripts. Notebooks and plots come in a later
  milestone under `python/notebooks/` (not yet created).
- No runtime IPC with the C++ layer. Files are the only interface.

## Missing pieces (documented, not yet solved)

- **Risk-free rate**: Yahoo does not publish one, so
  `download_option_chain.py` writes `0.0` into `metadata.csv`. A future
  script (FRED downloader) will replace that value; for now, users can
  edit the metadata CSV by hand.
- **Exercise style**: not currently written by Python and not currently
  read by the C++ loader. The loader defaults every contract to European
  regardless of the underlying — a simplification that will be revisited
  when American-style pricing (binomial / FD) lands in a later milestone.
