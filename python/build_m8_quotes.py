#!/usr/bin/env python3
"""
build_m8_quotes.py -- ETL for Research Milestone 8, Phase 6 (quote behavior).

Extracts a compact per-day quote-quality panel from calibration.csv (bid/ask/mid
of every calibrated contract) so Phase 6 can ask whether liquidity providers
widen spreads / withdraw quotes around IV shocks. The bulky per-run CSVs are
deleted immediately afterward. No new market data -- reuses the existing
HistoricalCalibrationStudy output.

Per trading day: number of calibrated quotes, and the mean/median relative
bid-ask spread (ask-bid)/mid over contracts with a finite positive mid.

Usage:  build_m8_quotes.py <research_dir> <master_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 1_000_000


def main():
    research = Path(sys.argv[1]); master = Path(sys.argv[2]); master.mkdir(parents=True, exist_ok=True)
    agg = {}   # date -> [n, sum_relspread, list? -> use running sum + count; median via approx]
    for ch in pd.read_csv(research / "calibration.csv",
                          usecols=["date", "bid", "ask", "mid_price"], chunksize=CHUNK):
        ch = ch[(ch["mid_price"] > 0) & (ch["ask"] >= ch["bid"])]
        ch["rs"] = (ch["ask"] - ch["bid"]) / ch["mid_price"]
        g = ch.groupby("date")["rs"].agg(["sum", "count", "median"])
        for d, row in g.iterrows():
            a = agg.setdefault(d, [0.0, 0, []])
            a[0] += row["sum"]; a[1] += int(row["count"]); a[2].append((row["median"], int(row["count"])))
    rows = []
    for d, (s, n, meds) in agg.items():
        # count-weighted average of chunk medians as a robust median proxy
        w = sum(c for _, c in meds); med = sum(m * c for m, c in meds) / w if w else np.nan
        rows.append({"date": d, "quote_count": n, "mean_rel_spread": s / n if n else np.nan,
                     "median_rel_spread": med})
    out = pd.DataFrame(rows).sort_values("date")
    header = not (master / "m8_quotes.csv").exists()
    out.to_csv(master / "m8_quotes.csv", mode="a", header=header, index=False)
    print(f"  {research.name}: {len(out)} days")


if __name__ == "__main__":
    main()
