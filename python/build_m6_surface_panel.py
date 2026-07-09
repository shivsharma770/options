#!/usr/bin/env python3
"""
build_m6_surface_panel.py -- ETL for Research Milestone 6 (surface geometry).

Extracts, from one run of `HistoricalCalibrationStudy`, a compact *multi-maturity*
smile panel per trading day, so the implied-vol surface can be studied as a 2-D
geometric object downstream. The bulky per-run CSVs are deleted immediately after.

Per trading day, for each target maturity bucket (calendar days), pick the listed
expiry nearest that bucket and keep its calibrated smile points with
|ln(K/S)| <= 0.15. Output columns: date, mat_bucket, time_to_expiry, moneyness,
implied_volatility, spot.

Usage:  build_m6_surface_panel.py <research_dir> <master_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BUCKETS = [14, 30, 60, 90, 180]          # calendar-day maturity buckets
MW = 0.15
CHUNK = 1_000_000


def main():
    research = Path(sys.argv[1]); master = Path(sys.argv[2]); master.mkdir(parents=True, exist_ok=True)
    parts = []
    for ch in pd.read_csv(research / "smiles.csv",
                          usecols=["date", "time_to_expiry", "strike", "moneyness", "implied_volatility"],
                          chunksize=CHUNK):
        band = ch[ch["moneyness"].abs().le(MW)
                  & ch["time_to_expiry"].between(8 / 365.0, 230 / 365.0)]
        if len(band):
            parts.append(band)
    df = pd.concat(parts, ignore_index=True)

    rows = []
    for d, g in df.groupby("date"):
        tdays = g["time_to_expiry"].values * 365.0
        for b in BUCKETS:
            j = np.abs(tdays - b).argmin()
            if abs(tdays[j] - b) > 0.45 * b:     # no expiry near this bucket
                continue
            t_star = g["time_to_expiry"].values[j]
            sm = g[np.isclose(g["time_to_expiry"], t_star)].copy()
            sm["mat_bucket"] = b
            sm["spot"] = sm["strike"] * np.exp(-sm["moneyness"])
            rows.append(sm)
    out = pd.concat(rows, ignore_index=True)[
        ["date", "mat_bucket", "time_to_expiry", "moneyness", "implied_volatility", "spot"]
    ].sort_values(["date", "mat_bucket", "moneyness"])

    header = not (master / "m6_surface.csv").exists()
    out.to_csv(master / "m6_surface.csv", mode="a", header=header, index=False)
    print(f"  {research.name}: {out['date'].nunique()} days, {len(out)} surface points")


if __name__ == "__main__":
    main()
