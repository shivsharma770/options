#!/usr/bin/env python3
"""
build_m4_greeks_panel.py -- ETL for Research Milestone 4 (higher-order Greeks).

Extracts, from one run of `HistoricalCalibrationStudy`, the ~1-month smile as a
compact per-day panel of (moneyness, strike, implied vol) points, so higher-order
Greeks can be computed downstream at every strike -- crucial because the
wing-sensitive Greeks (Vanna, Vomma, Ultima) vanish at the money and only reveal
themselves across the smile. The bulky per-run CSVs are deleted immediately
afterward; the surviving panel is a few MB over the full archive.

Per trading day: pick the listed expiry nearest 30 calendar days, keep its
strikes with |ln(K/S)| <= 0.20, and record spot (recovered as S = K*e^{-m}).

Usage:  build_m4_greeks_panel.py <research_dir> <master_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = 30.0 / 365.0
MW = 0.20
CHUNK = 1_000_000


def main():
    research = Path(sys.argv[1])
    master = Path(sys.argv[2])
    master.mkdir(parents=True, exist_ok=True)

    parts = []
    for ch in pd.read_csv(research / "smiles.csv",
                          usecols=["date", "time_to_expiry", "strike",
                                   "moneyness", "implied_volatility"],
                          chunksize=CHUNK):
        band = ch[ch["moneyness"].abs().le(MW)
                  & ch["time_to_expiry"].between(15 / 365.0, 55 / 365.0)]
        if len(band):
            parts.append(band)
    df = pd.concat(parts, ignore_index=True)

    rows = []
    for d, g in df.groupby("date"):
        t_star = g["time_to_expiry"].iloc[(g["time_to_expiry"] - TARGET).abs().values.argmin()]
        sm = g[np.isclose(g["time_to_expiry"], t_star)].copy()
        sm["spot"] = sm["strike"] * np.exp(-sm["moneyness"])
        rows.append(sm)
    out = pd.concat(rows, ignore_index=True)[
        ["date", "time_to_expiry", "spot", "strike", "moneyness", "implied_volatility"]
    ].sort_values(["date", "strike"])

    header = not (master / "m4_smile30.csv").exists()
    out.to_csv(master / "m4_smile30.csv", mode="a", header=header, index=False)
    print(f"  {research.name}: {out['date'].nunique()} days, {len(out)} smile points")


if __name__ == "__main__":
    main()
