#!/usr/bin/env python3
"""
build_m2_features.py -- ETL for Research Milestone 2 (smile-shape features).

Extracts the ~1-month smile-shape features from one run of
`HistoricalCalibrationStudy` and appends them to a master file, so the bulky
per-run CSVs can be deleted immediately. Reuses the C++ `volatility_analytics`
outputs directly:

  * 25Δ risk reversal, 25Δ butterfly, ATM IV -- read straight from skew.csv
    (already computed by `compute_skew_metrics`; NOT recomputed here).
  * smile slope (dσ/dm) and curvature (d²σ/dm²) -- least-squares quadratic
    σ = a + b·m + c·m² fitted to the ~1M smile in log-moneyness m = ln(K/S),
    the same fit used by the SPY-2020 study.

Term-structure slopes are derived later, in the analysis, from the existing
`atm_term_structure.csv` master (all tenors per day) -- no re-extraction needed.

Usage:  build_m2_features.py <research_dir> <master_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = 30.0 / 365.0                 # ~1-month reference tenor
BAND = (18.0 / 365.0, 45.0 / 365.0)   # expiries admitted to the smile fit
MW = 0.25                             # |ln(K/S)| window
CHUNK = 1_000_000


def nearest_per_date(df, target, col="time_to_expiry"):
    d = df.copy()
    d["_dist"] = (d[col] - target).abs()
    return d.loc[d.groupby("date")["_dist"].idxmin()].drop(columns="_dist")


def main():
    research = Path(sys.argv[1])
    master = Path(sys.argv[2])
    master.mkdir(parents=True, exist_ok=True)

    # --- 25d RR / BF / ATM from skew.csv (C++ volatility_analytics output) ----
    sk = pd.read_csv(research / "skew.csv")
    sk = sk[(sk["time_to_expiry"] > 0)]
    sk = nearest_per_date(sk, TARGET)[
        ["date", "atm_iv", "risk_reversal", "butterfly"]
    ].rename(columns={"atm_iv": "atm_iv", "risk_reversal": "rr25", "butterfly": "bf25"})

    # --- smile slope / curvature from smiles.csv (streamed quadratic fit) ------
    parts = []
    for ch in pd.read_csv(research / "smiles.csv",
                          usecols=["date", "time_to_expiry", "moneyness", "implied_volatility"],
                          chunksize=CHUNK):
        band = ch[ch["time_to_expiry"].between(*BAND) & ch["moneyness"].abs().le(MW)]
        if len(band):
            parts.append(band)
    band = pd.concat(parts, ignore_index=True)
    rows = []
    for d, g in band.groupby("date"):
        t_star = g["time_to_expiry"].iloc[(g["time_to_expiry"] - TARGET).abs().values.argmin()]
        gg = g[np.isclose(g["time_to_expiry"], t_star)].dropna(subset=["moneyness", "implied_volatility"])
        if len(gg) < 5:
            continue
        c, b, a = np.polyfit(gg["moneyness"].values, gg["implied_volatility"].values, 2)
        rows.append({"date": d, "slope": b, "curvature": 2.0 * c})
    fit = pd.DataFrame(rows)

    feat = sk.merge(fit, on="date", how="inner").sort_values("date")
    header = not (master / "m2_features.csv").exists()
    feat.to_csv(master / "m2_features.csv", mode="a", header=header, index=False)
    print(f"  {research.name}: {len(feat)} feature-days")


if __name__ == "__main__":
    main()
