#!/usr/bin/env python3
"""
build_m3_calib.py -- ETL for Research Milestone 3 (daily calibration RMSE).

Streams one run's calibration.csv and appends the per-day calibration quality
to a master file, so the multi-hundred-MB calibration.csv can be deleted
immediately. The RMSE is the same |provider_iv - computed_iv| metric the
SPY-2020 study reports; here it becomes a candidate *return* predictor (a proxy
for surface dislocation / stress).

Usage:  build_m3_calib.py <research_dir> <master_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 1_000_000


def main():
    research = Path(sys.argv[1])
    master = Path(sys.argv[2])
    master.mkdir(parents=True, exist_ok=True)

    ssq: dict[str, float] = {}
    sab: dict[str, float] = {}
    cnt: dict[str, int] = {}
    for ch in pd.read_csv(research / "calibration.csv",
                          usecols=["date", "absolute_error"],
                          dtype={"absolute_error": "float64"}, chunksize=CHUNK):
        ch = ch.dropna(subset=["absolute_error"])
        ch["sq"] = ch["absolute_error"] ** 2
        g = ch.groupby("date").agg(sq=("sq", "sum"), ab=("absolute_error", "sum"),
                                    n=("absolute_error", "count"))
        for d, r in g.iterrows():
            ssq[d] = ssq.get(d, 0.0) + r["sq"]
            sab[d] = sab.get(d, 0.0) + r["ab"]
            cnt[d] = cnt.get(d, 0) + int(r["n"])

    rows = [{"date": d, "calib_rmse": np.sqrt(ssq[d] / cnt[d]),
             "calib_mae": sab[d] / cnt[d], "calib_n": cnt[d]}
            for d in ssq if cnt[d] > 0]
    out = pd.DataFrame(rows).sort_values("date")
    path = master / "m3_calib.csv"
    out.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"  {research.name}: {len(out)} days")


if __name__ == "__main__":
    main()
