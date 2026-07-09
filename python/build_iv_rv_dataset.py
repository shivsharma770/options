#!/usr/bin/env python3
"""
build_iv_rv_dataset.py  --  ETL step for Research Milestone 1.

Extracts the two compact series the IV-vs-RV study needs from one run of
`HistoricalCalibrationStudy` and appends them to master files, so the bulky
per-run CSVs (calibration/smiles/surface, multi-GB over the full archive) can
be deleted immediately afterward:

  * daily underlying price   -- recovered exactly from smiles.csv as
                                S = K * exp(-ln(K/S)); every row of a given day
                                yields the same spot, so a per-day mean is exact.
  * ATM term structure       -- copied straight from term_structure.csv
                                (date, expiration, time_to_expiry, atm_iv).

Usage:
    build_iv_rv_dataset.py <research_dir> <master_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 1_000_000


def append_csv(df: pd.DataFrame, path: Path) -> None:
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def main() -> None:
    research = Path(sys.argv[1])
    master = Path(sys.argv[2])
    master.mkdir(parents=True, exist_ok=True)

    # --- daily spot from smiles.csv (streamed) --------------------------------
    ssum: dict[str, float] = {}
    scnt: dict[str, int] = {}
    for ch in pd.read_csv(research / "smiles.csv",
                          usecols=["date", "strike", "moneyness"], chunksize=CHUNK):
        ch["s"] = ch["strike"] * np.exp(-ch["moneyness"])
        g = ch.groupby("date")["s"].agg(["sum", "count"])
        for d, row in g.iterrows():
            ssum[d] = ssum.get(d, 0.0) + row["sum"]
            scnt[d] = scnt.get(d, 0) + int(row["count"])
    spot = pd.DataFrame(
        {"date": list(ssum), "spot": [ssum[d] / scnt[d] for d in ssum]}
    ).sort_values("date")
    append_csv(spot, master / "daily_underlying.csv")

    # --- ATM term structure (small; copy through) -----------------------------
    ts = pd.read_csv(research / "term_structure.csv")
    append_csv(ts, master / "atm_term_structure.csv")

    print(f"  extracted {len(spot)} days, {len(ts)} term-structure rows "
          f"from {research.name}")


if __name__ == "__main__":
    main()
