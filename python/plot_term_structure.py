"""
Plot the at-the-money implied-volatility term structure from a
`term_structure.csv` produced by the C++ volatility-analytics module.

Input CSV columns:
    expiration, time_to_expiry, atm_iv

Missing entries (spot outside the calibrated strike range at some
expiration) are `NaN` in the CSV and gaps in the plot.

Usage
-----

    python plot_term_structure.py path/to/term_structure.csv
    python plot_term_structure.py path/to/term_structure.csv --x-axis date
    python plot_term_structure.py path/to/term_structure.csv --save term.png

`--x-axis`:
    * `maturity` — numeric time-to-expiration in years (default).
    * `date`     — calendar expiration date.

Dependencies: pandas, matplotlib. Install with `pip install pandas matplotlib`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the ATM-IV term structure."
    )
    parser.add_argument("csv", type=Path, help="Path to term_structure.csv.")
    parser.add_argument(
        "--x-axis",
        choices=["maturity", "date"],
        default="maturity",
        help="X axis: numeric year-fraction (default) or calendar date.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Write the figure to this path instead of showing it.",
    )
    return parser.parse_args(argv)


def load_term(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["expiration"])
    required = {"expiration", "time_to_expiry", "atm_iv"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns in {path}: {sorted(missing)}")
    return df.sort_values("time_to_expiry").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_term(args.csv)

    fig, ax = plt.subplots(figsize=(9, 5))
    if args.x_axis == "maturity":
        ax.plot(df["time_to_expiry"], df["atm_iv"], marker="o", linewidth=1.4)
        ax.set_xlabel("Time to expiration (years)")
    else:
        ax.plot(df["expiration"], df["atm_iv"], marker="o", linewidth=1.4)
        ax.set_xlabel("Expiration date")
        fig.autofmt_xdate()

    ax.set_ylabel("ATM implied volatility (annualised)")
    ax.set_title("ATM implied-volatility term structure")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, dpi=150)
        print(f"wrote {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
