"""
Plot volatility smiles from a `smiles.csv` produced by the C++
volatility-analytics module.

Input CSV columns (see docs/VOLATILITY_ANALYTICS.md):
    expiration, time_to_expiry, strike, option_type,
    moneyness_convention, moneyness, implied_volatility

One line is drawn per expiration. Both calls and puts are shown on the
same axis (they should agree at the ATM strike; disagreement across
strikes is the smile itself).

Usage
-----

    python plot_smile.py path/to/smiles.csv
    python plot_smile.py path/to/smiles.csv --x-axis moneyness
    python plot_smile.py path/to/smiles.csv --otm-only --save smiles.png

`--x-axis`:
    * `strike`     — raw strike price (default).
    * `moneyness`  — the moneyness column emitted by the C++ side.
                     The header says which convention was used.
`--otm-only`:
    Drop calls with K < spot and puts with K > spot before plotting.
    Requires a `spot` value to be inferred from the ATM strike; passing
    it via `--spot` overrides the inference.

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
        description="Plot implied-volatility smiles from a smiles.csv."
    )
    parser.add_argument("csv", type=Path, help="Path to smiles.csv.")
    parser.add_argument(
        "--x-axis",
        choices=["strike", "moneyness"],
        default="strike",
        help="X axis: raw strike or the moneyness column.",
    )
    parser.add_argument(
        "--otm-only",
        action="store_true",
        help="Show only OTM options (calls: K >= spot, puts: K <= spot).",
    )
    parser.add_argument(
        "--spot",
        type=float,
        default=None,
        help="Spot price. Required for --otm-only if not inferrable.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Write the figure to this path instead of showing it.",
    )
    return parser.parse_args(argv)


def load_smiles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "expiration",
        "time_to_expiry",
        "strike",
        "option_type",
        "moneyness",
        "implied_volatility",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns in {path}: {sorted(missing)}")
    return df


def infer_spot(df: pd.DataFrame) -> float:
    # If moneyness is in K/S convention we can back it out; if ln(K/S) or
    # ln(K/F), K = S at moneyness=0 (LogSimple) or moneyness=-(r-q)T
    # (LogForward). Simplest robust choice: pick the strike whose moneyness
    # is closest to zero across all expirations; that must be at-or-near
    # spot regardless of convention.
    idx = df["moneyness"].abs().idxmin()
    return float(df.loc[idx, "strike"])


def filter_otm(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    call_mask = (df["option_type"] == "call") & (df["strike"] >= spot)
    put_mask  = (df["option_type"] == "put")  & (df["strike"] <= spot)
    return df[call_mask | put_mask].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_smiles(args.csv)

    if args.otm_only:
        spot = args.spot if args.spot is not None else infer_spot(df)
        df = filter_otm(df, spot)

    fig, ax = plt.subplots(figsize=(9, 6))

    x_col = "strike" if args.x_axis == "strike" else "moneyness"
    for exp, group in df.groupby("expiration", sort=True):
        group = group.sort_values(x_col)
        ax.plot(
            group[x_col],
            group["implied_volatility"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=str(exp),
        )

    conv = df["moneyness_convention"].iloc[0] if "moneyness_convention" in df.columns else ""
    xlabel = "Strike" if args.x_axis == "strike" else f"Moneyness ({conv})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Implied volatility (annualised)")
    ax.set_title("Implied-volatility smile by expiration")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(title="Expiration", loc="best", fontsize=9)
    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, dpi=150)
        print(f"wrote {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
