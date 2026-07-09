"""Plot the IV validation error distribution.

Consumes the CSV emitted by ``IVValidationStudy`` (default filename:
``iv_validation.csv`` under ``data/generated/research/``).  Produces
a histogram of ``absolute_error``, a scatter of ``provider_iv`` vs
``computed_iv``, and a per-date summary of mean absolute error.

Usage
-----

    python -m python.plot_iv_error \
        --input data/generated/research/iv_validation.csv \
        --output data/generated/plots/iv_error.png

The script never mutates the CSV or the plots directory: the input is
read-only and the output PNG is written wholesale.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd


def _load(path: pathlib.Path) -> pd.DataFrame:
    """Read the IV validation CSV, coercing date columns and dropping
    rows without a computed IV (non-convergent solves)."""
    df = pd.read_csv(path, parse_dates=["date", "expiration"])
    # `computed_iv` and `absolute_error` are ``NaN`` when the solver
    # did not converge.  Drop those for the summary plots but keep
    # the convergence-rate line honest (we still count them).
    return df


def _plot(df: pd.DataFrame, output: pathlib.Path) -> None:
    """Render three subplots: error histogram, scatter, per-date MAE."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    converged = df.dropna(subset=["computed_iv", "absolute_error"])

    # 1) |computed_iv - provider_iv| histogram, log-y so heavy tails
    # do not swallow the mode.
    axes[0].hist(converged["absolute_error"], bins=60, log=True,
                 color="steelblue", edgecolor="black")
    axes[0].set_xlabel("|computed IV - provider IV|")
    axes[0].set_ylabel("count (log scale)")
    axes[0].set_title("IV error distribution")

    # 2) Scatter of provider vs computed IV — the "y = x" diagonal is
    # the ideal.
    axes[1].scatter(converged["provider_iv"], converged["computed_iv"],
                    s=4, alpha=0.3, color="steelblue")
    lo = min(converged["provider_iv"].min(), converged["computed_iv"].min())
    hi = max(converged["provider_iv"].max(), converged["computed_iv"].max())
    axes[1].plot([lo, hi], [lo, hi], color="black", linewidth=1)
    axes[1].set_xlabel("provider IV")
    axes[1].set_ylabel("computed IV")
    axes[1].set_title("Computed vs provider IV")

    # 3) Mean absolute error per trading day — flags regime shifts.
    daily_mae = (converged
                 .groupby("date")["absolute_error"]
                 .mean()
                 .sort_index())
    axes[2].plot(daily_mae.index, daily_mae.values, color="steelblue")
    axes[2].set_xlabel("date")
    axes[2].set_ylabel("daily mean |IV error|")
    axes[2].set_title("Per-date MAE")
    axes[2].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=pathlib.Path,
                   help="CSV emitted by IVValidationStudy")
    p.add_argument("--output", type=pathlib.Path,
                   default=pathlib.Path("data/generated/plots/iv_error.png"),
                   help="destination PNG")
    args = p.parse_args()

    df = _load(args.input)
    if df.empty:
        print(f"{args.input}: no rows.", file=sys.stderr)
        return 1

    _plot(df, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
