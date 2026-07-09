"""Plot the pricing round-trip residual distribution.

Consumes the CSV emitted by ``PricingValidationStudy`` and produces:

  1. Histogram of ``|residual|`` on a log-y scale.
  2. Scatter of ``mid_price`` (x) vs ``residual`` (y) — reveals whether
     the residual scales with the option price (would indicate a
     relative-tolerance quirk in the solver rather than an absolute
     one).
  3. Per-date maximum absolute residual.

Usage
-----

    python -m python.plot_pricing_error \
        --input data/generated/research/pricing_validation.csv \
        --output data/generated/plots/pricing_error.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date", "expiration"])


def _plot(df: pd.DataFrame, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    conv = df.dropna(subset=["residual"])
    abs_res = conv["residual"].abs()

    # 1) Histogram of |residual|. Log-y and log-x so the tail is
    # readable — the residuals are heavy-tailed.
    axes[0].hist(abs_res[abs_res > 0], bins=np.logspace(-16, 0, 60),
                 log=True, color="steelblue", edgecolor="black")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("|residual|")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title("Residual magnitude")

    # 2) Residual vs price.
    axes[1].scatter(conv["mid_price"], conv["residual"], s=4, alpha=0.3,
                    color="steelblue")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("mid price")
    axes[1].set_ylabel("residual (repriced - mid)")
    axes[1].set_title("Residual vs mid price")

    # 3) Per-date worst absolute residual.
    daily_max = (conv.groupby("date")
                       .apply(lambda g: g["residual"].abs().max())
                       .sort_index())
    axes[2].plot(daily_max.index, daily_max.values, color="steelblue")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("date")
    axes[2].set_ylabel("worst |residual| (log)")
    axes[2].set_title("Per-date worst residual")
    axes[2].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=pathlib.Path)
    p.add_argument("--output", type=pathlib.Path,
                   default=pathlib.Path("data/generated/plots/pricing_error.png"))
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
