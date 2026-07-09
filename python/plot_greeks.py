"""
Plot analytical, tree-based, and Monte Carlo Greeks side by side across
the standard benchmark suite.

Consumes the main-suite CSV (typically ``benchmark_suite.csv``).
Requires every engine's row to contain ``delta``, ``gamma``, ``vega``,
``theta``, and ``rho`` columns — which they always will, since the C++
side zero-fills unset Greeks. Rows where every Greek is exactly zero
are ignored so an engine that skipped Greek computation doesn't
contaminate the plot.

The output is a 5-panel figure (one Greek per axis) with a grouped bar
per (case, engine).

Usage
-----

    python plot_greeks.py path/to/benchmark_suite.csv
    python plot_greeks.py benchmark_suite.csv --save greeks.png

Dependencies: pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GREEKS = ("delta", "gamma", "vega", "theta", "rho")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--save", type=Path, default=None)
    return parser.parse_args(argv)


def load_report(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"case_name", "engine_name", *GREEKS}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns in {path}: {sorted(missing)}")

    # Drop rows where every Greek is exactly zero (engine either did
    # not compute Greeks or is a placeholder). Guard against genuine
    # zeros — deep-OTM options can have all-tiny Greeks — with a small
    # magnitude threshold.
    mask = df[list(GREEKS)].abs().sum(axis=1) > 1e-12
    return df[mask]


def plot_greeks(df: pd.DataFrame, save_to: Path | None) -> None:
    case_order = df["case_name"].drop_duplicates().tolist()
    engine_order = df["engine_name"].drop_duplicates().tolist()
    x = np.arange(len(case_order))
    bar_width = 0.8 / max(len(engine_order), 1)

    fig, axes = plt.subplots(5, 1, figsize=(max(8, len(case_order) * 0.9), 14),
                             sharex=True)
    for ax, greek in zip(axes, GREEKS):
        pivot = df.pivot(index="case_name", columns="engine_name",
                         values=greek).reindex(case_order)
        for i, engine in enumerate(engine_order):
            if engine not in pivot.columns:
                continue
            offsets = x + (i - (len(engine_order) - 1) / 2) * bar_width
            ax.bar(offsets, pivot[engine].values, bar_width, label=engine)
        ax.set_ylabel(greek.capitalize())
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    axes[0].set_title("Greeks across engines and benchmark cases")
    axes[0].legend(title="Engine", loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(case_order, rotation=30, ha="right")
    axes[-1].set_xlabel("Benchmark case")
    fig.tight_layout()

    if save_to is not None:
        fig.savefig(save_to, dpi=150)
        print(f"wrote {save_to}", file=sys.stderr)
    else:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_report(args.csv)
    plot_greeks(df, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
