"""
Plot absolute pricing error against the Black-Scholes reference for
every non-reference engine across the standard benchmark suite.

Consumes the CSV produced by `BenchmarkReport::write_csv` (typically
``benchmark_suite.csv``). Emits a grouped bar chart of
``absolute_error`` per (case, engine). Reference-engine rows have
zero error by definition and are dropped from the chart so the
log-scaled y-axis is well behaved.

Usage
-----

    python plot_accuracy.py path/to/benchmark_suite.csv
    python plot_accuracy.py benchmark_suite.csv --save accuracy.png
    python plot_accuracy.py benchmark_suite.csv --relative

`--relative`:
    Plot ``relative_error`` (dimensionless) instead of
    ``absolute_error`` (price units).

Dependencies: pandas, matplotlib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--relative", action="store_true",
                        help="Plot relative_error instead of absolute_error.")
    return parser.parse_args(argv)


def load_report(path: Path, relative: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    err_col = "relative_error" if relative else "absolute_error"
    required = {"case_name", "engine_name", err_col}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns in {path}: {sorted(missing)}")
    # Drop rows without an error column value (reference engine and
    # any case where a reference wasn't provided).
    df = df.dropna(subset=[err_col])
    df = df[df[err_col] > 0]
    return df


def plot_accuracy(df: pd.DataFrame, save_to: Path | None,
                  relative: bool) -> None:
    err_col = "relative_error" if relative else "absolute_error"
    pivot = df.pivot(index="case_name", columns="engine_name",
                     values=err_col)
    case_order = df["case_name"].drop_duplicates().tolist()
    pivot = pivot.reindex(case_order)

    n_engines = pivot.shape[1]
    x = np.arange(len(pivot.index))
    bar_width = 0.8 / max(n_engines, 1)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.index) * 0.9), 5))
    for i, engine in enumerate(pivot.columns):
        offsets = x + (i - (n_engines - 1) / 2) * bar_width
        ax.bar(offsets, pivot[engine].values, bar_width, label=engine)

    ax.set_yscale("log")
    label = "Relative error" if relative else "Absolute error (price units)"
    ax.set_ylabel(f"{label} vs Black-Scholes (log scale)")
    ax.set_xlabel("Benchmark case")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_title("Pricing error against Black-Scholes reference")
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="Engine")
    fig.tight_layout()

    if save_to is not None:
        fig.savefig(save_to, dpi=150)
        print(f"wrote {save_to}", file=sys.stderr)
    else:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_report(args.csv, args.relative)
    plot_accuracy(df, args.save, args.relative)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
