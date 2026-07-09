"""
Plot per-engine median runtime across the standard benchmark suite.

Consumes the CSV produced by `BenchmarkReport::write_csv` for the main
suite (typically ``benchmark_suite.csv``). Emits a grouped bar chart:
one bar per (case, engine) pair, cases on the x-axis, engines side by
side. y-axis is microseconds on a log scale so Black-Scholes
(≈1 µs) and 200 000-path Monte Carlo (≈100 000 µs) can share the
same figure without one collapsing to a hairline.

Usage
-----

    python plot_runtime.py path/to/benchmark_suite.csv
    python plot_runtime.py benchmark_suite.csv --save runtime.png

Dependencies: pandas, matplotlib. Install with
``pip install pandas matplotlib``.
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
    parser.add_argument("csv", type=Path,
                        help="Path to the benchmark suite CSV.")
    parser.add_argument("--save", type=Path, default=None,
                        help="If set, write the figure to this path "
                             "instead of showing it interactively.")
    return parser.parse_args(argv)


def load_report(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"case_name", "engine_name", "runtime_us"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns in {path}: {sorted(missing)}")
    return df


def plot_runtime(df: pd.DataFrame, save_to: Path | None) -> None:
    # Pivot to a (case × engine) matrix of median runtimes. Engines
    # not present for a given case become NaN and are skipped by
    # matplotlib's bar plotter.
    pivot = df.pivot(index="case_name", columns="engine_name",
                     values="runtime_us")
    # Preserve the case order from the input so plots line up with the
    # docs table. `pd.Categorical` is the idiomatic way to enforce this.
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
    ax.set_ylabel("Median runtime per price() call (µs, log scale)")
    ax.set_xlabel("Benchmark case")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_title("Runtime across pricing engines and benchmark cases")
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
    df = load_report(args.csv)
    plot_runtime(df, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
