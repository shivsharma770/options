"""
Plot convergence curves for the Binomial and Monte Carlo engines from
the CSVs produced by
``BenchmarkRunner::run_binomial_convergence`` and
``run_monte_carlo_convergence``.

The two convergence sweeps share the same report schema, so this
script accepts either — the input file is auto-detected via the engine
names present in the CSV. Both file types may be passed at once for a
combined figure.

For each engine sweep the script produces a log-log plot of
``absolute_error`` vs ``iterations`` and — for Monte Carlo —
``standard_error`` vs ``iterations``. Reference dashed lines at
``1/N`` and ``1/sqrt(N)`` slopes are overlaid to make the theoretical
rate immediately readable.

Usage
-----

    python plot_convergence.py benchmark_binomial_convergence.csv
    python plot_convergence.py benchmark_binomial_convergence.csv \\
        benchmark_monte_carlo_convergence.csv --save convergence.png

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
    parser.add_argument("csvs", nargs="+", type=Path,
                        help="Convergence CSVs (Binomial and/or MC).")
    parser.add_argument("--save", type=Path, default=None)
    return parser.parse_args(argv)


def classify(df: pd.DataFrame) -> str:
    """Return `"binomial"`, `"monte_carlo"`, or `"unknown"` based on
    the `engine_name` column."""
    names = df["engine_name"].astype(str).str.lower()
    if names.str.contains("monte").any() or names.str.contains("mc").any():
        return "monte_carlo"
    if names.str.contains("binomial").any():
        return "binomial"
    return "unknown"


def plot_binomial(ax, df: pd.DataFrame) -> None:
    df = df.dropna(subset=["absolute_error"])
    df = df[df["absolute_error"] > 0]
    x = df["iterations"].to_numpy()
    y = df["absolute_error"].to_numpy()
    order = np.argsort(x)
    x, y = x[order], y[order]

    ax.loglog(x, y, marker="o", label="Binomial abs error")

    # Reference 1/N slope anchored at the first point.
    if len(x) >= 2:
        c = y[0] * x[0]
        ax.loglog(x, c / x, linestyle="--", alpha=0.5,
                  label="reference slope 1/N")


def plot_monte_carlo(ax, df: pd.DataFrame) -> None:
    df = df.dropna(subset=["absolute_error"])
    df = df[df["absolute_error"] > 0]
    x = df["iterations"].to_numpy()
    y = df["absolute_error"].to_numpy()
    order = np.argsort(x)
    x, y = x[order], y[order]

    ax.loglog(x, y, marker="o", label="MC abs error")

    if "standard_error" in df.columns and df["standard_error"].notna().any():
        se = df["standard_error"].to_numpy()[order]
        ax.loglog(x, se, marker="s", label="MC standard error")

    # Reference 1/sqrt(N) slope anchored at the first point.
    if len(x) >= 2:
        c = y[0] * np.sqrt(x[0])
        ax.loglog(x, c / np.sqrt(x), linestyle="--", alpha=0.5,
                  label="reference slope 1/√N")


def plot_convergence(dfs: list[pd.DataFrame], save_to: Path | None) -> None:
    kinds = [classify(df) for df in dfs]
    n_plots = len([k for k in kinds if k != "unknown"])
    if n_plots == 0:
        raise SystemExit(
            "no recognisable Binomial or Monte Carlo rows in the inputs")

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5),
                             squeeze=False)
    idx = 0
    for kind, df in zip(kinds, dfs):
        if kind == "binomial":
            ax = axes[0, idx]
            plot_binomial(ax, df)
            ax.set_title("Binomial (CRR) convergence to Black-Scholes")
            ax.set_xlabel("Steps N (log scale)")
            ax.set_ylabel("|price − BS price| (log scale)")
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            ax.legend()
            idx += 1
        elif kind == "monte_carlo":
            ax = axes[0, idx]
            plot_monte_carlo(ax, df)
            ax.set_title("Monte Carlo convergence to Black-Scholes")
            ax.set_xlabel("Paths P (log scale)")
            ax.set_ylabel("Error / standard error (log scale)")
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            ax.legend()
            idx += 1

    fig.tight_layout()

    if save_to is not None:
        fig.savefig(save_to, dpi=150)
        print(f"wrote {save_to}", file=sys.stderr)
    else:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dfs = [pd.read_csv(p) for p in args.csvs]
    plot_convergence(dfs, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
