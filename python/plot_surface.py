"""
Plot the implied-volatility surface from a `surface.csv` produced by the
C++ volatility-analytics module.

Input CSV columns:
    expiration, time_to_expiry, strike, implied_volatility

The CSV is in long format; this script pivots it into a `strike x
maturity` grid and renders it as a filled contour plot (default) or as
a 3D wireframe.

Usage
-----

    python plot_surface.py path/to/surface.csv
    python plot_surface.py path/to/surface.csv --kind wireframe
    python plot_surface.py path/to/surface.csv --save surface.png

Missing cells (NaN) are left transparent in the contour plot; the
wireframe view interpolates within matplotlib for display purposes only
— the underlying data still has holes.

Dependencies: pandas, matplotlib. `mpl_toolkits.mplot3d` ships with
matplotlib. Install with `pip install pandas matplotlib`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the implied-volatility surface."
    )
    parser.add_argument("csv", type=Path, help="Path to surface.csv.")
    parser.add_argument(
        "--kind",
        choices=["contour", "wireframe"],
        default="contour",
        help="Rendering style: filled contour (default) or 3D wireframe.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Write the figure to this path instead of showing it.",
    )
    return parser.parse_args(argv)


def load_surface(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"expiration", "time_to_expiry", "strike", "implied_volatility"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns in {path}: {sorted(missing)}")
    return df


def pivot_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pivot the long-format frame into a (maturity, strike) grid.

    Returns
    -------
    T, K, iv : np.ndarray
        1D maturity axis, 1D strike axis, and 2D iv grid shape (len(T), len(K)).
    """
    # `time_to_expiry` is the numeric maturity — index by it so the axis
    # is sorted numerically rather than lexicographically.
    grid = df.pivot_table(
        index="time_to_expiry",
        columns="strike",
        values="implied_volatility",
        aggfunc="first",  # long format is already unique per (T, K)
    ).sort_index()
    T = grid.index.to_numpy()
    K = grid.columns.to_numpy()
    return T, K, grid.to_numpy()


def plot_contour(T: np.ndarray, K: np.ndarray, iv: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    # `contourf` handles NaN by leaving those cells uncoloured.
    KK, TT = np.meshgrid(K, T)
    cs = ax.contourf(KK, TT, iv, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="Implied volatility")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to expiration (years)")
    ax.set_title("Implied-volatility surface")
    fig.tight_layout()
    return fig


def plot_wireframe(T: np.ndarray, K: np.ndarray, iv: np.ndarray) -> plt.Figure:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    KK, TT = np.meshgrid(K, T)

    # Wireframe treats NaN as gaps in the mesh; matplotlib handles this
    # gracefully for the default renderer.
    ax.plot_wireframe(KK, TT, iv, rstride=1, cstride=1, linewidth=0.6)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to expiration (years)")
    ax.set_zlabel("Implied volatility")
    ax.set_title("Implied-volatility surface")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_surface(args.csv)
    T, K, iv = pivot_grid(df)

    if T.size == 0 or K.size == 0:
        raise SystemExit("surface.csv is empty; nothing to plot.")

    fig = plot_contour(T, K, iv) if args.kind == "contour" else plot_wireframe(T, K, iv)

    if args.save is not None:
        fig.savefig(args.save, dpi=150)
        print(f"wrote {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
