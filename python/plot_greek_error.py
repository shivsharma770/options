"""Plot per-Greek validation errors.

Consumes the CSV emitted by ``GreeksValidationStudy`` and renders a
5-panel figure (delta, gamma, theta, vega, rho): each panel is a
histogram of the signed error (``computed - provider``) on a
log-scaled y-axis so the tails are visible.

Usage
-----

    python -m python.plot_greek_error \
        --input data/generated/research/greeks_validation.csv \
        --output data/generated/plots/greek_error.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd

_GREEKS = ("delta", "gamma", "theta", "vega", "rho")


def _load(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date", "expiration"])


def _plot(df: pd.DataFrame, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, greek in zip(axes, _GREEKS):
        col = f"{greek}_error"
        # Some rows omit a Greek when the vendor did not publish
        # that Greek. Dropna drops those from the histogram; they
        # still count toward the study's global row count.
        err = df[col].dropna()
        if err.empty:
            ax.set_title(f"{greek} (no data)")
            ax.set_xlabel(f"{greek} error")
            continue
        ax.hist(err, bins=60, log=True, color="steelblue", edgecolor="black")
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel(f"{greek} error (computed - provider)")
        ax.set_ylabel("count (log)")
        ax.set_title(f"{greek}: MAE = {err.abs().mean():.3e}")
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=pathlib.Path)
    p.add_argument("--output", type=pathlib.Path,
                   default=pathlib.Path("data/generated/plots/greek_error.png"))
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
