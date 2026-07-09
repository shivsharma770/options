"""Plot the runtime history of research studies.

Consumes a manually-maintained runtime log with columns

    study,timestamp,threads,processed_days,processed_contracts,runtime_seconds

(A convenient way to produce one is to append a line to
``data/generated/research/runtime_history.csv`` after each run; the
``ResearchReport`` returned by ``HistoricalResearchEngine::run``
carries every field needed.)  Renders:

  1. Runtime per run, coloured by ``study``.
  2. Contracts-per-second (throughput) over time.

Usage
-----

    python -m python.plot_runtime_history \
        --input data/generated/research/runtime_history.csv \
        --output data/generated/plots/runtime_history.png
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd


def _load(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # Compute throughput; guard against zero runtime.
    df["throughput"] = df["processed_contracts"] / df["runtime_seconds"].clip(lower=1e-9)
    return df


def _plot(df: pd.DataFrame, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for study, group in df.groupby("study"):
        axes[0].plot(group["timestamp"], group["runtime_seconds"],
                     marker="o", label=study)
        axes[1].plot(group["timestamp"], group["throughput"],
                     marker="o", label=study)

    axes[0].set_ylabel("runtime (s)")
    axes[0].set_xlabel("timestamp")
    axes[0].set_title("Runtime per run")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].set_ylabel("contracts / s")
    axes[1].set_xlabel("timestamp")
    axes[1].set_title("Throughput per run")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=pathlib.Path)
    p.add_argument("--output", type=pathlib.Path,
                   default=pathlib.Path("data/generated/plots/runtime_history.png"))
    args = p.parse_args()
    if not args.input.exists():
        print(f"{args.input}: file not found.", file=sys.stderr)
        return 1
    df = _load(args.input)
    if df.empty:
        print(f"{args.input}: no rows.", file=sys.stderr)
        return 1
    _plot(df, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
