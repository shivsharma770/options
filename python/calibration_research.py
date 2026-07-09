#!/usr/bin/env python3
"""
calibration_research.py
=======================

Analysis of the ``HistoricalCalibrationStudy`` outputs for a single
underlying over one calendar year. Consumes the five long-format CSVs the
study writes to ``data/generated/research/`` and produces:

  * publication-quality figures under ``data/generated/research/figures/``
  * a machine-readable summary (``summary_stats.json``) plus a printed table

It does not touch the C++ pricing or calibration code in any way -- it is a
pure downstream consumer of the calibration / smile / term-structure /
surface / skew CSVs.

Metrics
-------
  * average calibration RMSE       (computed-vs-provider implied vol)
  * IV smile curvature             (d^2 IV / d m^2 at ~30 DTE, and 25d fly)
  * 25-delta risk reversal         (call25 - put25, ~30 DTE)
  * ATM term structure             (ATM IV vs maturity, and its evolution)
  * skew evolution over time       (RR / butterfly time series)

Usage
-----
    python calibration_research.py [research_dir]
    # default research_dir = data/generated/research
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")  # headless
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
TARGET_DTE_YEARS = 30.0 / 365.0          # "1-month" reference maturity
DTE_BAND = (18.0 / 365.0, 45.0 / 365.0)  # band used for smile-curvature fits
MONEYNESS_CLIP = 0.30                     # |ln(K/S)| window for smile work
CHUNK = 1_000_000                         # rows per chunk for the big CSVs

# Maturity buckets (calendar days) for the ATM term-structure heatmap.
TS_BUCKETS_DAYS = [7, 14, 30, 60, 90, 120, 180, 270, 365]

# COVID-19 regime split (used only for reporting, never for calibration).
REGIMES = {
    "pre-crash (Jan 2 - Feb 19)": ("2020-01-01", "2020-02-19"),
    "crash    (Feb 20 - Apr 30)": ("2020-02-20", "2020-04-30"),
    "recovery (May 1 - Dec 31)": ("2020-05-01", "2020-12-31"),
}

# Publication styling ------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
})
CmapSeq = "viridis"
CmapDiv = "coolwarm"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _nearest_row(group: pd.DataFrame, target: float, col: str = "time_to_expiry"):
    """Row of `group` whose `col` is closest to `target`."""
    idx = (group[col] - target).abs().values.argmin()
    return group.iloc[idx]


def load_small(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


# --------------------------------------------------------------------------
# 1. Calibration quality (RMSE) -- streamed over the big calibration.csv
# --------------------------------------------------------------------------
def calibration_quality(path: Path):
    """Per-day and overall stats of |provider_iv - computed_iv| and the
    price-space solver residual, streamed so we never hold the whole file."""
    per_day = {}          # date -> [sum_sq_err, n_err, sum_abs_err, sum_resid, n_resid]
    for chunk in pd.read_csv(
        path,
        usecols=["date", "absolute_error", "solver_residual"],
        dtype={"absolute_error": "float64", "solver_residual": "float64"},
        chunksize=CHUNK,
    ):
        chunk["sq"] = chunk["absolute_error"] ** 2
        g = chunk.groupby("date")
        agg = g.agg(
            sum_sq=("sq", "sum"),
            n_err=("absolute_error", "count"),
            sum_abs=("absolute_error", "sum"),
            sum_resid=("solver_residual", "sum"),
            n_resid=("solver_residual", "count"),
        )
        for d, row in agg.iterrows():
            acc = per_day.setdefault(d, [0.0, 0, 0.0, 0.0, 0])
            acc[0] += row.sum_sq
            acc[1] += int(row.n_err)
            acc[2] += row.sum_abs
            acc[3] += row.sum_resid
            acc[4] += int(row.n_resid)

    rows = []
    for d, (ssq, n, sab, sres, nres) in per_day.items():
        if n == 0:
            continue
        rows.append({
            "date": pd.Timestamp(d),
            "rmse": np.sqrt(ssq / n),
            "mae": sab / n,
            "n": n,
            "mean_resid": (sres / nres) if nres else np.nan,
        })
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    overall_rmse = float(np.sqrt((daily.rmse ** 2 * daily.n).sum() / daily.n.sum()))
    return daily, overall_rmse


# --------------------------------------------------------------------------
# 2. Term structure: ~30d ATM series + date x maturity ATM grid
# --------------------------------------------------------------------------
def term_structure_analysis(path: Path):
    ts = load_small(path).dropna(subset=["atm_iv"])
    ts = ts[ts["time_to_expiry"] > 0].copy()

    # ~30-day ATM series: nearest expiry to target per date (idxmin, no apply).
    ts["_dist"] = (ts["time_to_expiry"] - TARGET_DTE_YEARS).abs()
    idx = ts.groupby("date")["_dist"].idxmin()
    atm30 = (ts.loc[idx, ["date", "atm_iv"]]
             .rename(columns={"atm_iv": "atm_iv_30d"})
             .sort_values("date").reset_index(drop=True))
    ts = ts.drop(columns="_dist")

    # date x bucket grid (nearest expiry to each bucket, within 40% tolerance).
    dates = np.sort(ts["date"].unique())
    grid = np.full((len(TS_BUCKETS_DAYS), len(dates)), np.nan)
    date_to_col = {d: i for i, d in enumerate(dates)}
    for d, g in ts.groupby("date"):
        col = date_to_col[d]
        tvals = g["time_to_expiry"].values * 365.0
        ivals = g["atm_iv"].values
        for r, bkt in enumerate(TS_BUCKETS_DAYS):
            j = np.abs(tvals - bkt).argmin()
            if abs(tvals[j] - bkt) <= 0.4 * bkt:
                grid[r, col] = ivals[j]
    return ts, atm30, dates, grid


# --------------------------------------------------------------------------
# 3. Smiles: filtered band for curvature time series + snapshot smiles
# --------------------------------------------------------------------------
def smile_analysis(path: Path, snapshot_dates):
    keep = []
    snap = {pd.Timestamp(d).strftime("%Y-%m-%d"): [] for d in snapshot_dates}
    for chunk in pd.read_csv(
        path,
        usecols=["date", "time_to_expiry", "strike", "option_type", "moneyness",
                 "implied_volatility"],
        chunksize=CHUNK,
    ):
        # Out-of-the-money construction: one IV per strike (put below spot,
        # call above). Keeping both types plots two IVs per strike, and under
        # the r=q=0 pricing assumption call/put IV diverge on deeper strikes
        # (put-call parity no longer holds), producing a sawtooth. OTM is the
        # standard, clean smile.
        otm = chunk[((chunk["option_type"] == "Put") & (chunk["moneyness"] < 0)) |
                    ((chunk["option_type"] == "Call") & (chunk["moneyness"] >= 0))]
        band = otm[
            otm["time_to_expiry"].between(*DTE_BAND)
            & otm["moneyness"].abs().le(MONEYNESS_CLIP)
        ]
        if len(band):
            keep.append(band)
        for ds in snap:
            sub = otm[(otm["date"] == ds) & otm["moneyness"].abs().le(MONEYNESS_CLIP)]
            if len(sub):
                snap[ds].append(sub)

    band = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    band["date"] = pd.to_datetime(band["date"])

    # Curvature time series: per date, take the single expiry nearest 30 DTE,
    # fit IV = a + b*m + c*m^2, report curvature = 2c = d^2 IV / d m^2.
    curv_rows = []
    for d, g in band.groupby("date"):
        # pick nearest-to-target maturity present that day
        t_star = g["time_to_expiry"].iloc[(g["time_to_expiry"] - TARGET_DTE_YEARS).abs().values.argmin()]
        gg = g[np.isclose(g["time_to_expiry"], t_star)]
        gg = gg.dropna(subset=["moneyness", "implied_volatility"])
        if len(gg) < 5:
            continue
        c, b, a = np.polyfit(gg["moneyness"].values, gg["implied_volatility"].values, 2)
        curv_rows.append({"date": d, "curvature": 2.0 * c, "atm_fit": a, "slope": b})
    curvature = pd.DataFrame(curv_rows).sort_values("date").reset_index(drop=True)

    # Snapshot smiles: for each requested date, the single ~30-DTE expiry.
    # Also recover the spot (S = K * exp(-ln(K/S))) so the surface figure can
    # work in moneyness space and stay comparable across dates.
    snap_smiles = {}
    spots = {}
    for ds, parts in snap.items():
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        spots[ds] = float(np.median(df["strike"] * np.exp(-df["moneyness"])))
        t_star = df["time_to_expiry"].iloc[(df["time_to_expiry"] - TARGET_DTE_YEARS).abs().values.argmin()]
        sm = df[np.isclose(df["time_to_expiry"], t_star)].sort_values("moneyness")
        snap_smiles[ds] = (sm, float(t_star * 365.0))
    return curvature, snap_smiles, spots


# --------------------------------------------------------------------------
# 4. Surface snapshot(s): (maturity x moneyness) grid for chosen date(s)
# --------------------------------------------------------------------------
def surface_snapshots(path: Path, dates):
    wanted = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates}
    parts = {d: [] for d in wanted}
    for chunk in pd.read_csv(
        path,
        usecols=["date", "time_to_expiry", "strike", "implied_volatility"],
        chunksize=CHUNK,
    ):
        sub = chunk[chunk["date"].isin(wanted)]
        for d in wanted:
            s = sub[sub["date"] == d]
            if len(s):
                parts[d].append(s)
    out = {}
    for d, ps in parts.items():
        if ps:
            out[d] = pd.concat(ps, ignore_index=True)
    return out


# --------------------------------------------------------------------------
# 5. Skew evolution: ~30d RR / butterfly time series
# --------------------------------------------------------------------------
def skew_analysis(path: Path):
    sk = load_small(path)
    sk = sk[sk["time_to_expiry"] > 0].copy()
    sk["_dist"] = (sk["time_to_expiry"] - TARGET_DTE_YEARS).abs()
    idx = sk.groupby("date")["_dist"].idxmin()
    cols = ["date", "atm_iv", "call_25delta_iv", "put_25delta_iv", "risk_reversal", "butterfly"]
    return sk.loc[idx, cols].sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_calibration(daily, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(daily["date"], daily["rmse"], color="#1f77b4", lw=1.4, label="daily RMSE")
    ax[0].plot(daily["date"], daily["mae"], color="#ff7f0e", lw=1.0, alpha=0.8, label="daily MAE")
    ax[0].set_title("(a) Calibration error vs provider IV")
    ax[0].set_ylabel("implied-vol error (abs.)")
    ax[0].legend(loc="upper right")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    ax[1].hist(daily["rmse"], bins=40, color="#4c72b0", edgecolor="white")
    ax[1].axvline(daily["rmse"].mean(), color="crimson", ls="--", lw=1.4,
                  label=f"mean = {daily['rmse'].mean():.4f}")
    ax[1].set_title("(b) Distribution of daily RMSE")
    ax[1].set_xlabel("daily RMSE")
    ax[1].set_ylabel("trading days")
    ax[1].legend()
    fig.suptitle("Calibration quality — SPY 2020", fontweight="bold")
    fig.savefig(out / "fig1_calibration_quality.png")
    plt.close(fig)


def fig_term_structure(ts, atm30, dates, grid, snapshot_dates, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(snapshot_dates)))
    for d, c in zip(snapshot_dates, colors):
        ds = pd.Timestamp(d)
        g = ts[ts["date"] == ds].sort_values("time_to_expiry")
        g = g[g["time_to_expiry"] <= 1.05]
        ax[0].plot(g["time_to_expiry"] * 365, g["atm_iv"], "-o", ms=3, color=c,
                   label=ds.strftime("%Y-%m-%d"))
    ax[0].set_title("(a) ATM term structure on representative dates")
    ax[0].set_xlabel("days to expiry")
    ax[0].set_ylabel("ATM implied vol")
    ax[0].legend(title="date")

    im = ax[1].imshow(
        grid, aspect="auto", origin="lower", cmap=CmapSeq,
        extent=[mdates.date2num(dates[0]), mdates.date2num(dates[-1]),
                0, len(TS_BUCKETS_DAYS)],
        interpolation="nearest",
    )
    ax[1].set_yticks(np.arange(len(TS_BUCKETS_DAYS)) + 0.5)
    ax[1].set_yticklabels([f"{b}d" for b in TS_BUCKETS_DAYS])
    ax[1].xaxis_date()
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[1].set_title("(b) ATM implied-vol surface over time")
    ax[1].set_ylabel("maturity bucket")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], label="ATM implied vol", fraction=0.046, pad=0.02)
    fig.suptitle("ATM term structure — SPY 2020", fontweight="bold")
    fig.savefig(out / "fig2_term_structure.png")
    plt.close(fig)


def fig_smiles(snap_smiles, curvature, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(snap_smiles)))
    for (ds, (sm, dte)), c in zip(sorted(snap_smiles.items()), colors):
        ax[0].plot(sm["moneyness"], sm["implied_volatility"], "-o", ms=2.5,
                   color=c, label=f"{ds}  (~{dte:.0f}d)")
    ax[0].axvline(0.0, color="grey", lw=0.8, ls=":")
    ax[0].set_title("(a) Volatility smiles at ~1M maturity")
    ax[0].set_xlabel("log-moneyness  ln(K/S)")
    ax[0].set_ylabel("implied vol")
    ax[0].legend(title="date")

    ax[1].plot(curvature["date"], curvature["curvature"], color="#2c7fb8", lw=1.3)
    ax[1].set_title("(b) Smile curvature  d²σ/d(lnK/S)²  (~1M)")
    ax[1].set_xlabel("date")
    ax[1].set_ylabel("curvature")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.suptitle("Volatility smiles & curvature — SPY 2020", fontweight="bold")
    fig.savefig(out / "fig3_smiles.png")
    plt.close(fig)


def fig_skew(sk30, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(sk30["date"], sk30["risk_reversal"], color="#d62728", lw=1.3,
               label="25Δ risk reversal")
    ax[0].plot(sk30["date"], sk30["butterfly"], color="#2ca02c", lw=1.1,
               label="25Δ butterfly")
    ax[0].axhline(0, color="grey", lw=0.8, ls=":")
    ax[0].set_title("(a) 25-delta skew metrics (~1M)")
    ax[0].set_ylabel("vol points")
    ax[0].legend(loc="lower right")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    sc = ax[1].scatter(sk30["atm_iv"], sk30["risk_reversal"], c=mdates.date2num(sk30["date"]),
                       cmap=CmapSeq, s=14)
    ax[1].set_title("(b) Skew–vol relationship")
    ax[1].set_xlabel("ATM implied vol")
    ax[1].set_ylabel("25Δ risk reversal")
    cb = fig.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.02)
    cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.suptitle("Skew evolution — SPY 2020", fontweight="bold")
    fig.savefig(out / "fig4_skew_evolution.png")
    plt.close(fig)


def fig_surface(surfaces, spots, out):
    """Filled-contour IV surface in (log-moneyness, DTE) space. Working in
    moneyness keeps the two dates comparable (spot fell ~26% in the crash)
    and restricting to the liquid |ln(K/S)| <= 0.25 window drops the low-vega
    deep-wing contracts whose IV is ill-conditioned and would otherwise
    dominate the colour scale."""
    MW = 0.25
    items = [(ds, df) for ds, df in sorted(surfaces.items()) if ds in spots]
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 4.8), squeeze=False)

    prepared = []
    lo_all, hi_all = [], []
    for ds, df in items:
        d = df.dropna(subset=["implied_volatility"]).copy()
        d["m"] = np.log(d["strike"] / spots[ds])
        d["dte"] = d["time_to_expiry"] * 365.0
        d = d[(d["m"].abs() <= MW) & (d["dte"] <= 365) & (d["dte"] >= 3)]
        prepared.append((ds, d))
        lo_all.append(np.nanpercentile(d["implied_volatility"], 2))
        hi_all.append(np.nanpercentile(d["implied_volatility"], 98))
    vmin, vmax = min(lo_all), max(hi_all)
    levels = np.linspace(vmin, vmax, 16)
    norm = Normalize(vmin=vmin, vmax=vmax)

    last = None
    for k, (ds, d) in enumerate(prepared):
        ax = axes[0][k]
        iv = d["implied_volatility"].clip(vmin, vmax)
        last = ax.tricontourf(d["m"].values, d["dte"].values, iv.values,
                              levels=levels, cmap=CmapSeq, norm=norm, extend="both")
        ax.scatter(d["m"], d["dte"], s=1, c="k", alpha=0.10, linewidths=0)
        ax.axvline(0.0, color="white", lw=0.8, ls=":")
        ax.set_title(f"{ds}")
        ax.set_xlabel("log-moneyness  ln(K/S)")
        if k == 0:
            ax.set_ylabel("days to expiry")
        ax.set_ylim(0, 365)
        ax.grid(False)
    fig.colorbar(last, ax=axes[0].tolist(), label="implied vol", fraction=0.046, pad=0.02)
    fig.suptitle("Implied-volatility surface — SPY 2020  (liquid window)", fontweight="bold")
    fig.savefig(out / "fig5_surface.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    research = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/generated/research")
    figdir = research / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] calibration quality  ({(research/'calibration.csv').stat().st_size/1e6:.0f} MB, streamed)")
    daily, overall_rmse = calibration_quality(research / "calibration.csv")

    print("[2/6] term structure")
    ts, atm30, ts_dates, ts_grid = term_structure_analysis(research / "term_structure.csv")

    print("[3/6] skew evolution")
    sk30 = skew_analysis(research / "skew.csv")

    # Representative dates: calm (first), stress (max ~30d ATM), recovery (last).
    calm_date = atm30.iloc[0]["date"]
    stress_date = atm30.loc[atm30["atm_iv_30d"].idxmax(), "date"]
    recov_date = atm30.iloc[-1]["date"]
    snapshot_dates = [calm_date, stress_date, recov_date]
    print(f"      representative dates: calm={calm_date:%Y-%m-%d} "
          f"stress={stress_date:%Y-%m-%d} recovery={recov_date:%Y-%m-%d}")

    print(f"[4/6] smiles + curvature  ({(research/'smiles.csv').stat().st_size/1e6:.0f} MB, streamed)")
    curvature, snap_smiles, spots = smile_analysis(research / "smiles.csv", snapshot_dates)

    print(f"[5/6] surface snapshots  ({(research/'surface.csv').stat().st_size/1e6:.0f} MB, streamed)")
    surfaces = surface_snapshots(research / "surface.csv", [calm_date, stress_date])

    print("[6/6] figures")
    fig_calibration(daily, figdir)
    fig_term_structure(ts, atm30, ts_dates, ts_grid, snapshot_dates, figdir)
    fig_smiles(snap_smiles, curvature, figdir)
    fig_skew(sk30, figdir)
    fig_surface(surfaces, spots, figdir)

    # ---- summary statistics ---------------------------------------------
    merged = atm30.merge(sk30[["date", "risk_reversal", "butterfly"]], on="date", how="inner")
    merged = merged.merge(curvature[["date", "curvature"]], on="date", how="inner")

    def regime_rmse(lo, hi):
        m = daily[(daily["date"] >= lo) & (daily["date"] <= hi)]
        if m["n"].sum() == 0:
            return None
        return float(np.sqrt((m["rmse"] ** 2 * m["n"]).sum() / m["n"].sum()))

    stats = {
        "underlying": "SPY",
        "period": {"start": str(daily["date"].min().date()),
                   "end": str(daily["date"].max().date()),
                   "trading_days": int(len(daily))},
        "calibration": {
            "overall_iv_rmse_vs_provider": overall_rmse,
            "overall_iv_mae_vs_provider": float((daily["mae"] * daily["n"]).sum() / daily["n"].sum()),
            "mean_daily_iv_rmse": float(daily["rmse"].mean()),
            "mean_daily_iv_mae": float(daily["mae"].mean()),
            "median_daily_iv_rmse": float(daily["rmse"].median()),
            "worst_daily_iv_rmse": float(daily["rmse"].max()),
            "worst_day": str(daily.loc[daily["rmse"].idxmax(), "date"].date()),
            "mean_price_solver_residual": float(daily["mean_resid"].mean()),
            "by_regime": {k: regime_rmse(*[pd.Timestamp(x) for x in v])
                          for k, v in REGIMES.items()},
        },
        "atm_term_structure_30d": {
            "min": float(atm30["atm_iv_30d"].min()),
            "max": float(atm30["atm_iv_30d"].max()),
            "peak_date": str(stress_date.date()),
            "year_start": float(atm30.iloc[0]["atm_iv_30d"]),
            "year_end": float(atm30.iloc[-1]["atm_iv_30d"]),
        },
        "smile_curvature_30d": {
            "mean": float(curvature["curvature"].mean()),
            "min": float(curvature["curvature"].min()),
            "max": float(curvature["curvature"].max()),
            "at_stress": float(curvature.loc[
                (curvature["date"] - stress_date).abs().idxmin(), "curvature"]),
        },
        "risk_reversal_25d_30d": {
            "mean": float(sk30["risk_reversal"].mean()),
            "min": float(sk30["risk_reversal"].min()),
            "max": float(sk30["risk_reversal"].max()),
            "most_negative_date": str(sk30.loc[sk30["risk_reversal"].idxmin(), "date"].date()),
        },
        "butterfly_25d_30d": {
            "mean": float(sk30["butterfly"].mean()),
            "min": float(sk30["butterfly"].min()),
            "max": float(sk30["butterfly"].max()),
        },
        "skew_vol_correlation": {
            "corr_rr_atm": float(merged["risk_reversal"].corr(merged["atm_iv_30d"])),
            "corr_curvature_atm": float(merged["curvature"].corr(merged["atm_iv_30d"])),
        },
    }

    with open(research / "summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # also drop the per-day merged series for the report / further work
    merged.to_csv(research / "daily_series_30d.csv", index=False)
    daily.to_csv(research / "daily_calibration_rmse.csv", index=False)

    print("\n" + "=" * 66)
    print(f"SUMMARY — {stats['underlying']} {stats['period']['start']}..{stats['period']['end']} "
          f"({stats['period']['trading_days']} days)")
    print("=" * 66)
    print(json.dumps(stats, indent=2))
    print("\nFigures written to", figdir)


if __name__ == "__main__":
    main()
