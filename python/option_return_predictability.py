#!/usr/bin/env python3
"""
option_return_predictability.py -- Research Milestone 3.

Do option-surface features predict future SPY *returns* (not volatility)?
Pairs each daily surface with subsequent returns over h in {5,10,20,30,60}
trading days and tests four targets:

    future log return, future excess return, downside return, forward drawdown

against seven predictors (as of day t, ~1-month reference tenor):

    atm_iv, rr25 (risk reversal), bf25 (butterfly), curvature, slope,
    ts_slope (σ90 − σ30 term-structure slope), calib_rmse

Reports HAC (Newey-West) coefficient estimates and standard errors, predictive
R², individual and joint (Wald) significance tests, rolling coefficient/R²
estimates, and subperiod stability. Reuses the Milestone-1 HAC-OLS estimator
and the M1/M2 masters; introduces no trading strategy and no ML.

Usage:  option_return_predictability.py [master_dir] [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iv_rv_study import ols_hac                       # reuse HAC-OLS (no dup)

HORIZONS = [5, 10, 20, 30, 60]
TD = 252
RF_ANNUAL = 0.006                                     # ~avg 3M T-bill 2010-21 (documented proxy)
FEATURES = ["atm_iv", "rr25", "bf25", "curvature", "slope", "ts_slope", "calib_rmse"]
LABELS = {"atm_iv": "ATM IV", "rr25": "25Δ RR", "bf25": "25Δ fly",
          "curvature": "smile curv.", "slope": "smile slope",
          "ts_slope": "TS slope", "calib_rmse": "calib RMSE"}
TARGETS = ["logret", "excess", "downside", "drawdown"]
TGT_LABEL = {"logret": "future log return", "excess": "future excess return",
             "downside": "downside return", "drawdown": "forward drawdown"}
FOCAL_T, FOCAL_H = "excess", 20
ROLL = 504                                            # ~2y rolling window

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.grid": True, "grid.alpha": 0.30,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
ACCENT, WARM, GOOD, NEUT = "#1f6feb", "#d1495b", "#2a9d8f", "#8d99ae"


def nearest_iv(ts, target):
    t = ts.copy(); t["_d"] = (t["time_to_expiry"] - target).abs()
    idx = t.groupby("date")["_d"].idxmin()
    pick = t.loc[idx, ["date", "time_to_expiry", "atm_iv"]]
    ok = (pick["time_to_expiry"] - target).abs() <= 0.6 * target
    return pick[ok].set_index("date")["atm_iv"]


def build_panel(master):
    spot = (pd.read_csv(master / "daily_underlying.csv", parse_dates=["date"])
            .dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True))
    feat = pd.read_csv(master / "m2_features.csv", parse_dates=["date"]).drop_duplicates("date")
    calib = pd.read_csv(master / "m3_calib.csv", parse_dates=["date"])[["date", "calib_rmse"]]
    ts = pd.read_csv(master / "atm_term_structure.csv", parse_dates=["date"])
    ts = ts[(ts["time_to_expiry"] > 0) & (ts["atm_iv"] > 0)]
    tslope = (nearest_iv(ts, 90 / 365.0) - nearest_iv(ts, 30 / 365.0)).rename("ts_slope")

    df = (spot.set_index("date").join(feat.set_index("date"))
          .join(calib.set_index("date")).join(tslope)).reset_index()

    S = df["spot"].values
    n = len(df)
    logS = np.log(S)
    for h in HORIZONS:
        lr = np.full(n, np.nan); dd = np.full(n, np.nan)
        for i in range(n):
            if i + h < n:
                lr[i] = logS[i + h] - logS[i]
                path = S[i:i + h + 1]                 # includes entry as initial peak
                run_max = np.maximum.accumulate(path)
                dd[i] = float(np.min(path / run_max - 1.0))
        df[f"logret_{h}"] = lr
        df[f"excess_{h}"] = lr - RF_ANNUAL * (h / TD)
        df[f"downside_{h}"] = np.minimum(0.0, lr)
        df[f"drawdown_{h}"] = dd

    # standardize features (z-score) for comparable coefficients
    for f in FEATURES:
        df[f + "_z"] = (df[f] - df[f].mean()) / df[f].std(ddof=0)
    return df


def regress(df, target, h, cols=None):
    cols = cols or [f + "_z" for f in FEATURES]
    sub = df[["date"] + cols + [f"{target}_{h}"]].dropna().reset_index(drop=True)
    y = sub[f"{target}_{h}"].values
    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in cols])
    fit = ols_hac(y, X, max(h - 1, 1))
    k = X.shape[1]
    # joint Wald on all slope coefficients (exclude intercept)
    b = fit["beta"][1:]; V = fit["cov"][1:, 1:]
    wald = float(b @ np.linalg.solve(V, b))
    wald_p = float(stats.chi2.sf(wald, k - 1))
    return fit, sub, wald, wald_p


def analyse(df):
    res = {}
    for tgt in TARGETS:
        res[tgt] = {}
        for h in HORIZONS:
            fit, sub, wald, wald_p = regress(df, tgt, h)
            names = ["const"] + FEATURES
            res[tgt][h] = {
                "n": int(fit["n"]), "r2": float(fit["r2"]), "adj_r2": float(fit["adj_r2"]),
                "wald": wald, "wald_p": wald_p,
                "coef": {names[j]: float(fit["beta"][j]) for j in range(len(names))},
                "se": {names[j]: float(fit["se"][j]) for j in range(len(names))},
                "t": {names[j]: float(fit["tstat"][j]) for j in range(len(names))},
            }
    return res


def subperiods(df):
    edges = [("2010-01-01", "2013-12-31"), ("2014-01-01", "2017-12-31"),
             ("2018-01-01", "2021-12-31")]
    out = {}
    for lo, hi in edges:
        d = df[(df["date"] >= lo) & (df["date"] <= hi)]
        fit, sub, wald, wald_p = regress(d, FOCAL_T, FOCAL_H)
        names = ["const"] + FEATURES
        out[f"{lo[:4]}-{hi[:4]}"] = {
            "n": int(fit["n"]), "r2": float(fit["r2"]), "wald_p": wald_p,
            "t": {names[j]: float(fit["tstat"][j]) for j in range(len(names))},
            "coef": {names[j]: float(fit["beta"][j]) for j in range(len(names))},
        }
    return out


def rolling(df, target, h, window=ROLL):
    cols = [f + "_z" for f in FEATURES]
    sub = df[["date"] + cols + [f"{target}_{h}"]].dropna().reset_index(drop=True)
    y = sub[f"{target}_{h}"].values
    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in cols])
    dates, coefs, r2s = [], [], []
    for e in range(window, len(sub)):
        s = e - window
        Xt, yt = X[s:e], y[s:e]
        b = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        r = yt - Xt @ b
        r2 = 1 - (r @ r) / (((yt - yt.mean()) ** 2).sum())
        dates.append(sub["date"].iloc[e]); coefs.append(b[1:]); r2s.append(r2)
    return pd.DataFrame(coefs, columns=FEATURES, index=pd.to_datetime(dates)), pd.Series(r2s, index=pd.to_datetime(dates))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_coef_evolution(rc, out):
    key = ["atm_iv", "rr25", "calib_rmse", "ts_slope"]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    colors = {"atm_iv": ACCENT, "rr25": WARM, "calib_rmse": GOOD, "ts_slope": "#8552d6"}
    for f in key:
        ax.plot(rc.index, rc[f], lw=1.3, color=colors[f], label=LABELS[f])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Coefficient evolution — {ROLL//252}y rolling regression of "
                 f"{TGT_LABEL[FOCAL_T]} (h={FOCAL_H}d) on standardized features")
    ax.set_ylabel("coefficient (per 1σ feature)")
    ax.legend(ncol=4, loc="upper center")
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(out / "fig1_coef_evolution.png"); plt.close(fig)


def fig_rolling_power(df, out):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(HORIZONS)))
    for h, c in zip(HORIZONS, cmap):
        _, r2 = rolling(df, FOCAL_T, h)
        ax.plot(r2.index, r2, lw=1.2, color=c, label=f"h={h}d")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Rolling predictive power — {ROLL//252}y in-sample R² of "
                 f"{TGT_LABEL[FOCAL_T]} on all features")
    ax.set_ylabel("rolling R²"); ax.legend(ncol=5, loc="upper center")
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(out / "fig2_rolling_power.png"); plt.close(fig)


def fig_scatter(df, results, out):
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(4.0 * len(HORIZONS), 4.0), squeeze=False)
    cols = [f + "_z" for f in FEATURES]
    for k, h in enumerate(HORIZONS):
        ax = axes[0][k]
        sub = df[["date"] + cols + [f"{FOCAL_T}_{h}"]].dropna()
        X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in cols])
        b = np.linalg.lstsq(X, sub[f"{FOCAL_T}_{h}"].values, rcond=None)[0]
        pred = X @ b; act = sub[f"{FOCAL_T}_{h}"].values
        ax.scatter(pred, act, s=6, alpha=0.25, color=ACCENT, linewidths=0)
        lim = [min(pred.min(), act.min()), max(pred.max(), act.max())]
        ax.plot(lim, lim, "k:", lw=0.9)
        ax.set_title(f"h = {h}d   (R²={results[FOCAL_T][h]['r2']:.3f})")
        ax.set_xlabel("predicted")
        if k == 0:
            ax.set_ylabel(f"actual {TGT_LABEL[FOCAL_T]}")
    fig.suptitle(f"In-sample fit: predicted vs actual {TGT_LABEL[FOCAL_T]} — SPY", fontweight="bold")
    fig.savefig(out / "fig3_return_scatter.png"); plt.close(fig)


def fig_crashes(df, out):
    # worst forward-20d drawdown day within each calendar crisis window
    windows = {"2011 (Euro/US debt)": ("2011-06-01", "2011-10-31"),
               "2018 (Q4 selloff)": ("2018-09-01", "2018-12-31"),
               "2020 (COVID-19)": ("2020-01-15", "2020-04-30")}
    fig, axes = plt.subplots(1, len(windows), figsize=(5.4 * len(windows), 4.4), squeeze=False)
    for k, (name, (lo, hi)) in enumerate(windows.items()):
        ax = axes[0][k]
        w = df[(df["date"] >= lo) & (df["date"] <= hi)]
        trough = w.loc[w["drawdown_20"].idxmin(), "date"]
        win = df[(df["date"] >= trough - pd.Timedelta(days=50)) &
                 (df["date"] <= trough + pd.Timedelta(days=25))].copy()
        l_spy, = ax.plot(win["date"], win["spot"] / win["spot"].iloc[0], color="k",
                         lw=1.8, label="SPY (norm., left)")
        ax2 = ax.twinx()
        l_iv, = ax2.plot(win["date"], win["atm_iv"], color=ACCENT, lw=1.5, label="ATM IV")
        l_rr, = ax2.plot(win["date"], -win["rr25"], color=WARM, lw=1.5, label="−RR (put skew)")
        ax.axvline(trough, color="grey", ls=":", lw=1.2)
        ax.set_title(name); ax.grid(False); ax2.grid(False)
        ax.set_ylabel("SPY (normalized)" if k == 0 else "")
        if k == len(windows) - 1:
            ax2.set_ylabel("implied-vol / skew level")
        ax.legend([l_spy, l_iv, l_rr], [h.get_label() for h in (l_spy, l_iv, l_rr)],
                  loc="lower left", fontsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.suptitle("Feature behaviour around major drawdowns — dotted line = forward-DD trough",
                 fontweight="bold")
    fig.savefig(out / "fig4_crash_events.png"); plt.close(fig)


def fig_predrawdown(df, out):
    thr = df["drawdown_20"].quantile(0.10)                 # worst 10% forward drawdowns
    pre = df["drawdown_20"] <= thr
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    diffs = {}
    for ax, f in zip(axes.ravel(), FEATURES):
        a = df.loc[pre & df[f].notna(), f]; b = df.loc[~pre & df[f].notna(), f]
        lo, hi = df[f].quantile(0.01), df[f].quantile(0.99)
        bins = np.linspace(lo, hi, 40)
        ax.hist(b, bins=bins, density=True, alpha=0.55, color=NEUT, label="normal")
        ax.hist(a, bins=bins, density=True, alpha=0.6, color=WARM, label="pre-drawdown")
        ax.axvline(b.mean(), color=NEUT, ls="--", lw=1); ax.axvline(a.mean(), color=WARM, ls="--", lw=1)
        ax.set_title(LABELS[f], fontsize=11)
        diffs[f] = float((a.mean() - b.mean()) / df[f].std())
    axes.ravel()[0].legend(fontsize=8, loc="upper right")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Feature distributions: days before the worst-decile 20-day drawdowns vs normal days",
                 fontweight="bold")
    fig.savefig(out / "fig5_predrawdown_distributions.png"); plt.close(fig)
    return diffs


# --------------------------------------------------------------------------
def main():
    master = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/generated/research_m1")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/generated/research_m3")
    figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)

    print("[1/5] panel"); df = build_panel(master)
    print("[2/5] regressions"); results = analyse(df)
    print("[3/5] subperiods + rolling"); subs = subperiods(df)
    rc, _ = rolling(df, FOCAL_T, FOCAL_H)

    print("[4/5] figures")
    fig_coef_evolution(rc, figdir)
    fig_rolling_power(df, figdir)
    fig_scatter(df, results, figdir)
    fig_crashes(df, figdir)
    predd = fig_predrawdown(df, figdir)

    print("[5/5] exports")
    keep = ["date"] + FEATURES + [f"{t}_{h}" for t in TARGETS for h in HORIZONS]
    df[keep].to_csv(out / "m3_return_dataset.csv", index=False)
    flat = [{"target": t, "horizon": h, "n": results[t][h]["n"],
             "r2": results[t][h]["r2"], "adj_r2": results[t][h]["adj_r2"],
             "wald_p": results[t][h]["wald_p"],
             **{f"t_{f}": results[t][h]["t"][f] for f in FEATURES}}
            for t in TARGETS for h in HORIZONS]
    pd.DataFrame(flat).to_csv(out / "m3_regression_results.csv", index=False)

    summary = {
        "question": "Do option-surface features predict future SPY returns?",
        "underlying": "SPY", "rf_annual_proxy": RF_ANNUAL,
        "period": {"start": str(df["date"].min().date()), "end": str(df["date"].max().date())},
        "features": FEATURES, "targets": TARGETS,
        "predictive_r2": {t: {h: results[t][h]["r2"] for h in HORIZONS} for t in TARGETS},
        "joint_wald_p": {t: {h: results[t][h]["wald_p"] for h in HORIZONS} for t in TARGETS},
        "focal": {"target": FOCAL_T, "horizon": FOCAL_H,
                  "coef": results[FOCAL_T][FOCAL_H]["coef"],
                  "t": results[FOCAL_T][FOCAL_H]["t"]},
        "subperiods": subs,
        "predrawdown_mean_shift_in_sd": predd,
    }
    with open(out / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 96)
    print(f"SPY {df['date'].min().date()}..{df['date'].max().date()}  option features -> future returns")
    print("=" * 96)
    print("Predictive R² (joint Wald p in parentheses):")
    print(f"{'target':>18} " + " ".join(f"{h:>12}d" for h in HORIZONS))
    for t in TARGETS:
        cells = [f"{results[t][h]['r2']*100:5.2f}% ({results[t][h]['wald_p']:.2f})" for h in HORIZONS]
        print(f"{TGT_LABEL[t]:>18} " + " ".join(f"{c:>13}" for c in cells))
    print("=" * 96)
    print(f"Focal ({TGT_LABEL[FOCAL_T]}, h={FOCAL_H}) HAC t-stats:",
          {f: round(results[FOCAL_T][FOCAL_H]['t'][f], 2) for f in FEATURES})
    print("Pre-drawdown mean shift (in σ):", {k: round(v, 2) for k, v in predd.items()})
    print("figures ->", figdir)


if __name__ == "__main__":
    main()
