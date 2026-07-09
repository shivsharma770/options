#!/usr/bin/env python3
"""
iv_smile_forecast_study.py -- Research Milestone 2.

Does the *shape* of the volatility surface (skew, curvature, term-structure
slope) forecast future realized volatility beyond the ATM implied-vol *level*?

Extends Milestone 1. Reuses:
  * the realized-volatility construction and HAC-OLS estimator from
    ``iv_rv_study`` (imported, not reimplemented);
  * the daily underlying price and ATM term structure masters from M1;
  * the smile-shape features extracted by ``build_m2_features.py`` from the
    C++ ``volatility_analytics`` outputs (skew.csv / smiles.csv).

Feature set (as of day t, ~1-month reference tenor):
    atm_iv, rr25, bf25, slope, curvature, ts_slope_short, ts_slope_long

Nested models, per horizon h in {5,10,20,30,60} trading days:
    A  ATM IV only          RV ~ atm_iv
    B  ATM IV + smile        RV ~ atm_iv + (all shape features)
    C  smile features alone  RV ~ (all shape features, no atm_iv)

Reports in-sample coefficients with Newey-West HAC SEs, adjusted R²,
incremental F-test (B vs A) and a HAC-robust Wald test, VIFs, the feature
correlation matrix, and expanding-window out-of-sample R². Produces
publication-quality figures and reproducible CSVs.

Usage:  iv_smile_forecast_study.py [master_dir] [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iv_rv_study import ols_hac                       # reuse HAC-OLS (no dup)

HORIZONS = [5, 10, 20, 30, 60]
TD = 252
SMILE = ["rr25", "bf25", "slope", "curvature", "ts_slope_short", "ts_slope_long"]
FEATURES = ["atm_iv"] + SMILE
LABELS = {"atm_iv": "ATM IV", "rr25": "25Δ RR", "bf25": "25Δ fly",
          "slope": "smile slope", "curvature": "smile curv.",
          "ts_slope_short": "TS slope (S)", "ts_slope_long": "TS slope (L)"}
OOS_MIN_TRAIN = 504                                   # ~2y before first OOS forecast

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.grid": True, "grid.alpha": 0.30,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
ACCENT, WARM, GOOD, NEUT = "#1f6feb", "#d1495b", "#2a9d8f", "#8d99ae"


# --------------------------------------------------------------------------
def nearest_iv(ts, target):
    t = ts.copy()
    t["_d"] = (t["time_to_expiry"] - target).abs()
    idx = t.groupby("date")["_d"].idxmin()
    pick = t.loc[idx, ["date", "time_to_expiry", "atm_iv"]]
    ok = (pick["time_to_expiry"] - target).abs() <= 0.6 * target
    return pick[ok].set_index("date")["atm_iv"]


def build_panel(master: Path) -> pd.DataFrame:
    spot = (pd.read_csv(master / "daily_underlying.csv", parse_dates=["date"])
            .dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True))
    spot["ret"] = np.log(spot["spot"]).diff()

    feat = pd.read_csv(master / "m2_features.csv", parse_dates=["date"]).drop_duplicates("date")

    ts = pd.read_csv(master / "atm_term_structure.csv", parse_dates=["date"])
    ts = ts[(ts["time_to_expiry"] > 0) & (ts["atm_iv"] > 0)]
    iv7 = nearest_iv(ts, 7 / 365.0)
    iv30 = nearest_iv(ts, 30 / 365.0)
    iv90 = nearest_iv(ts, 90 / 365.0)
    tss = pd.DataFrame({"ts_slope_short": iv30 - iv7, "ts_slope_long": iv90 - iv30}).dropna()

    df = spot.set_index("date").join(feat.set_index("date")).join(tss)

    r = df["ret"].values
    n = len(df)
    for h in HORIZONS:
        fwd = np.full(n, np.nan)
        for i in range(n):
            if i + h < n:
                seg = r[i + 1:i + 1 + h]
                if not np.isnan(seg).any():
                    fwd[i] = np.sqrt(TD / h * np.sum(seg ** 2))
        df[f"rv_{h}"] = fwd
    return df.reset_index()


# --------------------------------------------------------------------------
def vif_table(X: np.ndarray, names):
    """X: n x k feature matrix (no intercept). VIF_j = 1/(1-R2_j)."""
    n, k = X.shape
    out = {}
    for j in range(k):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        Z = np.column_stack([np.ones(n), others])
        b = np.linalg.lstsq(Z, y, rcond=None)[0]
        e = y - Z @ b
        r2 = 1 - (e @ e) / (((y - y.mean()) ** 2).sum())
        out[names[j]] = float(1.0 / max(1e-12, 1.0 - r2))
    return out


def oos_r2(dates, X, y, h, n0=OOS_MIN_TRAIN):
    """Expanding-window OOS R2 vs the historical-mean benchmark. Look-ahead
    guard: at origin i, train only on rows whose h-day target has closed
    (index <= i-h)."""
    n = len(y)
    num = den = 0.0
    for i in range(n0, n):
        te = i - h
        if te < 252:
            continue
        Xt, yt = X[:te + 1], y[:te + 1]
        b = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        pred = X[i] @ b
        bench = yt.mean()
        num += (y[i] - pred) ** 2
        den += (y[i] - bench) ** 2
    return 1.0 - num / den if den > 0 else np.nan


def analyse_horizon(df, h):
    cols = FEATURES + [f"rv_{h}"]
    sub = df[["date"] + cols].dropna().reset_index(drop=True)
    y = sub[f"rv_{h}"].values
    n = len(sub)
    L = max(h - 1, 1)
    ones = np.ones(n)
    Xfull = sub[FEATURES].values
    Xsm = sub[SMILE].values

    designs = {
        "A_atm":   np.column_stack([ones, sub[["atm_iv"]].values]),
        "B_all":   np.column_stack([ones, Xfull]),
        "C_smile": np.column_stack([ones, Xsm]),
    }
    names = {"A_atm": ["const", "atm_iv"],
             "B_all": ["const"] + FEATURES,
             "C_smile": ["const"] + SMILE}

    fits = {m: ols_hac(y, X, L) for m, X in designs.items()}

    # incremental F-test (classic) B vs A, and HAC-robust Wald on the 6 smile coefs
    rss_a = float(fits["A_atm"]["resid"] @ fits["A_atm"]["resid"])
    rss_b = float(fits["B_all"]["resid"] @ fits["B_all"]["resid"])
    q = len(SMILE)
    dfd = n - designs["B_all"].shape[1]
    F = ((rss_a - rss_b) / q) / (rss_b / dfd)
    F_p = float(stats.f.sf(F, q, dfd))
    # HAC Wald: R selects the 6 smile coefs in model B (positions 2..7)
    fb = fits["B_all"]
    sel = list(range(2, 2 + q))
    bsel = fb["beta"][sel]
    Vsel = fb["cov"][np.ix_(sel, sel)]
    wald = float(bsel @ np.linalg.solve(Vsel, bsel))
    wald_p = float(stats.chi2.sf(wald, q))

    # standardized coefficients for model B (comparability across features/horizons)
    sy = y.std(ddof=1)
    std_coef = {}
    for j, f in enumerate(FEATURES):
        std_coef[f] = float(fb["beta"][1 + j] * sub[f].std(ddof=1) / sy)

    res = {
        "horizon_td": h, "n": int(n),
        "r2_A": float(fits["A_atm"]["r2"]), "adjr2_A": float(fits["A_atm"]["adj_r2"]),
        "r2_B": float(fits["B_all"]["r2"]), "adjr2_B": float(fits["B_all"]["adj_r2"]),
        "r2_C": float(fits["C_smile"]["r2"]), "adjr2_C": float(fits["C_smile"]["adj_r2"]),
        "incr_F": float(F), "incr_F_df": [q, dfd], "incr_F_p": F_p,
        "hac_wald": wald, "hac_wald_df": q, "hac_wald_p": wald_p,
        "oos_r2_A": float(oos_r2(sub["date"], designs["A_atm"], y, h)),
        "oos_r2_B": float(oos_r2(sub["date"], designs["B_all"], y, h)),
        "oos_r2_C": float(oos_r2(sub["date"], designs["C_smile"], y, h)),
        "coef_B": {names["B_all"][k]: float(fb["beta"][k]) for k in range(len(fb["beta"]))},
        "se_B": {names["B_all"][k]: float(fb["se"][k]) for k in range(len(fb["se"]))},
        "t_B": {names["B_all"][k]: float(fb["tstat"][k]) for k in range(len(fb["tstat"]))},
        "std_coef_B": std_coef,
    }
    return res, sub, fits


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_coef_heatmap(results, out):
    M = np.array([[results[h]["std_coef_B"][f] for h in HORIZONS] for f in FEATURES])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(HORIZONS))); ax.set_xticklabels([f"{h}d" for h in HORIZONS])
    ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels([LABELS[f] for f in FEATURES])
    for i in range(len(FEATURES)):
        for j in range(len(HORIZONS)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(M[i, j]) > 0.55 * lim else "black")
    ax.set_title("Standardized coefficients — model B (ATM IV + smile) by horizon")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="std. coefficient", fraction=0.046, pad=0.02)
    fig.savefig(out / "fig1_coef_heatmap.png"); plt.close(fig)


def fig_importance(results, out):
    imp = {f: np.mean([abs(results[h]["t_B"][f]) for h in HORIZONS]) for f in FEATURES}
    order = sorted(FEATURES, key=lambda f: imp[f])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh([LABELS[f] for f in order], [imp[f] for f in order],
            color=[ACCENT if f == "atm_iv" else NEUT for f in order])
    ax.axvline(1.96, color=WARM, ls="--", lw=1.1, label="|t| = 1.96")
    ax.set_title("Feature importance — mean |HAC t| across horizons (model B)")
    ax.set_xlabel("mean |t-statistic|"); ax.legend()
    fig.savefig(out / "fig2_feature_importance.png"); plt.close(fig)


def fig_corr(df, out):
    C = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(C.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels([LABELS[f] for f in FEATURES],
                                                            rotation=40, ha="right")
    ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels([LABELS[f] for f in FEATURES])
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(C.values[i, j]) > 0.6 else "black")
    ax.set_title("Feature correlation matrix")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out / "fig3_correlation_heatmap.png"); plt.close(fig)


def fig_incremental(results, out):
    x = np.arange(len(HORIZONS)); w = 0.26
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for k, (key, lab, col) in enumerate([("adjr2_A", "A: ATM only", ACCENT),
                                         ("adjr2_B", "B: ATM+smile", GOOD),
                                         ("adjr2_C", "C: smile only", NEUT)]):
        ax[0].bar(x + (k - 1) * w, [results[h][key] for h in HORIZONS], w, color=col, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{h}d" for h in HORIZONS])
    ax[0].set_title("(a) In-sample adjusted R²"); ax[0].set_ylabel("adj R²")
    ax[0].set_xlabel("horizon"); ax[0].legend()

    for k, (key, lab, col) in enumerate([("oos_r2_A", "A: ATM only", ACCENT),
                                         ("oos_r2_B", "B: ATM+smile", GOOD),
                                         ("oos_r2_C", "C: smile only", NEUT)]):
        ax[1].bar(x + (k - 1) * w, [results[h][key] for h in HORIZONS], w, color=col, label=lab)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{h}d" for h in HORIZONS])
    ax[1].set_title("(b) Out-of-sample R²  (vs historical mean)"); ax[1].set_ylabel("OOS R²")
    ax[1].set_xlabel("horizon"); ax[1].legend()
    fig.suptitle("Incremental explanatory power: does smile add beyond ATM IV? — SPY",
                 fontweight="bold")
    fig.savefig(out / "fig4_incremental_power.png"); plt.close(fig)


def fig_resid(fits, sub, h, out):
    fit = fits["B_all"]
    e = fit["resid"]
    yhat = sub[f"rv_{h}"].values - e
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].scatter(yhat, e, s=6, alpha=0.25, color=ACCENT, linewidths=0)
    ax[0, 0].axhline(0, color="k", lw=0.8)
    ax[0, 0].set_title("(a) Residuals vs fitted"); ax[0, 0].set_xlabel("fitted RV"); ax[0, 0].set_ylabel("residual")

    stats.probplot(e, dist="norm", plot=ax[0, 1])
    ax[0, 1].set_title("(b) Normal Q–Q")

    m = min(len(e) - 1, 60)
    ac = [1.0] + [np.corrcoef(e[:-k], e[k:])[0, 1] for k in range(1, m + 1)]
    ax[1, 0].bar(range(len(ac)), ac, color=NEUT)
    ci = 1.96 / np.sqrt(len(e))
    ax[1, 0].axhline(ci, color=WARM, ls="--", lw=1); ax[1, 0].axhline(-ci, color=WARM, ls="--", lw=1)
    ax[1, 0].axvline(h, color=GOOD, ls=":", lw=1.2, label=f"lag = h = {h}")
    ax[1, 0].set_title("(c) Residual autocorrelation (overlap → HAC needed)")
    ax[1, 0].set_xlabel("lag (days)"); ax[1, 0].set_ylabel("ACF"); ax[1, 0].legend()

    ax[1, 1].hist(e, bins=50, density=True, color="#4c72b0", edgecolor="white")
    xs = np.linspace(e.min(), e.max(), 200)
    ax[1, 1].plot(xs, stats.norm.pdf(xs, e.mean(), e.std()), color=WARM, lw=1.5, label="normal")
    ax[1, 1].set_title("(d) Residual distribution"); ax[1, 1].set_xlabel("residual"); ax[1, 1].legend()
    fig.suptitle(f"Residual diagnostics — model B, h = {h} trading days", fontweight="bold")
    fig.savefig(out / "fig5_residual_diagnostics.png"); plt.close(fig)


# --------------------------------------------------------------------------
def main():
    master = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/generated/research_m1")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/generated/research_m2")
    figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)

    print("[1/5] panel"); df = build_panel(master)
    print("[2/5] regressions + OOS")
    results, subs, fitset = {}, {}, {}
    for h in HORIZONS:
        r, s, f = analyse_horizon(df, h)
        results[h], subs[h], fitset[h] = r, s, f

    # VIF + correlation on the full-model feature set (drop NaNs jointly)
    feat_df = df[["date"] + FEATURES].dropna().reset_index(drop=True)
    vifs = vif_table(feat_df[FEATURES].values, FEATURES)
    corr = feat_df[FEATURES].corr()

    print("[3/5] figures")
    fig_coef_heatmap(results, figdir)
    fig_importance(results, figdir)
    fig_corr(feat_df, figdir)
    fig_incremental(results, figdir)
    fig_resid(fitset[20], subs[20], 20, figdir)

    print("[4/5] exports")
    df[["date"] + FEATURES + [f"rv_{h}" for h in HORIZONS]].to_csv(
        out / "m2_forecast_dataset.csv", index=False)
    flat = []
    for h in HORIZONS:
        r = results[h]
        row = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        flat.append(row)
    pd.DataFrame(flat).to_csv(out / "m2_regression_results.csv", index=False)
    pd.Series(vifs, name="VIF").to_csv(out / "m2_vif.csv")
    corr.to_csv(out / "m2_feature_correlation.csv")

    summary = {
        "question": "Do smile-shape features forecast realized volatility beyond ATM IV?",
        "underlying": "SPY",
        "period": {"start": str(df["date"].min().date()), "end": str(df["date"].max().date())},
        "features": FEATURES,
        "vif": vifs,
        "by_horizon": results,
    }
    with open(out / "summary_stats.json", "w") as fjs:
        json.dump(summary, fjs, indent=2)

    # console
    print("[5/5] summary\n" + "=" * 100)
    print(f"SPY {df['date'].min().date()}..{df['date'].max().date()}   smile vs ATM-IV forecasting")
    print("=" * 100)
    print(f"{'h':>4} {'adjR2_A':>8} {'adjR2_B':>8} {'adjR2_C':>8} {'incrF':>7} {'F_p':>8} "
          f"{'Wald_p':>8} {'OOS_A':>7} {'OOS_B':>7} {'OOS_C':>7}")
    for h in HORIZONS:
        r = results[h]
        print(f"{h:>4} {r['adjr2_A']:>8.3f} {r['adjr2_B']:>8.3f} {r['adjr2_C']:>8.3f} "
              f"{r['incr_F']:>7.2f} {r['incr_F_p']:>8.1e} {r['hac_wald_p']:>8.1e} "
              f"{r['oos_r2_A']:>7.3f} {r['oos_r2_B']:>7.3f} {r['oos_r2_C']:>7.3f}")
    print("=" * 100)
    print("VIF:", {k: round(v, 1) for k, v in vifs.items()})
    print("figures ->", figdir)


if __name__ == "__main__":
    main()
