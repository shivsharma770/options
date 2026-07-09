#!/usr/bin/env python3
"""
iv_rv_study.py -- Research Milestone 1: does today's ATM implied volatility
predict future realized volatility?

Reads the compact dataset produced by ``build_iv_rv_dataset.py``
(``daily_underlying.csv`` + ``atm_term_structure.csv``) and runs the standard
volatility-forecasting econometrics for SPY:

  * For each forecast horizon h in {5, 10, 20, 30, 60} trading days, take
    today's ATM implied vol at the *matched tenor* (nearest listed expiry to
    h/252 years) and the subsequent h-day annualised realised vol.
  * Mincer-Zarnowitz regression  RV = alpha + beta*IV  with Newey-West HAC
    standard errors (overlapping windows induce MA(h-1) residuals, so plain
    OLS SEs are invalid). Test the unbiased-efficient forecast null alpha=0,
    beta=1.
  * Log specification  ln RV = alpha + beta*ln IV.
  * Horse race vs the naive trailing-realised-vol forecast, plus an
    encompassing regression  RV = a + b1*IV + b2*HV.
  * Forecast-accuracy (RMSE/MAE/correlation) and bias / volatility-risk-premium
    diagnostics.

Outputs publication-quality figures, reproducible CSVs, and summary_stats.json.

Usage:
    iv_rv_study.py [master_dir] [out_dir]
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

# --------------------------------------------------------------------------
HORIZONS = [5, 10, 20, 30, 60]          # forecast horizons, trading days
TD_PER_YEAR = 252
TENOR_TOL = 0.60                        # accept matched IV within +/-60% of target tenor

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.grid": True, "grid.alpha": 0.30,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
ACCENT = "#1f6feb"; WARM = "#d1495b"; GOOD = "#2a9d8f"; NEUT = "#8d99ae"


# --------------------------------------------------------------------------
# Econometrics
# --------------------------------------------------------------------------
def ols_hac(y: np.ndarray, X: np.ndarray, L: int):
    """OLS with Newey-West (Bartlett) HAC covariance. X includes intercept.

    Returns dict with beta, se, tstat (H0: coef=0), R2, adj_R2, n, resid.
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    u = X * e[:, None]                      # (n,k) score contributions
    S = u.T @ u                             # lag-0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)             # Bartlett weight
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    ss_res = float(e @ e)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    return {"beta": beta, "se": se, "cov": cov, "tstat": beta / se,
            "r2": r2, "adj_r2": adj, "n": n, "resid": e}


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------
def build_panel(master: Path):
    spot = (pd.read_csv(master / "daily_underlying.csv", parse_dates=["date"])
            .dropna().drop_duplicates("date").sort_values("date")
            .reset_index(drop=True))
    spot["ret"] = np.log(spot["spot"]).diff()

    ts = pd.read_csv(master / "atm_term_structure.csv", parse_dates=["date"])
    ts = ts.dropna(subset=["atm_iv"])
    ts = ts[(ts["time_to_expiry"] > 0) & (ts["atm_iv"] > 0)]

    # matched-tenor ATM IV per (date, horizon)
    iv_cols = {}
    for h in HORIZONS:
        target = h / TD_PER_YEAR
        t = ts.copy()
        t["dist"] = (t["time_to_expiry"] - target).abs()
        idx = t.groupby("date")["dist"].idxmin()
        pick = t.loc[idx, ["date", "time_to_expiry", "atm_iv"]].copy()
        # reject if nearest listed expiry is too far from the target tenor
        ok = (pick["time_to_expiry"] - target).abs() <= TENOR_TOL * target
        pick = pick[ok]
        iv_cols[h] = pick.set_index("date")["atm_iv"].rename(f"iv_{h}")

    df = spot.set_index("date")
    for h in HORIZONS:
        df = df.join(iv_cols[h])

    r = df["ret"].values
    n = len(df)
    # realised vol: forward h and trailing h, annualised
    for h in HORIZONS:
        fwd = np.full(n, np.nan)
        trl = np.full(n, np.nan)
        for i in range(n):
            if i + h < n:
                seg = r[i + 1:i + 1 + h]
                if not np.isnan(seg).any():
                    fwd[i] = np.sqrt(TD_PER_YEAR / h * np.sum(seg ** 2))
            if i - h + 1 >= 1:
                seg = r[i - h + 1:i + 1]
                if not np.isnan(seg).any():
                    trl[i] = np.sqrt(TD_PER_YEAR / h * np.sum(seg ** 2))
        df[f"rv_{h}"] = fwd
        df[f"hv_{h}"] = trl
    return df.reset_index()


# --------------------------------------------------------------------------
# Per-horizon analysis
# --------------------------------------------------------------------------
def analyse_horizon(df: pd.DataFrame, h: int):
    sub = df[["date", f"iv_{h}", f"rv_{h}", f"hv_{h}"]].dropna()
    sub = sub.rename(columns={f"iv_{h}": "iv", f"rv_{h}": "rv", f"hv_{h}": "hv"})
    iv, rv, hv = sub["iv"].values, sub["rv"].values, sub["hv"].values
    n = len(sub)
    L = max(h - 1, 1)                      # Newey-West lag for h-overlap

    ones = np.ones(n)
    # Mincer-Zarnowitz: rv = a + b*iv
    mz = ols_hac(rv, np.column_stack([ones, iv]), L)
    a, b = mz["beta"]; se_a, se_b = mz["se"]
    t_b1 = (b - 1.0) / se_b               # H0: beta = 1
    # log spec
    mask = (iv > 0) & (rv > 0)
    lg = ols_hac(np.log(rv[mask]),
                 np.column_stack([ones[mask], np.log(iv[mask])]), L)
    # naive trailing-RV forecast
    hvr = ols_hac(rv, np.column_stack([ones, hv]), L)
    # encompassing: rv = a + b1*iv + b2*hv
    enc = ols_hac(rv, np.column_stack([ones, iv, hv]), L)

    err = iv - rv                          # forecast error (IV as forecast of RV)
    res = {
        "horizon_td": h,
        "n": n,
        "mean_iv": float(iv.mean()),
        "mean_rv": float(rv.mean()),
        "corr_iv_rv": float(np.corrcoef(iv, rv)[0, 1]),
        # Mincer-Zarnowitz
        "alpha": float(a), "se_alpha": float(se_a), "t_alpha": float(a / se_a),
        "beta": float(b), "se_beta": float(se_b),
        "t_beta_zero": float(b / se_b), "t_beta_one": float(t_b1),
        "r2": float(mz["r2"]), "adj_r2": float(mz["adj_r2"]),
        # forecast accuracy / bias
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias_iv_minus_rv": float(err.mean()),
        "vol_risk_premium_ratio": float(iv.mean() / rv.mean()),
        # log spec
        "log_beta": float(lg["beta"][1]), "log_se_beta": float(lg["se"][1]),
        "log_r2": float(lg["r2"]),
        # horse race
        "r2_iv_only": float(mz["r2"]),
        "r2_hv_only": float(hvr["r2"]),
        "enc_beta_iv": float(enc["beta"][1]), "enc_se_iv": float(enc["se"][1]),
        "enc_t_iv": float(enc["beta"][1] / enc["se"][1]),
        "enc_beta_hv": float(enc["beta"][2]), "enc_se_hv": float(enc["se"][2]),
        "enc_t_hv": float(enc["beta"][2] / enc["se"][2]),
        "enc_r2": float(enc["r2"]),
    }
    return res, sub


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_timeseries(df, h, out):
    sub = df[["date", f"iv_{h}", f"rv_{h}"]].dropna()
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.plot(sub["date"], sub[f"iv_{h}"], color=ACCENT, lw=1.1,
            label=f"ATM implied vol (tenor ~{h}d)")
    ax.plot(sub["date"], sub[f"rv_{h}"], color=WARM, lw=1.1, alpha=0.85,
            label=f"subsequent {h}-day realised vol")
    ax.set_title(f"Implied vs subsequently realised volatility — SPY  (h = {h} trading days)")
    ax.set_ylabel("annualised volatility")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(out / "fig1_iv_rv_timeseries.png")
    plt.close(fig)


def fig_scatter_grid(subs, results, out):
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(4.0 * len(HORIZONS), 4.0),
                             squeeze=False)
    for k, h in enumerate(HORIZONS):
        ax = axes[0][k]
        s = subs[h]; r = results[h]
        lim = [0, max(s["iv"].max(), s["rv"].max()) * 1.02]
        ax.scatter(s["iv"], s["rv"], s=6, alpha=0.25, color=ACCENT, linewidths=0)
        ax.plot(lim, lim, color="k", lw=0.9, ls=":", label="45° (RV = IV)")
        xs = np.linspace(s["iv"].min(), s["iv"].max(), 50)
        ax.plot(xs, r["alpha"] + r["beta"] * xs, color=WARM, lw=1.6,
                label=f"fit β={r['beta']:.2f}")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"h = {h}d   (R²={r['r2']:.2f})")
        ax.set_xlabel("implied vol")
        if k == 0:
            ax.set_ylabel("realised vol")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Realised vs implied volatility by horizon — SPY", fontweight="bold")
    fig.savefig(out / "fig2_scatter_by_horizon.png")
    plt.close(fig)


def fig_coeffs(results, out):
    hs = HORIZONS
    beta = [results[h]["beta"] for h in hs]
    beta_se = [results[h]["se_beta"] for h in hs]
    r2_iv = [results[h]["r2_iv_only"] for h in hs]
    r2_hv = [results[h]["r2_hv_only"] for h in hs]
    x = np.arange(len(hs))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].errorbar(x, beta, yerr=[1.96 * s for s in beta_se], fmt="o", color=ACCENT,
                   capsize=4, lw=1.4, label="β ± 1.96·HAC SE")
    ax[0].axhline(1.0, color=GOOD, ls="--", lw=1.2, label="β = 1 (unbiased)")
    ax[0].axhline(0.0, color=NEUT, ls=":", lw=1.0)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{h}d" for h in hs])
    ax[0].set_title("(a) Slope β  (RV = α + β·IV)")
    ax[0].set_xlabel("forecast horizon"); ax[0].set_ylabel("β")
    ax[0].legend(loc="best")

    w = 0.38
    ax[1].bar(x - w / 2, r2_iv, w, color=ACCENT, label="implied vol")
    ax[1].bar(x + w / 2, r2_hv, w, color=NEUT, label="trailing realised vol")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{h}d" for h in hs])
    ax[1].set_title("(b) Forecast R²: IV vs naive benchmark")
    ax[1].set_xlabel("forecast horizon"); ax[1].set_ylabel("R²")
    ax[1].legend(loc="best")
    fig.suptitle("Predictive power across horizons — SPY", fontweight="bold")
    fig.savefig(out / "fig3_coefficients.png")
    plt.close(fig)


def fig_bias(df, results, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    hs = HORIZONS
    bias = [results[h]["bias_iv_minus_rv"] for h in hs]
    ax[0].bar([f"{h}d" for h in hs], bias, color=[GOOD if b > 0 else WARM for b in bias])
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("(a) Mean forecast bias  E[IV − RV]  (volatility risk premium)")
    ax[0].set_xlabel("forecast horizon"); ax[0].set_ylabel("annualised vol points")

    h = 20
    sub = df[["date", f"iv_{h}", f"rv_{h}"]].dropna()
    vrp = sub[f"iv_{h}"] - sub[f"rv_{h}"]
    ax[1].plot(sub["date"], vrp, color="#5f6c7b", lw=0.9)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].fill_between(sub["date"], vrp, 0, where=(vrp >= 0), color=GOOD, alpha=0.35)
    ax[1].fill_between(sub["date"], vrp, 0, where=(vrp < 0), color=WARM, alpha=0.35)
    ax[1].set_title(f"(b) IV − RV over time  (h = {h}d)")
    ax[1].set_ylabel("vol points")
    ax[1].xaxis.set_major_locator(mdates.YearLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Forecast bias & the volatility risk premium — SPY", fontweight="bold")
    fig.savefig(out / "fig4_bias_vrp.png")
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    master = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/generated/research_m1")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else master
    figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)

    print("[1/4] building panel")
    df = build_panel(master)

    print("[2/4] regressions")
    results, subs = {}, {}
    long_rows = []
    for h in HORIZONS:
        res, sub = analyse_horizon(df, h)
        results[h] = res; subs[h] = sub
        tmp = sub.copy(); tmp["horizon_td"] = h
        long_rows.append(tmp)

    print("[3/4] figures")
    fig_timeseries(df, 20, figdir)
    fig_scatter_grid(subs, results, figdir)
    fig_coeffs(results, figdir)
    fig_bias(df, results, figdir)

    print("[4/4] exports")
    pd.concat(long_rows, ignore_index=True).to_csv(out / "iv_rv_dataset.csv", index=False)
    reg = pd.DataFrame([results[h] for h in HORIZONS])
    reg.to_csv(out / "regression_results.csv", index=False)

    period = (df["date"].min().date(), df["date"].max().date())
    summary = {
        "question": "Does today's ATM implied volatility predict future realised volatility?",
        "underlying": "SPY",
        "period": {"start": str(period[0]), "end": str(period[1]),
                   "trading_days": int(df["spot"].notna().sum())},
        "horizons_trading_days": HORIZONS,
        "method": "Mincer-Zarnowitz RV = alpha + beta*IV, Newey-West HAC (lag h-1), "
                  "matched-tenor ATM IV; log spec; IV-vs-trailing-RV horse race + encompassing.",
        "by_horizon": {h: results[h] for h in HORIZONS},
    }
    with open(out / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    # console table
    print("\n" + "=" * 92)
    print(f"SPY  {period[0]}..{period[1]}   IV -> future RV")
    print("=" * 92)
    hdr = f"{'h':>4} {'n':>5} {'beta':>7} {'t(b=1)':>7} {'alpha':>7} {'R2_IV':>6} " \
          f"{'R2_HV':>6} {'corr':>6} {'bias':>7} {'IV/RV':>6} {'enc_t_IV':>8} {'enc_t_HV':>8}"
    print(hdr)
    for h in HORIZONS:
        r = results[h]
        print(f"{h:>4} {r['n']:>5} {r['beta']:>7.3f} {r['t_beta_one']:>7.2f} "
              f"{r['alpha']:>7.3f} {r['r2']:>6.3f} {r['r2_hv_only']:>6.3f} "
              f"{r['corr_iv_rv']:>6.3f} {r['bias_iv_minus_rv']:>7.3f} "
              f"{r['vol_risk_premium_ratio']:>6.3f} {r['enc_t_iv']:>8.2f} {r['enc_t_hv']:>8.2f}")
    print("=" * 92)
    print("figures ->", figdir)


if __name__ == "__main__":
    main()
