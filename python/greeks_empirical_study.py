#!/usr/bin/env python3
"""
greeks_empirical_study.py -- Research Milestone 4, empirical part.

Uses the compact per-day ~1-month smile panel (`build_m4_greeks_panel.py`) plus
the closed-form Greeks in `higher_order_greeks.py` to answer, on 12 years of SPY
option surfaces, which higher-order Greeks carry economically meaningful
information. Produces:

  * Greek profiles across moneyness (calm vs March-2020) -- theory on real data
  * second-order P&L attribution by market regime (where 1st-order Greeks fail)
  * higher-order Greek time series and March-2020 extremes
  * cross-Greek correlation matrix and PCA (redundancy / new information)
  * regime comparison (low-vol / COVID / recovery)
  * empirical tests: does Vomma precede IV moves? does Vanna/Gamma explain the
    delta-hedge error?

Reuses the calibration outputs and analytics; introduces no pricing model.
Usage:  greeks_empirical_study.py [master_dir] [out_dir]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import higher_order_greeks as hg
from iv_rv_study import ols_hac                      # reuse HAC-OLS

TD = 252
HO = ["vanna", "vomma", "charm", "veta", "speed", "zomma", "color", "ultima"]
ALLG = ["delta", "gamma", "vega", "theta", "rho"] + HO
REP_M = -0.05                                        # representative ~25Δ-put moneyness
COVID = ("2020-02-20", "2020-04-30")

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.grid": True, "grid.alpha": 0.30,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
ACCENT, WARM, GOOD, NEUT = "#1f6feb", "#d1495b", "#2a9d8f", "#8d99ae"


def load_days(master: Path):
    df = pd.read_csv(master / "m4_smile30.csv", parse_dates=["date"])
    df = df[(df["implied_volatility"] > 0) & (df["time_to_expiry"] > 0)]
    return {d: g.sort_values("moneyness") for d, g in df.groupby("date")}


def greek_at(g, m_target, which):
    """Greek `which` (call) at the smile strike nearest moneyness m_target."""
    i = (g["moneyness"] - m_target).abs().values.argmin()
    row = g.iloc[i]
    gk = hg.greeks(row["spot"], row["strike"], row["time_to_expiry"],
                   row["implied_volatility"], 0.0, 0.0, call=True)
    return float(gk[which]), row


def build_daily(days):
    """Daily panel: representative-option Greeks + ATM inputs for attribution."""
    recs = []
    for d, g in days.items():
        S = float(g["spot"].iloc[0]); T = float(g["time_to_expiry"].median())
        atm_iv = float(greek_at(g, 0.0, "vega")[1]["implied_volatility"])
        i0 = (g["moneyness"]).abs().values.argmin()
        atm = g.iloc[i0]
        # representative 25Δ-put-ish option: all Greeks
        ir = (g["moneyness"] - REP_M).abs().values.argmin()
        rr = g.iloc[ir]
        gk = hg.greeks(rr["spot"], rr["strike"], rr["time_to_expiry"],
                       rr["implied_volatility"], 0.0, 0.0, call=True)
        rec = {"date": d, "spot": S, "T": T, "atm_iv": atm_iv,
               "atm_K": float(atm["strike"]), "atm_T": float(atm["time_to_expiry"])}
        rec.update({f"g_{k}": float(gk[k]) for k in ALLG})
        recs.append(rec)
    daily = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
    daily["ret"] = np.log(daily["spot"]).diff()
    daily["d_iv"] = daily["atm_iv"].diff()
    return daily


def smile_interp(g):
    m = g["moneyness"].values; v = g["implied_volatility"].values
    return lambda x: float(np.interp(x, m, v))


def pnl_attribution(days, daily):
    """Hold the ATM option one day; decompose actual dV into 1st/2nd-order Greek
    terms using the realized dS, d(sigma) and the fixed-strike smile."""
    dts = daily["date"].tolist()
    interp = {d: smile_interp(g) for d, g in days.items()}
    rows = []
    for i in range(len(dts) - 1):
        d, dn = dts[i], dts[i + 1]
        r = daily.iloc[i]
        S, K, T, sig = r["spot"], r["atm_K"], r["atm_T"], r["atm_iv"]
        Sn = daily.iloc[i + 1]["spot"]
        Tn = max(T - 1.0 / TD, 1e-4)
        mn = np.log(K / Sn)
        sign = interp[dn](mn)                    # fixed-strike vol next day
        dV = hg.bs_price(Sn, K, Tn, sign, 0, 0, True) - hg.bs_price(S, K, T, sig, 0, 0, True)
        gk = hg.greeks(S, K, T, sig, 0, 0, True)
        dS, dsig = Sn - S, sign - sig
        # theta is d/dt (per year of calendar time); one day forward => +1/252
        first = gk["delta"] * dS + gk["theta"] * (1.0 / TD) + gk["vega"] * dsig
        second = (0.5 * gk["gamma"] * dS ** 2
                  + gk["vanna"] * dS * dsig
                  + 0.5 * gk["vomma"] * dsig ** 2
                  + gk["charm"] * dS * (1.0 / TD)
                  + gk["veta"] * dsig * (1.0 / TD))
        rows.append({"date": d, "dV": dV, "first": first, "second": second,
                     "ret": np.log(Sn / S), "d_iv": dsig, "atm_iv": sig,
                     "gamma_term": 0.5 * gk["gamma"] * dS ** 2,
                     "vanna_term": gk["vanna"] * dS * dsig,
                     "vega_term": gk["vega"] * dsig,
                     "hedge_err": dV - gk["delta"] * dS})
    return pd.DataFrame(rows)


def expl_var(actual, model):
    a = np.asarray(actual); m = np.asarray(model)
    return 1.0 - np.nansum((a - m) ** 2) / np.nansum((a - np.nanmean(a)) ** 2)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_profiles(days, daily, out):
    calm = daily.loc[daily["atm_iv"].idxmin(), "date"]
    crash = daily[(daily["date"] >= COVID[0]) & (daily["date"] <= COVID[1])]
    crash = crash.loc[crash["atm_iv"].idxmax(), "date"]
    profile = ["vanna", "vomma", "charm", "speed", "zomma", "color"]
    grid = np.linspace(-0.18, 0.18, 90)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, gname in zip(axes.flat, profile):
        for d, c, lab in [(calm, ACCENT, f"calm {pd.Timestamp(calm):%Y-%m-%d}"),
                          (crash, WARM, f"COVID {pd.Timestamp(crash):%Y-%m-%d}")]:
            g = days[pd.Timestamp(d)]
            S = float(g["spot"].iloc[0]); T = float(g["time_to_expiry"].median())
            # smooth the raw smile (degree-5 in log-moneyness) so higher-order
            # differencing is not swamped by strike-grid / quote quantization
            coef = np.polyfit(g["moneyness"], g["implied_volatility"], 5)
            ivg = np.polyval(coef, grid)
            vals = hg.greeks(S, S * np.exp(grid), T, ivg, 0, 0, True)[gname]
            ax.plot(grid, vals, color=c, lw=1.8, label=lab)
        ax.axvline(0, color="grey", lw=0.7, ls=":")
        ax.set_title(gname.capitalize()); ax.set_xlabel("log-moneyness")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Higher-order Greek profiles across the 1-month smile — SPY", fontweight="bold")
    fig.savefig(out / "fig1_greek_profiles.png"); plt.close(fig)


def fig_timeseries(daily, out):
    fig, ax = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    ax[0].plot(daily["date"], daily["atm_iv"], color="k", lw=0.9)
    ax[0].set_ylabel("ATM IV"); ax[0].set_title("(a) ATM implied volatility")
    ax[1].plot(daily["date"], daily["g_vanna"], color=ACCENT, lw=0.8, label="Vanna")
    ax[1].plot(daily["date"], daily["g_charm"], color=GOOD, lw=0.8, label="Charm")
    ax[1].set_ylabel("per-option"); ax[1].legend(loc="upper left"); ax[1].set_title("(b) Vanna, Charm")
    ax[2].plot(daily["date"], daily["g_vomma"], color=WARM, lw=0.8, label="Vomma")
    ax[2].plot(daily["date"], daily["g_ultima"] / 50.0, color="#8250df", lw=0.8, label="Ultima /50")
    ax[2].set_ylabel("per-option"); ax[2].legend(loc="upper left"); ax[2].set_title("(c) Vomma, Ultima")
    for a in ax:
        a.axvspan(pd.Timestamp(COVID[0]), pd.Timestamp(COVID[1]), color=WARM, alpha=0.10)
        a.xaxis.set_major_locator(mdates.YearLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Higher-order Greeks of a 1-month 25Δ SPY put, 2010–2021  (COVID shaded)",
                 fontweight="bold")
    fig.savefig(out / "fig2_greek_timeseries.png"); plt.close(fig)


def fig_pnl(att, out):
    quiet = att[att["atm_iv"] < att["atm_iv"].median()]
    stress = att[att["atm_iv"] >= att["atm_iv"].median()]
    regimes = {"all": att, "quiet": quiet, "stress": stress,
               "COVID": att[(att["date"] >= COVID[0]) & (att["date"] <= COVID[1])]}
    r1 = {k: expl_var(v["dV"], v["first"]) for k, v in regimes.items()}
    r2 = {k: expl_var(v["dV"], v["first"] + v["second"]) for k, v in regimes.items()}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.arange(len(regimes)); w = 0.38
    ax[0].bar(x - w / 2, [r1[k] for k in regimes], w, color=NEUT, label="1st-order only")
    ax[0].bar(x + w / 2, [r2[k] for k in regimes], w, color=GOOD, label="1st + 2nd order")
    ax[0].set_xticks(x); ax[0].set_xticklabels(list(regimes))
    ax[0].set_ylabel("fraction of daily P&L variance explained")
    ax[0].set_title("(a) Option P&L attribution by regime"); ax[0].legend()
    ax[0].set_ylim(min(0, min(r1.values())) - 0.05, 1.02)

    s = stress.copy()
    ax[1].scatter(s["first"], s["dV"] - s["first"], s=7, alpha=0.3, color=NEUT, label="1st-order residual")
    ax[1].scatter(s["first"], s["dV"] - s["first"] - s["second"], s=7, alpha=0.4, color=GOOD,
                  label="after 2nd-order")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("1st-order predicted ΔV"); ax[1].set_ylabel("unexplained ΔV")
    ax[1].set_title("(b) Residual P&L, stress days"); ax[1].legend()
    fig.suptitle("Where first-order Greeks fail: second-order P&L attribution — SPY",
                 fontweight="bold")
    fig.savefig(out / "fig3_pnl_attribution.png"); plt.close(fig)
    return r1, r2


def fig_corr_pca(daily, out):
    G = daily[[f"g_{k}" for k in ALLG]].dropna()
    Z = (G - G.mean()) / G.std(ddof=0)
    C = Z.corr()
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
    im = ax[0].imshow(C.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(ALLG))); ax[0].set_xticklabels(ALLG, rotation=55, ha="right", fontsize=8)
    ax[0].set_yticks(range(len(ALLG))); ax[0].set_yticklabels(ALLG, fontsize=8)
    for i in range(len(ALLG)):
        for j in range(len(ALLG)):
            ax[0].text(j, i, f"{C.values[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                       color="white" if abs(C.values[i, j]) > 0.6 else "black")
    ax[0].set_title("(a) Cross-Greek correlation"); ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)

    # PCA
    cov = np.cov(Z.values.T)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]; w = w[order]; V = V[:, order]
    evr = w / w.sum()
    ax[1].bar(range(1, len(evr) + 1), np.cumsum(evr), color=ACCENT, alpha=0.85)
    ax[1].axhline(0.9, color=WARM, ls="--", lw=1, label="90%")
    ax[1].set_xlabel("principal component"); ax[1].set_ylabel("cumulative variance explained")
    ax[1].set_title("(b) PCA of the Greek vector"); ax[1].legend(loc="lower right")
    ax[1].set_ylim(0, 1.02)
    fig.suptitle("Cross-Greek relationships — redundancy and principal components",
                 fontweight="bold")
    fig.savefig(out / "fig4_correlation_pca.png"); plt.close(fig)
    return C, evr, V, ALLG


def fig_regimes(daily, out):
    d = daily.copy()
    d["regime"] = "normal"
    d.loc[d["atm_iv"] < 0.13, "regime"] = "low-vol"
    d.loc[(d["date"] >= COVID[0]) & (d["date"] <= COVID[1]), "regime"] = "COVID"
    d.loc[(d["date"] >= "2020-05-01") & (d["date"] <= "2020-12-31"), "regime"] = "recovery"
    order = ["low-vol", "normal", "recovery", "COVID"]
    show = ["gamma", "vega", "vanna", "vomma", "charm", "color"]
    means = {r: {k: d[d["regime"] == r][f"g_{k}"].abs().mean() for k in show} for r in order}
    # normalize each Greek by its low-vol level for comparability
    base = {k: means["low-vol"][k] for k in show}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(show)); w = 0.2
    for i, r in enumerate(order):
        ax.bar(x + (i - 1.5) * w, [means[r][k] / base[k] for k in show], w,
               label=r, color=[NEUT, ACCENT, GOOD, WARM][i])
    ax.set_xticks(x); ax.set_xticklabels([s.capitalize() for s in show])
    ax.set_ylabel("mean |Greek| ÷ low-vol level")
    ax.set_title("Greek magnitude by market regime (normalized to low-vol)")
    ax.legend(ncol=4, loc="upper left")
    fig.savefig(out / "fig5_regime_comparison.png"); plt.close(fig)
    return means


def fig_tests(daily, att, out):
    # Vomma(t) vs future |Δ ATM IV| over 10 days
    d = daily.copy()
    d["fwd_absdiv"] = d["atm_iv"].shift(-10).sub(d["atm_iv"]).abs()
    v = d.dropna(subset=["fwd_absdiv", "g_vomma"])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].scatter(v["g_vomma"], v["fwd_absdiv"], s=6, alpha=0.25, color=ACCENT, linewidths=0)
    b = np.polyfit(v["g_vomma"], v["fwd_absdiv"], 1)
    xs = np.linspace(v["g_vomma"].min(), v["g_vomma"].max(), 40)
    ax[0].plot(xs, np.polyval(b, xs), color=WARM, lw=1.6)
    rho = v["g_vomma"].corr(v["fwd_absdiv"])
    ax[0].set_title(f"(a) Vomma → future |ΔIV| (10d)   ρ={rho:.2f}")
    ax[0].set_xlabel("Vomma"); ax[0].set_ylabel("|ΔATM IV| over next 10d")

    a = att.dropna(subset=["hedge_err", "gamma_term", "vanna_term"])
    ax[1].scatter(a["gamma_term"].abs() + a["vanna_term"].abs(), a["hedge_err"].abs(),
                  s=6, alpha=0.25, color=GOOD, linewidths=0)
    rho2 = (a["gamma_term"].abs() + a["vanna_term"].abs()).corr(a["hedge_err"].abs())
    ax[1].set_title(f"(b) |Γ term|+|Vanna term| → |delta-hedge error|   ρ={rho2:.2f}")
    ax[1].set_xlabel("|Γ·ΔS²/2| + |Vanna·ΔS·Δσ|"); ax[1].set_ylabel("|delta-hedge P&L error|")
    fig.suptitle("Empirical tests of higher-order Greek information — SPY", fontweight="bold")
    fig.savefig(out / "fig6_empirical_tests.png"); plt.close(fig)
    return float(rho), float(rho2)


# --------------------------------------------------------------------------
def main():
    master = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/generated/research_m1")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/generated/research_m4")
    figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading smile panel"); days = load_days(master)
    print("[2/6] daily Greek panel"); daily = build_daily(days)
    print("[3/6] P&L attribution"); att = pnl_attribution(days, daily)

    print("[4/6] figures")
    fig_profiles(days, daily, figdir)
    fig_timeseries(daily, figdir)
    r1, r2 = fig_pnl(att, figdir)
    C, evr, V, gnames = fig_corr_pca(daily, figdir)
    means = fig_regimes(daily, figdir)
    rho_vomma, rho_vanna = fig_tests(daily, att, figdir)

    print("[5/6] exports")
    daily.to_csv(out / "m4_daily_greeks.csv", index=False)
    att.to_csv(out / "m4_pnl_attribution.csv", index=False)
    C.to_csv(out / "m4_greek_correlation.csv")

    # March 2020 extremes (z-score of each Greek at its COVID peak)
    covid = daily[(daily["date"] >= COVID[0]) & (daily["date"] <= COVID[1])]
    extremes = {}
    for k in ALLG:
        s = daily[f"g_{k}"]
        z = (covid[f"g_{k}"].abs().max() - s.abs().mean()) / (s.abs().std() + 1e-12)
        extremes[k] = float(z)

    pc1_load = dict(zip(gnames, np.round(V[:, 0], 2).tolist()))
    summary = {
        "underlying": "SPY",
        "period": {"start": str(daily["date"].min().date()), "end": str(daily["date"].max().date()),
                   "days": int(len(daily))},
        "pnl_attribution_expl_var": {"first_order": r1, "first_plus_second": r2},
        "pca_cumvar": np.round(np.cumsum(evr), 3).tolist(),
        "pc1_loadings": pc1_load,
        "regime_mean_abs_greek": {r: {k: float(v) for k, v in m.items()} for r, m in means.items()},
        "march2020_extreme_zscore": extremes,
        "vomma_predicts_future_absdIV_corr": rho_vomma,
        "gammavanna_explains_hedgeerr_corr": rho_vanna,
    }
    with open(out / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[6/6] summary\n" + "=" * 84)
    print(f"SPY {summary['period']['start']}..{summary['period']['end']} ({len(daily)} days)")
    print("P&L variance explained (1st | 1st+2nd):")
    for k in r1:
        print(f"  {k:>7}: {r1[k]:+.3f} | {r2[k]:+.3f}")
    print(f"PCA cumulative variance (PC1..): {summary['pca_cumvar'][:4]}")
    print(f"Vomma→future|ΔIV| corr = {rho_vomma:.3f};  (Γ+Vanna)→|hedge err| corr = {rho_vanna:.3f}")
    print("March-2020 |Greek| z-scores:", {k: round(v, 1) for k, v in extremes.items()})
    print("figures ->", figdir)


if __name__ == "__main__":
    main()
