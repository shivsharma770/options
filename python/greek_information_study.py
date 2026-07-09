#!/usr/bin/env python3
"""
greek_information_study.py -- Research Milestone 5.

Do higher-order Greeks carry economically meaningful information beyond the
standard Greeks? Seven empirical studies on 12 years of SPY option surfaces,
reusing the validated Greeks in `higher_order_greeks.py`, the per-day 1-month
smile panel (`m4_smile30.csv`), the M1 spot/term-structure masters, and the
HAC-OLS estimator -- no new pricing model, no regenerated data.

  Study 1  volatility forecasting: ATM-IV vs ATM-IV + each higher-order Greek,
           OOS R²/RMSE/MAE and Diebold-Mariano tests.
  Study 2  dealer-flow hypothesis: do aggregate Gamma/Vanna/Charm predict
           next-day/week returns and RV?
  Study 3  regime changes: does extreme Vanna/Vomma precede vol spikes / IV moves?
  Study 4  hedging-error attribution: which Greek explains the delta-hedge residual?
  Study 5  cross-sectional option returns: decile sorts on Vanna/Vomma/Charm/Speed.
  Study 6  correlation structure: PCA, hierarchical clustering, VIF.
  Study 7  importance ranking: OLS/LASSO/ElasticNet/RandomForest(+permutation/SHAP).

Usage:  greek_information_study.py [master_dir] [out_dir]
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import higher_order_greeks as hg
from iv_rv_study import ols_hac

TD = 252
HORIZONS = [5, 10, 20, 60]
HO = ["vanna", "vomma", "charm", "veta", "speed", "zomma", "color", "ultima"]
ALLG = ["delta", "gamma", "vega", "theta", "rho"] + HO
REP_M = -0.05
COVID = ("2020-02-20", "2020-04-30")

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.grid": True, "grid.alpha": 0.30,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
ACCENT, WARM, GOOD, NEUT = "#1f6feb", "#d1495b", "#2a9d8f", "#8d99ae"


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------
def load_days(master):
    df = pd.read_csv(master / "m4_smile30.csv", parse_dates=["date"])
    df = df[(df["implied_volatility"] > 0) & (df["time_to_expiry"] > 0)]
    return {d: g.sort_values("moneyness") for d, g in df.groupby("date")}


def build_panel(days):
    recs = []
    for d, g in days.items():
        S = float(g["spot"].iloc[0]); T = float(g["time_to_expiry"].median())
        i0 = g["moneyness"].abs().values.argmin()
        atm_iv = float(g.iloc[i0]["implied_volatility"])
        ir = (g["moneyness"] - REP_M).abs().values.argmin()
        rr = g.iloc[ir]
        gk = hg.greeks(rr["spot"], rr["strike"], rr["time_to_expiry"],
                       rr["implied_volatility"], 0, 0, True)
        # aggregate (equal-weighted across listed strikes) as a dealer-book proxy
        agg = hg.greeks(g["spot"].values, g["strike"].values, g["time_to_expiry"].values,
                        g["implied_volatility"].values, 0, 0, True)
        rec = {"date": d, "spot": S, "T": T, "atm_iv": atm_iv, "atm_K": float(g.iloc[i0]["strike"]),
               "agg_gamma": float(np.nansum(agg["gamma"])),
               "agg_vanna": float(np.nansum(agg["vanna"])),
               "agg_charm": float(np.nansum(agg["charm"]))}
        rec.update({f"g_{k}": float(gk[k]) for k in ALLG})
        recs.append(rec)
    p = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
    p["ret"] = np.log(p["spot"]).diff()
    p["d_iv"] = p["atm_iv"].diff()
    r = p["ret"].values; n = len(p); iv = p["atm_iv"].values
    for h in HORIZONS:
        fwd = np.full(n, np.nan); fret = np.full(n, np.nan); fdiv = np.full(n, np.nan)
        for i in range(n):
            if i + h < n:
                seg = r[i + 1:i + 1 + h]
                if not np.isnan(seg).any():
                    fwd[i] = np.sqrt(TD / h * np.sum(seg ** 2))
                fret[i] = np.log(p["spot"].iloc[i + h] / p["spot"].iloc[i])
                fdiv[i] = iv[i + h] - iv[i]
        p[f"rv_{h}"] = fwd; p[f"ret_{h}"] = fret; p[f"div_{h}"] = fdiv
    return p


# --------------------------------------------------------------------------
# OOS + Diebold-Mariano
# --------------------------------------------------------------------------
def oos_forecast(X, y, n0=504, h=20):
    n = len(y); preds = np.full(n, np.nan)
    for i in range(n0, n):
        te = i - h
        if te < 252:
            continue
        b = np.linalg.lstsq(X[:te + 1], y[:te + 1], rcond=None)[0]
        preds[i] = X[i] @ b
    return preds


def r2_oos(y, preds, ybench):
    m = ~np.isnan(preds) & ~np.isnan(y)
    return 1 - np.sum((y[m] - preds[m]) ** 2) / np.sum((y[m] - ybench[m]) ** 2)


def dm_test(y, p1, p2, h):
    """Diebold-Mariano (squared-error loss), HAC var with lag h-1."""
    m = ~np.isnan(p1) & ~np.isnan(p2) & ~np.isnan(y)
    d = (y[m] - p1[m]) ** 2 - (y[m] - p2[m]) ** 2
    d = d - d.mean() + d.mean()
    n = len(d); dbar = d.mean(); dc = d - dbar
    var = np.sum(dc ** 2)
    for l in range(1, h):
        w = 1 - l / h
        var += 2 * w * np.sum(dc[l:] * dc[:-l])
    var /= n
    dmstat = dbar / np.sqrt(var / n)
    return float(dmstat), float(2 * stats.norm.sf(abs(dmstat)))


# --------------------------------------------------------------------------
def study1_vol_forecast(p, figdir):
    res = {}
    dm = {}
    for h in HORIZONS:
        sub = p[["atm_iv"] + [f"g_{k}" for k in HO] + [f"rv_{h}"]].dropna().reset_index(drop=True)
        y = sub[f"rv_{h}"].values; n = len(sub); ones = np.ones(n)
        ybench = np.full(n, np.nan)
        for i in range(504, n):
            te = i - h
            if te >= 252:
                ybench[i] = y[:te + 1].mean()
        models = {"ATM": np.column_stack([ones, sub["atm_iv"]])}
        for g in ["vanna", "vomma", "charm"]:
            models[f"ATM+{g}"] = np.column_stack([ones, sub["atm_iv"], sub[f"g_{g}"]])
        models["ATM+all HO"] = np.column_stack([ones, sub["atm_iv"]] + [sub[f"g_{k}"] for k in HO])
        preds = {m: oos_forecast(X, y, h=h) for m, X in models.items()}
        res[h] = {m: {"oos_r2": r2_oos(y, pr, ybench),
                      "rmse": float(np.sqrt(np.nanmean((y - pr) ** 2))),
                      "mae": float(np.nanmean(np.abs(y - pr)))}
                  for m, pr in preds.items()}
        dm[h] = {m: dm_test(y, preds["ATM"], preds[m], h) for m in models if m != "ATM"}

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    models_l = ["ATM", "ATM+vanna", "ATM+vomma", "ATM+charm", "ATM+all HO"]
    x = np.arange(len(HORIZONS)); w = 0.15
    cols = [NEUT, ACCENT, GOOD, WARM, "#8250df"]
    for k, m in enumerate(models_l):
        ax[0].bar(x + (k - 2) * w, [res[h][m]["oos_r2"] for h in HORIZONS], w, label=m, color=cols[k])
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{h}d" for h in HORIZONS])
    ax[0].set_title("(a) OOS R² of RV forecast"); ax[0].set_ylabel("OOS R²"); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("horizon")
    # DM stats (ATM+all vs ATM)
    dms = [dm[h]["ATM+all HO"][0] for h in HORIZONS]
    ax[1].bar([f"{h}d" for h in HORIZONS], dms, color=[GOOD if abs(v) > 1.96 else NEUT for v in dms])
    ax[1].axhline(1.96, color=WARM, ls="--", lw=1); ax[1].axhline(-1.96, color=WARM, ls="--", lw=1)
    ax[1].set_title("(b) Diebold–Mariano: ATM+all HO vs ATM\n(>0 ⇒ HO better; |·|>1.96 significant)")
    ax[1].set_ylabel("DM statistic")
    fig.suptitle("Study 1 — do higher-order Greeks improve RV forecasts beyond ATM IV?",
                 fontweight="bold")
    fig.savefig(figdir / "fig1_study1_vol_forecast.png"); plt.close(fig)
    return res, dm


def study2_dealer_flow(p, figdir):
    out = {}
    feats = ["agg_gamma", "agg_vanna", "agg_charm"]
    z = p.copy()
    for f in feats:
        z[f] = (z[f] - z[f].mean()) / z[f].std()
    rows = []
    for tgt in ["ret_5", "ret_10", "rv_20"]:
        for f in feats:
            sub = z[[f, tgt]].dropna()
            X = np.column_stack([np.ones(len(sub)), sub[f]])
            fit = ols_hac(sub[tgt].values, X, 10)
            rows.append({"target": tgt, "feature": f, "beta": float(fit["beta"][1]),
                         "t": float(fit["tstat"][1]), "r2": float(fit["r2"])})
    out["regressions"] = rows
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax[0].plot(p["date"], (p["agg_gamma"] - p["agg_gamma"].mean()) / p["agg_gamma"].std(),
               color=ACCENT, lw=0.7, label="agg Gamma (z)")
    ax[0].plot(p["date"], (p["agg_vanna"] - p["agg_vanna"].mean()) / p["agg_vanna"].std(),
               color=WARM, lw=0.7, alpha=0.8, label="agg Vanna (z)")
    ax[0].set_title("(a) Aggregate market Greek exposure (equal-weighted proxy)")
    ax[0].legend(loc="upper left"); ax[0].xaxis.set_major_locator(mdates.YearLocator())
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    tbl = pd.DataFrame(rows)
    piv = tbl.pivot(index="feature", columns="target", values="t")
    im = ax[1].imshow(piv.values, cmap="coolwarm", vmin=-3, vmax=3, aspect="auto")
    ax[1].set_xticks(range(len(piv.columns))); ax[1].set_xticklabels(piv.columns)
    ax[1].set_yticks(range(len(piv.index))); ax[1].set_yticklabels(piv.index)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax[1].text(j, i, f"{piv.values[i, j]:.1f}", ha="center", va="center", fontsize=9)
    ax[1].set_title("(b) HAC t-stats: aggregate Greek → future return / RV"); ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)
    fig.suptitle("Study 2 — dealer-flow hypothesis (SPY, unsigned exposure proxy)", fontweight="bold")
    fig.savefig(figdir / "fig2_study2_dealer_flow.png"); plt.close(fig)
    return out


def study3_regime(p, figdir):
    # extreme Greek (top-decile |·|) precedes future vol move?
    out = {}
    for g, tgt in [("g_vanna", "rv_20"), ("g_vomma", "div_20")]:
        z = p[[g, tgt, "date", "atm_iv"]].dropna().copy()
        thr = z[g].abs().quantile(0.9)
        hi = z[z[g].abs() >= thr][tgt].abs()
        lo = z[z[g].abs() < thr][tgt].abs()
        t, pv = stats.ttest_ind(hi, lo, equal_var=False)
        out[g] = {"target": tgt, "hi_mean": float(hi.mean()), "lo_mean": float(lo.mean()),
                  "t": float(t), "p": float(pv), "corr": float(z[g].corr(z[tgt].abs()))}
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    ax[0].plot(p["date"], p["atm_iv"], color="k", lw=0.8)
    ax[0].set_ylabel("ATM IV"); ax[0].set_title("(a) ATM implied volatility")
    ax[1].plot(p["date"], (p["g_vomma"] - p["g_vomma"].mean()) / p["g_vomma"].std(),
               color=WARM, lw=0.7, label="Vomma (z)")
    ax[1].plot(p["date"], (p["g_vanna"] - p["g_vanna"].mean()) / p["g_vanna"].std(),
               color=ACCENT, lw=0.7, alpha=0.8, label="Vanna (z)")
    ax[1].legend(loc="upper left"); ax[1].set_title("(b) Higher-order Greeks (z-scored)")
    for a in ax:
        a.axvspan(pd.Timestamp(COVID[0]), pd.Timestamp(COVID[1]), color=WARM, alpha=0.10)
        a.xaxis.set_major_locator(mdates.YearLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Study 3 — do extreme higher-order Greeks precede volatility events?",
                 fontweight="bold")
    fig.savefig(figdir / "fig3_study3_regime.png"); plt.close(fig)
    return out


def study4_hedge_attrib(days, p, figdir):
    dts = p["date"].tolist()
    interp = {d: (lambda gg: (lambda x: float(np.interp(x, gg["moneyness"].values,
              gg["implied_volatility"].values))))(g) for d, g in days.items()}
    rows = []
    for i in range(len(dts) - 1):
        r = p.iloc[i]; S, K, T, sig = r["spot"], r["atm_K"], r["T"], r["atm_iv"]
        Sn = p.iloc[i + 1]["spot"]; Tn = max(T - 1 / TD, 1e-4); mn = np.log(K / Sn)
        sign = interp[dts[i + 1]](mn)
        dV = hg.bs_price(Sn, K, Tn, sign, 0, 0, True) - hg.bs_price(S, K, T, sig, 0, 0, True)
        gk = hg.greeks(S, K, T, sig, 0, 0, True)
        dS, dsig = Sn - S, sign - sig
        resid = dV - gk["delta"] * dS - gk["theta"] / TD - gk["vega"] * dsig  # after 1st order
        rows.append({"resid": resid,
                     "gamma": 0.5 * gk["gamma"] * dS ** 2,
                     "vanna": gk["vanna"] * dS * dsig,
                     "vomma": 0.5 * gk["vomma"] * dsig ** 2,
                     "charm": gk["charm"] * dS / TD,
                     "speed": gk["speed"] * dS ** 3 / 6,
                     "color": 0.5 * gk["color"] * dS ** 2 / TD})
    a = pd.DataFrame(rows).dropna()
    terms = ["gamma", "vanna", "vomma", "charm", "speed", "color"]
    # variance decomposition: since resid ~= sum(terms), each term's share of the
    # residual variance is cov(term, resid)/var(resid) (shares sum to ~1). This
    # reflects magnitude, unlike squared correlation which is inflated for terms
    # that merely share a factor (e.g. Color and Gamma are both proportional to dS^2).
    vr = float(np.var(a["resid"]))
    r2 = {t: float(np.cov(a[t], a["resid"])[0, 1] / vr) for t in terms}
    # full 2nd-order model
    X = np.column_stack([np.ones(len(a))] + [a[t] for t in terms])
    full = ols_hac(a["resid"].values, X, 5)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    order = sorted(terms, key=lambda t: r2[t])
    ax.barh(order, [r2[t] for t in order], color=[GOOD if t in ("gamma", "vanna") else NEUT for t in order])
    ax.set_title(f"Study 4 — share of 1st-order-residual P&L variance explained per term\n"
                 f"(full 2nd-order model R² = {full['r2']:.3f})")
    ax.set_xlabel("share of delta/theta/vega residual variance (covariance decomposition)")
    fig.savefig(figdir / "fig4_study4_hedge_attribution.png"); plt.close(fig)
    return {"univariate_r2": r2, "full_r2": float(full["r2"]),
            "term_t": {terms[k]: float(full["tstat"][k + 1]) for k in range(len(terms))}}


def study5_xsection(days, figdir):
    rows = []
    dlist = sorted(days)
    dmap = {d: i for i, d in enumerate(dlist)}
    for d in dlist[:-5]:
        g = days[d]; S = g["spot"].iloc[0]
        dn = dlist[dmap[d] + 5]
        gn = days[dn]; Sn = gn["spot"].iloc[0]
        interp = lambda x: float(np.interp(x, gn["moneyness"].values, gn["implied_volatility"].values))
        for _, row in g.iterrows():
            K, T, iv = row["strike"], row["time_to_expiry"], row["implied_volatility"]
            V0 = hg.bs_price(S, K, T, iv, 0, 0, True)
            if V0 < 0.05:
                continue
            Tn = max(T - 5 / TD, 1e-4); mn = np.log(K / Sn); ivn = interp(mn)
            Vn = hg.bs_price(Sn, K, Tn, ivn, 0, 0, True)
            gk = hg.greeks(S, K, T, iv, 0, 0, True)
            rows.append({"date": d, "ret": Vn / V0 - 1, "m": row["moneyness"],
                         "vanna": gk["vanna"], "vomma": gk["vomma"],
                         "charm": gk["charm"], "speed": gk["speed"]})
    df = pd.DataFrame(rows)
    out = {}
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for k, gname in enumerate(["vanna", "vomma", "charm", "speed"]):
        df["dec"] = df.groupby("date")[gname].transform(
            lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
        dm = df.groupby("dec")["ret"].mean() * 100
        mm = df.groupby("dec")["m"].mean()
        ls = float(dm.iloc[-1] - dm.iloc[0])
        # t-stat of the long-short across days
        daily_ls = df[df["dec"].isin([1, 10])].groupby(["date", "dec"])["ret"].mean().unstack()
        d_ls = (daily_ls[10] - daily_ls[1]).dropna()
        tstat = float(d_ls.mean() / (d_ls.std() / np.sqrt(len(d_ls))))
        out[gname] = {"long_short_pct": ls, "t": tstat, "d1_moneyness": float(mm.iloc[0]),
                      "d10_moneyness": float(mm.iloc[-1])}
        ax[k].bar(dm.index, dm.values, color=ACCENT)
        ax[k].set_title(f"{gname}\nD10−D1={ls:.1f}%  t={tstat:.1f}")
        ax[k].set_xlabel("Greek decile")
    ax[0].set_ylabel("mean 5-day option return (%)")
    fig.suptitle("Study 5 — cross-sectional 5-day option returns by higher-order Greek decile",
                 fontweight="bold")
    fig.savefig(figdir / "fig5_study5_xsection.png"); plt.close(fig)
    return out


def study6_structure(p, figdir):
    G = p[[f"g_{k}" for k in ALLG]].dropna()
    G.columns = ALLG
    Z = (G - G.mean()) / G.std()
    C = Z.corr()
    # VIF
    vif = {}
    for j, f in enumerate(ALLG):
        y = Z[f].values; Xo = np.column_stack([np.ones(len(Z))] + [Z[c].values for c in ALLG if c != f])
        b = np.linalg.lstsq(Xo, y, rcond=None)[0]; e = y - Xo @ b
        r2 = 1 - (e @ e) / np.sum((y - y.mean()) ** 2)
        vif[f] = float(1 / max(1e-9, 1 - r2))
    cov = np.cov(Z.values.T); w, V = np.linalg.eigh(cov)
    o = np.argsort(w)[::-1]; evr = (w[o] / w.sum())
    Lk = linkage(Z.values.T, method="average", metric="correlation")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    im = ax[0].imshow(C.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(ALLG))); ax[0].set_xticklabels(ALLG, rotation=60, ha="right", fontsize=7)
    ax[0].set_yticks(range(len(ALLG))); ax[0].set_yticklabels(ALLG, fontsize=7)
    ax[0].set_title("(a) Correlation"); ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)
    ax[1].bar(range(1, len(evr) + 1), np.cumsum(evr), color=ACCENT)
    ax[1].axhline(0.9, color=WARM, ls="--", lw=1)
    ax[1].set_title("(b) PCA cumulative variance"); ax[1].set_xlabel("component"); ax[1].set_ylim(0, 1.02)
    dendrogram(Lk, labels=ALLG, ax=ax[2], leaf_font_size=7, color_threshold=0.5)
    ax[2].set_title("(c) Hierarchical clustering (1−|corr|)")
    fig.suptitle("Study 6 — Greek correlation structure: PCA, clustering, VIF", fontweight="bold")
    fig.savefig(figdir / "fig6_study6_structure.png"); plt.close(fig)
    return {"vif": vif, "pca_cumvar": np.round(np.cumsum(evr), 3).tolist(),
            "corr": C.round(2).to_dict()}


def study7_importance(p, figdir):
    feats = HO + ["g_delta", "g_gamma", "g_vega", "g_theta"]
    feats = [f if f.startswith("g_") else f"g_{f}" for f in feats]
    names = [f.replace("g_", "") for f in feats]
    targets = {"rv_20": "future RV", "div_20": "|ΔIV|", "ret_10": "return"}
    imp = {t: {} for t in targets}
    for tgt in targets:
        sub = p[feats + [tgt]].dropna()
        y = sub[tgt].values.copy()
        if tgt == "div_20":
            y = np.abs(y)
        X = StandardScaler().fit_transform(sub[feats].values)
        lasso = LassoCV(cv=5, max_iter=5000).fit(X, y)
        enet = ElasticNetCV(cv=5, max_iter=5000).fit(X, y)
        rf = RandomForestRegressor(n_estimators=200, max_depth=6, n_jobs=-1, random_state=0).fit(X, y)
        pi = permutation_importance(rf, X, y, n_repeats=10, random_state=0, n_jobs=-1)
        imp[tgt] = {"lasso": np.abs(lasso.coef_), "enet": np.abs(enet.coef_),
                    "rf": rf.feature_importances_, "perm": pi.importances_mean}
    # aggregate rank across models & targets
    ranks = np.zeros(len(feats))
    for tgt in targets:
        for mdl in ["lasso", "enet", "rf", "perm"]:
            v = imp[tgt][mdl]
            ranks += stats.rankdata(-np.abs(v))
    order = np.argsort(ranks)
    fig, ax = plt.subplots(1, len(targets), figsize=(15, 4.4), sharey=True)
    for k, tgt in enumerate(targets):
        rfimp = imp[tgt]["perm"]
        idx = np.argsort(rfimp)
        ax[k].barh(np.array(names)[idx], rfimp[idx],
                   color=[GOOD if names[i] in ("vanna", "vomma", "charm") else NEUT for i in idx])
        ax[k].set_title(f"{targets[tgt]}\n(RF permutation importance)")
    fig.suptitle("Study 7 — Greek importance ranking (LASSO/ElasticNet/RandomForest)",
                 fontweight="bold")
    fig.savefig(figdir / "fig7_study7_importance.png"); plt.close(fig)
    ranking = [names[i] for i in order]
    return {"overall_ranking": ranking,
            "perm_importance": {t: dict(zip(names, np.round(imp[t]["perm"], 5).tolist())) for t in targets}}


# --------------------------------------------------------------------------
def main():
    master = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/generated/research_m1")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/generated/research_m5")
    figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)

    print("loading + panel"); days = load_days(master); p = build_panel(days)
    p.to_csv(out / "m5_panel.csv", index=False)

    print("Study 1"); s1, dm = study1_vol_forecast(p, figdir)
    print("Study 2"); s2 = study2_dealer_flow(p, figdir)
    print("Study 3"); s3 = study3_regime(p, figdir)
    print("Study 4"); s4 = study4_hedge_attrib(days, p, figdir)
    print("Study 5"); s5 = study5_xsection(days, figdir)
    print("Study 6"); s6 = study6_structure(p, figdir)
    print("Study 7"); s7 = study7_importance(p, figdir)

    summary = {"underlying": "SPY",
               "period": {"start": str(p["date"].min().date()), "end": str(p["date"].max().date()),
                          "days": int(len(p))},
               "study1_vol_forecast": {str(h): s1[h] for h in HORIZONS},
               "study1_DM": {str(h): dm[h] for h in HORIZONS},
               "study2_dealer_flow": s2, "study3_regime": s3, "study4_hedge": s4,
               "study5_xsection": s5, "study6_structure": {k: v for k, v in s6.items() if k != "corr"},
               "study7_importance": s7}
    with open(out / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 88)
    print("STUDY 1 — OOS R² (ATM vs ATM+all HO) + DM:")
    for h in HORIZONS:
        print(f"  {h:>3}d: ATM={s1[h]['ATM']['oos_r2']:.3f}  ATM+HO={s1[h]['ATM+all HO']['oos_r2']:.3f}  "
              f"DM={dm[h]['ATM+all HO'][0]:+.2f} (p={dm[h]['ATM+all HO'][1]:.2f})")
    print("STUDY 4 — 1st-order-residual variance explained:", {k: round(v, 3) for k, v in s4["univariate_r2"].items()})
    print("STUDY 5 — decile long-short (D10-D1):", {k: (round(v['long_short_pct'], 1), round(v['t'], 1)) for k, v in s5.items()})
    print("STUDY 6 — PCA cumvar:", s6["pca_cumvar"][:4], " VIF:", {k: round(v, 1) for k, v in s6["vif"].items()})
    print("STUDY 7 — overall Greek ranking:", s7["overall_ranking"])
    print("figures ->", figdir)


if __name__ == "__main__":
    main()
