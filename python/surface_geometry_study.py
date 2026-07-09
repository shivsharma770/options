#!/usr/bin/env python3
"""
surface_geometry_study.py -- Research Milestone 6.

Treats the implied-volatility surface as a geometric object: for every daily
calibrated surface it computes geometric descriptors (smile/term-structure slope
and curvature, skew asymmetry, butterfly intensity, 1-D and 2-D roughness, ridge
location and migration) and four novel aggregate indices (Surface Stress,
Surface Stability, Curvature Concentration, Smile Asymmetry). It then tests
whether these carry incremental predictive information beyond ATM IV + historical
vol for future RV, the volatility risk premium, skew changes, and large-vol
events -- via incremental HAC regressions, likelihood-ratio tests, and
expanding-window out-of-sample forecasts.

Reuses the M6 surface panel (`m6_surface.csv`), the M1 spot master, and the
HAC-OLS estimator. No pricing model, calibration, or interpolation is modified;
descriptors are computed by local least-squares fits of the calibrated points.

Usage:  surface_geometry_study.py [master_dir] [out_dir]
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

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from iv_rv_study import ols_hac

TD = 252
HORIZONS = [5, 10, 20, 60]
BUCKETS = [14, 30, 60, 90, 180]
MG = np.linspace(-0.12, 0.12, 13)
COVID = ("2020-02-20", "2020-04-30")
GEO = ["smile_slope", "smile_curv", "ts_slope", "ts_curv", "skew_asym",
       "bf_intensity", "rough1d", "rough2d", "ridge", "SSI", "stability", "CCI", "SAI"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.grid": True, "grid.alpha": 0.30,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
ACCENT, WARM, GOOD, NEUT = "#1f6feb", "#d1495b", "#2a9d8f", "#8d99ae"


# --------------------------------------------------------------------------
def load_surface(master):
    df = pd.read_csv(master / "m6_surface.csv", parse_dates=["date"])
    df = df[(df["implied_volatility"] > 0) & df["moneyness"].abs().le(0.13)]
    return {d: g for d, g in df.groupby("date")}


def fit_smile(sm):
    """Quadratic IV = a + b m + c m^2; returns (a, b, 2c, raw roughness)."""
    m, v = sm["moneyness"].values, sm["implied_volatility"].values
    if len(m) < 5:
        return None
    c, b, a = np.polyfit(m, v, 2)
    order = np.argsort(m)
    vs = v[order]
    rough = float(np.sum(np.diff(vs, 2) ** 2)) if len(vs) >= 3 else np.nan
    return a, b, 2 * c, rough


def build_geometry(surfaces):
    recs = []
    grids = {}
    for d, g in surfaces.items():
        S = float(g["spot"].iloc[0])
        fits = {}
        atmiv = {}
        for b in BUCKETS:
            sm = g[g["mat_bucket"] == b]
            f = fit_smile(sm)
            if f is not None:
                fits[b] = f
                atmiv[b] = f[0]
        if 30 not in fits or len(atmiv) < 3:
            continue
        a30, b30, cur30, rough30 = fits[30]
        # term structure: ATM IV vs sqrt(maturity-years)
        bs = sorted(atmiv)
        x = np.sqrt(np.array(bs) / 365.0); yv = np.array([atmiv[b] for b in bs])
        ts_slope = float(np.polyfit(x, yv, 1)[0])
        ts_curv = float(np.polyfit(x, yv, 2)[0]) if len(bs) >= 3 else np.nan
        # skew asymmetry & butterfly at 30d (from fitted smile)
        iv_m10 = a30 + b30 * (-0.10) + 0.5 * cur30 * 0.01
        iv_p10 = a30 + b30 * (0.10) + 0.5 * cur30 * 0.01
        skew_asym = float(iv_m10 - iv_p10)
        bf = float(0.5 * (iv_m10 + iv_p10) - a30)
        # ridge = moneyness of smile minimum at 30d (trough of the vol smile)
        ridge = float(np.clip(-b30 / (cur30 + 1e-9), -0.12, 0.12))
        # standardized grid over present buckets -> 2-D roughness
        G = np.full((len(BUCKETS), len(MG)), np.nan)
        for i, b in enumerate(BUCKETS):
            if b in fits:
                aa, bb, cc, _ = fits[b]
                G[i] = aa + bb * MG + 0.5 * cc * MG ** 2
        rough2d = float(np.nanmean(np.abs(np.diff(G, 2, axis=1)))) if np.isfinite(G).any() else np.nan
        # curvature concentration: ATM curvature vs mean |curvature| across buckets
        curvs = [fits[b][2] for b in fits]
        cci = float(abs(cur30) / (np.mean(np.abs(curvs)) + 1e-9))
        recs.append({"date": d, "spot": S, "atm_iv": a30,
                     "smile_slope": b30, "smile_curv": cur30, "ts_slope": ts_slope,
                     "ts_curv": ts_curv, "skew_asym": skew_asym, "bf_intensity": bf,
                     "rough1d": rough30, "rough2d": rough2d, "ridge": ridge, "CCI": cci})
        grids[d] = G
    p = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
    # Surface Stability Index: day-to-day Frobenius distance of the grid
    dts = p["date"].tolist()
    stab = [np.nan]
    for i in range(1, len(dts)):
        A, B = grids.get(dts[i]), grids.get(dts[i - 1])
        if A is not None and B is not None:
            m = np.isfinite(A) & np.isfinite(B)
            stab.append(float(np.sqrt(np.nanmean((A[m] - B[m]) ** 2))) if m.any() else np.nan)
        else:
            stab.append(np.nan)
    p["stability"] = stab
    p["ridge_migration"] = p["ridge"].diff().abs()
    # Surface Stress Index: z-sum of |slope| + |curv| + roughness + |ts_slope|
    def z(s):
        return (s - s.mean()) / s.std()
    p["SSI"] = z(p["smile_slope"].abs()) + z(p["smile_curv"].abs()) + z(p["rough2d"]) + z(p["ts_slope"].abs())
    p["SAI"] = z(p["skew_asym"]) + z(p["smile_slope"])
    # targets
    p["ret"] = np.log(p["spot"]).diff()
    r = p["ret"].values; n = len(p); iv = p["atm_iv"].values; sk = p["skew_asym"].values
    for h in HORIZONS:
        fwd = np.full(n, np.nan); dsk = np.full(n, np.nan); hv = np.full(n, np.nan)
        for i in range(n):
            if i + h < n:
                seg = r[i + 1:i + 1 + h]
                if not np.isnan(seg).any():
                    fwd[i] = np.sqrt(TD / h * np.sum(seg ** 2))
                dsk[i] = sk[i + h] - sk[i]
            if i - h + 1 >= 1:
                seg = r[i - h + 1:i + 1]
                if not np.isnan(seg).any():
                    hv[i] = np.sqrt(TD / h * np.sum(seg ** 2))
        p[f"rv_{h}"] = fwd; p[f"hv_{h}"] = hv; p[f"dskew_{h}"] = dsk
        p[f"vrp_{h}"] = p["atm_iv"] - p[f"rv_{h}"]
    return p, grids


# --------------------------------------------------------------------------
def lr_test(rss0, rss1, n, q):
    lr = n * np.log(rss0 / rss1)
    return float(lr), float(stats.chi2.sf(lr, q))


def oos(X, y, h, n0=504):
    n = len(y); pr = np.full(n, np.nan)
    for i in range(n0, n):
        te = i - h
        if te < 252:
            continue
        b = np.linalg.lstsq(X[:te + 1], y[:te + 1], rcond=None)[0]
        pr[i] = X[i] @ b
    return pr


def predictive(p, figdir):
    geo_block = ["SSI", "stability", "CCI", "SAI", "rough2d", "ts_curv"]
    res = {}
    for tgt_name, tcol in [("future RV", "rv_20"), ("VRP", "vrp_20"), ("Δskew", "dskew_20")]:
        cols = ["atm_iv", "hv_20"] + geo_block + [tcol]
        sub = p[cols].dropna().reset_index(drop=True)
        y = sub[tcol].values; n = len(sub); ones = np.ones(n)
        X0 = np.column_stack([ones, sub["atm_iv"], sub["hv_20"]])
        X1 = np.column_stack([ones, sub["atm_iv"], sub["hv_20"]] + [sub[c] for c in geo_block])
        f0, f1 = ols_hac(y, X0, 20), ols_hac(y, X1, 20)
        rss0 = float(f0["resid"] @ f0["resid"]); rss1 = float(f1["resid"] @ f1["resid"])
        lr, lrp = lr_test(rss0, rss1, n, len(geo_block))
        # HAC Wald on the geo block
        sel = list(range(3, 3 + len(geo_block)))
        bsel = f1["beta"][sel]; V = f1["cov"][np.ix_(sel, sel)]
        wald = float(bsel @ np.linalg.solve(V, bsel)); waldp = float(stats.chi2.sf(wald, len(geo_block)))
        # OOS
        ybench = np.full(n, np.nan)
        for i in range(504, n):
            te = i - 20
            if te >= 252:
                ybench[i] = y[:te + 1].mean()
        p0, p1 = oos(X0, y, 20), oos(X1, y, 20)
        m = ~np.isnan(p0) & ~np.isnan(ybench)
        oos0 = 1 - np.sum((y[m] - p0[m]) ** 2) / np.sum((y[m] - ybench[m]) ** 2)
        oos1 = 1 - np.sum((y[m] - p1[m]) ** 2) / np.sum((y[m] - ybench[m]) ** 2)
        res[tgt_name] = {"r2_bench": float(f0["r2"]), "r2_geo": float(f1["r2"]),
                         "adj_bench": float(f0["adj_r2"]), "adj_geo": float(f1["adj_r2"]),
                         "lr": lr, "lr_p": lrp, "wald_p": waldp,
                         "oos_bench": float(oos0), "oos_geo": float(oos1),
                         "geo_t": {geo_block[k]: float(f1["tstat"][3 + k]) for k in range(len(geo_block))}}
    # large-vol event prediction: does SSI separate top-decile RV days?
    q = p[["SSI", "rv_20"]].dropna()
    thr = q["rv_20"].quantile(0.9)
    hi = q[q["rv_20"] >= thr]["SSI"]; lo = q[q["rv_20"] < thr]["SSI"]
    tt, tp = stats.ttest_ind(hi, lo, equal_var=False)
    res["large_vol_event"] = {"SSI_hi_mean": float(hi.mean()), "SSI_lo_mean": float(lo.mean()),
                              "t": float(tt), "p": float(tp)}
    # figure: OOS bench vs geo + LR + coefficients
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    names = list(res.keys())[:3]
    x = np.arange(len(names)); w = 0.38
    ax[0].bar(x - w / 2, [res[nm]["oos_bench"] for nm in names], w, color=NEUT, label="ATM IV + HV")
    ax[0].bar(x + w / 2, [res[nm]["oos_geo"] for nm in names], w, color=GOOD, label="+ geometry")
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels(names)
    ax[0].set_title("(a) Out-of-sample R² (h=20d)"); ax[0].set_ylabel("OOS R²"); ax[0].legend()
    lrp = [res[nm]["lr_p"] for nm in names]
    ax[1].bar(names, [-np.log10(v + 1e-300) for v in lrp], color=[GOOD if v < 0.05 else NEUT for v in lrp])
    ax[1].axhline(-np.log10(0.05), color=WARM, ls="--", lw=1, label="p=0.05")
    ax[1].set_title("(b) In-sample LR test: geometry block"); ax[1].set_ylabel("−log10 p"); ax[1].legend()
    gt = res["future RV"]["geo_t"]
    ax[2].barh(list(gt.keys()), list(gt.values()),
               color=[GOOD if abs(v) > 1.96 else NEUT for v in gt.values()])
    ax[2].axvline(1.96, color=WARM, ls="--", lw=1); ax[2].axvline(-1.96, color=WARM, ls="--", lw=1)
    ax[2].set_title("(c) HAC t: geometry → future RV"); ax[2].set_xlabel("t-stat")
    fig.suptitle("Study — does surface geometry predict beyond ATM IV + HV?", fontweight="bold")
    fig.savefig(figdir / "fig5_predictive.png"); plt.close(fig)
    return res


# --------------------------------------------------------------------------
def fig_surface_evol(surfaces, p, grids, figdir):
    calm = p.loc[p["atm_iv"].idxmin(), "date"]
    cov = p[(p["date"] >= COVID[0]) & (p["date"] <= COVID[1])]
    crash = cov.loc[cov["atm_iv"].idxmax(), "date"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    vmax = max(np.nanmax(grids[pd.Timestamp(crash)]), 0.1)
    for ax, d, lab in [(axes[0], calm, "calm"), (axes[1], crash, "COVID")]:
        G = grids[pd.Timestamp(d)]
        im = ax.imshow(G, aspect="auto", origin="lower", cmap="viridis",
                       extent=[MG[0], MG[-1], 0, len(BUCKETS)], vmin=0.05, vmax=vmax)
        ax.set_yticks(np.arange(len(BUCKETS)) + 0.5); ax.set_yticklabels([f"{b}d" for b in BUCKETS])
        ax.set_xlabel("log-moneyness"); ax.set_title(f"{lab}  {pd.Timestamp(d):%Y-%m-%d}")
        ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="IV")
    fig.suptitle("Figure 1 — implied-vol surface: calm vs COVID", fontweight="bold")
    fig.savefig(figdir / "fig1_surface_evolution.png"); plt.close(fig)


def fig_curv_heatmap(p, figdir):
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    ax[0].plot(p["date"], p["atm_iv"], color="k", lw=0.8); ax[0].set_ylabel("ATM IV")
    ax[0].set_title("(a) ATM implied vol")
    ax[1].plot(p["date"], p["smile_curv"], color=ACCENT, lw=0.7, label="smile curvature (30d)")
    ax[1].plot(p["date"], p["ts_curv"], color=WARM, lw=0.7, alpha=0.8, label="term-structure curvature")
    ax[1].legend(loc="upper left"); ax[1].set_title("(b) Surface curvature over time")
    for a in ax:
        a.axvspan(pd.Timestamp(COVID[0]), pd.Timestamp(COVID[1]), color=WARM, alpha=0.10)
        a.xaxis.set_major_locator(mdates.YearLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Figure 2 — curvature evolution", fontweight="bold")
    fig.savefig(figdir / "fig2_curvature_heatmap.png"); plt.close(fig)


def fig_ridge(p, figdir):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(p["date"], p["ridge"], color=GOOD, lw=0.7)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.axvspan(pd.Timestamp(COVID[0]), pd.Timestamp(COVID[1]), color=WARM, alpha=0.10)
    ax.set_ylabel("ridge (min-IV) log-moneyness"); ax.set_title("Figure 3 — smile ridge trajectory (30d)")
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(figdir / "fig3_ridge_trajectory.png"); plt.close(fig)


def fig_pca(p, figdir):
    Z = p[GEO].dropna()
    Zs = (Z - Z.mean()) / Z.std()
    cov = np.cov(Zs.values.T); w, V = np.linalg.eigh(cov)
    o = np.argsort(w)[::-1]; w = w[o]; V = V[:, o]; evr = w / w.sum()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].bar(range(1, len(evr) + 1), np.cumsum(evr), color=ACCENT)
    ax[0].axhline(0.9, color=WARM, ls="--", lw=1); ax[0].set_ylim(0, 1.02)
    ax[0].set_title("(a) PCA of surface geometry"); ax[0].set_xlabel("component"); ax[0].set_ylabel("cum. var")
    im = ax[1].imshow(V[:, :4], cmap="coolwarm", vmin=-0.6, vmax=0.6, aspect="auto")
    ax[1].set_yticks(range(len(GEO))); ax[1].set_yticklabels(GEO, fontsize=8)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([f"PC{i+1}" for i in range(4)])
    ax[1].set_title("(b) Loadings"); ax[1].grid(False); fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)
    fig.suptitle("Figure 4 — principal components of daily surface geometry", fontweight="bold")
    fig.savefig(figdir / "fig4_geometry_pca.png"); plt.close(fig)
    return np.round(np.cumsum(evr), 3).tolist()


# --------------------------------------------------------------------------
def main():
    master = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/generated/research_m1")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/generated/research_m6")
    figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)

    print("loading surface"); surfaces = load_surface(master)
    print("geometry"); p, grids = build_geometry(surfaces)
    p.to_csv(out / "m6_geometry.csv", index=False)
    print("figures"); fig_surface_evol(surfaces, p, grids, figdir)
    fig_curv_heatmap(p, figdir); fig_ridge(p, figdir); pca = fig_pca(p, figdir)
    print("predictive"); res = predictive(p, figdir)

    summary = {"underlying": "SPY",
               "period": {"start": str(p["date"].min().date()), "end": str(p["date"].max().date()),
                          "days": int(len(p))},
               "geometry_pca_cumvar": pca, "predictive": res}
    with open(out / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 90)
    print(f"SPY {summary['period']['start']}..{summary['period']['end']} ({len(p)} days)")
    print("Surface geometry PCA cumvar:", pca[:5])
    for nm in ["future RV", "VRP", "Δskew"]:
        r = res[nm]
        print(f"  {nm:>9}: R2 bench={r['r2_bench']:.3f} +geo={r['r2_geo']:.3f}  "
              f"LR p={r['lr_p']:.1e}  Wald p={r['wald_p']:.1e}  "
              f"OOS bench={r['oos_bench']:.3f} +geo={r['oos_geo']:.3f}")
    e = res["large_vol_event"]
    print(f"  large-vol event: SSI hi={e['SSI_hi_mean']:.2f} lo={e['SSI_lo_mean']:.2f} t={e['t']:.1f} p={e['p']:.1e}")
    print("figures ->", figdir)


if __name__ == "__main__":
    main()
