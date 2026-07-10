#!/usr/bin/env python3
"""
surface_propagation_study.py -- Research Milestone 8.

Treats the implied-vol surface as a dynamic system and asks how an information
shock propagates through it: where does it appear first, how fast does the rest
of the surface adjust, and how does it recover? Ten phases -- shock identification,
propagation maps, lead-lag, recovery, shock decomposition (level/slope/curvature/
twist), market-maker quote behavior, diffusion anisotropy, regime comparison,
novel propagation metrics, and robustness.

Central caveat, stated up front: this uses END-OF-DAY data, so it resolves
*day-level* propagation and recovery only. Intraday lead-lag (which region ticks
first within a day) is invisible here; the resolvable questions are the spatial
pattern of the same-day response, its multi-day recovery, and its anisotropy.

Reuses the M6 region surface (`m6_surface.csv`) and the M8 quote panel
(`m8_quotes.csv`); no pricing/calibration/interpolation/data changes.

Usage:  surface_propagation_study.py [surf_csv] [quotes_csv] [out_dir]
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

MB = [14, 30, 60, 90, 180]
MA = [-0.12, -0.08, 0.0, 0.08, 0.12]
MA_LAB = ["10Δp", "25Δp", "ATM", "25Δc", "10Δc"]
NODES = [f"{MA_LAB[i]}_{mb}d" for mb in MB for i in range(len(MA))]
PRE, POST = 5, 10
REGIMES = {"calm": ("2013-01-01", "2017-12-31"), "2018": ("2018-01-01", "2018-12-31"),
           "COVID": ("2020-02-20", "2020-06-30")}
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.grid": True, "grid.alpha": 0.30, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.constrained_layout.use": True})
ACCENT, WARM, GOOD, NEUT = "#1f6feb", "#d1495b", "#2a9d8f", "#8d99ae"


# --------------------------------------------------------------------------
def build_grid(surf_csv):
    df = pd.read_csv(surf_csv, parse_dates=["date"])
    df = df[df["implied_volatility"] > 0]
    recs = []
    for d, g in df.groupby("date"):
        row = {"date": d, "spot": float(g["spot"].iloc[0])}
        ok = True
        for mb in MB:
            gb = g[g["mat_bucket"] == mb].sort_values("moneyness")
            if len(gb) < 4:
                ok = False; break
            for i, ma in enumerate(MA):
                row[f"{MA_LAB[i]}_{mb}d"] = float(np.interp(ma, gb["moneyness"].values,
                                                            gb["implied_volatility"].values))
        if ok:
            recs.append(row)
    p = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
    dIV = p[NODES].diff()
    return p, dIV


# --------------------------------------------------------------------------
# Phase 1: shocks
# --------------------------------------------------------------------------
def shocks(p, dIV):
    ev = {}
    atm = dIV["ATM_30d"].abs()
    ev["ATM top1%"] = p.index[atm >= atm.quantile(0.99)].tolist()
    ev["ATM top5%"] = p.index[atm >= atm.quantile(0.95)].tolist()
    ev["ATM decile"] = p.index[atm >= atm.quantile(0.90)].tolist()
    wing = dIV["25Δp_30d"].abs()
    ev["Wing shock"] = p.index[wing >= wing.quantile(0.90)].tolist()
    term = dIV["ATM_180d"].abs()
    ev["Term shock"] = p.index[term >= term.quantile(0.90)].tolist()
    Z = dIV[NODES].fillna(0.0).values
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    cov = np.cov(Zs.T); w, V = np.linalg.eigh(cov)
    pc1 = np.abs(Zs @ V[:, np.argmax(w)])
    ev["Surface PCA"] = p.index[pc1 >= np.quantile(pc1, 0.90)].tolist()
    return ev


def event_panel(p, dIV, events, sign_by=None):
    """Average cumulative IV change per node from Day -PRE..+POST, aligned so a
    shock has a consistent sign (multiply by sign of the shock driver)."""
    n = len(p)
    cum = {nd: np.zeros(PRE + POST + 1) for nd in NODES}
    cnt = np.zeros(PRE + POST + 1)
    driver = dIV[sign_by].values if sign_by else None
    for e in events:
        if e - PRE < 1 or e + POST >= n:
            continue
        s = np.sign(driver[e]) if driver is not None else 1.0
        base = p[NODES].iloc[e - 1]
        for k, day in enumerate(range(-PRE, POST + 1)):
            lvl = (p[NODES].iloc[e + day] - base) * s
            for nd in NODES:
                cum[nd][k] += lvl[nd]
        cnt += 1
    for nd in NODES:
        cum[nd] /= max(cnt.max(), 1)
    return cum, int(cnt.max())


# --------------------------------------------------------------------------
# Phase 3: lead-lag + Granger
# --------------------------------------------------------------------------
def xcorr(x, y, maxlag=5):
    x, y = x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)]
    out = {}
    for k in range(-maxlag, maxlag + 1):
        if k < 0:
            out[k] = float(np.corrcoef(x[:k], y[-k:])[0, 1])
        elif k > 0:
            out[k] = float(np.corrcoef(x[k:], y[:-k])[0, 1])
        else:
            out[k] = float(np.corrcoef(x, y)[0, 1])
    return out


def granger(y, x, L=3):
    """F-test that lags of x help predict y beyond lags of y."""
    n = len(y); Yt = y[L:]
    def lags(v):
        return np.column_stack([v[L - i - 1:n - i - 1] for i in range(L)])
    Xr = np.column_stack([np.ones(n - L), lags(y)])
    Xf = np.column_stack([Xr, lags(x)])
    br = np.linalg.lstsq(Xr, Yt, rcond=None)[0]; er = Yt - Xr @ br
    bf = np.linalg.lstsq(Xf, Yt, rcond=None)[0]; ef = Yt - Xf @ bf
    rss_r, rss_f = er @ er, ef @ ef
    F = ((rss_r - rss_f) / L) / (rss_f / (n - L - Xf.shape[1]))
    return float(F), float(stats.f.sf(F, L, n - L - Xf.shape[1]))


# --------------------------------------------------------------------------
# Phase 4: recovery half-lives (AR1 per node)
# --------------------------------------------------------------------------
def half_lives(p):
    out = {}
    for nd in NODES:
        s = p[nd].dropna().values
        ar1 = np.corrcoef(s[1:], s[:-1])[0, 1]
        out[nd] = float(np.log(0.5) / np.log(ar1)) if 0 < ar1 < 1 else np.nan
    return out


# --------------------------------------------------------------------------
# Phase 5: shock decomposition (interpret PCs)
# --------------------------------------------------------------------------
def decompose(dIV, events_by_type):
    Z = dIV[NODES].dropna()
    Zs = (Z - Z.mean()) / Z.std()
    cov = np.cov(Zs.values.T); w, V = np.linalg.eigh(cov)
    o = np.argsort(w)[::-1]; V = V[:, o]; evr = (w[o] / w.sum())
    # project each shock type's average day-0 response onto PC1..PC4
    proj = {}
    idx = Z.index
    for name, evs in events_by_type.items():
        evs = [e for e in evs if e in idx]
        if not evs:
            continue
        avg = Zs.loc[evs].mean().values
        proj[name] = [float(avg @ V[:, k]) for k in range(4)]
    return V, evr, proj


# --------------------------------------------------------------------------
# Phase 7 + 9: diffusion anisotropy + novel metrics
# --------------------------------------------------------------------------
def diffusion(dIV):
    D = dIV[NODES].dropna()
    C = D.corr()
    strike_pairs, mat_pairs = [], []
    for mb in MB:
        for i in range(len(MA) - 1):
            strike_pairs.append(C.loc[f"{MA_LAB[i]}_{mb}d", f"{MA_LAB[i+1]}_{mb}d"])
    for i in range(len(MA)):
        for j in range(len(MB) - 1):
            mat_pairs.append(C.loc[f"{MA_LAB[i]}_{MB[j]}d", f"{MA_LAB[i]}_{MB[j+1]}d"])
    cs, cm = float(np.mean(strike_pairs)), float(np.mean(mat_pairs))
    return {"corr_across_strikes": cs, "corr_across_maturities": cm,
            "directional_diffusion_ratio": cs / cm}, C


def novel_metrics(p, dIV, events):
    """Surface Synchronization Index, Propagation Entropy, Wavefront width per event."""
    sync, ent, wf = [], [], []
    sd = dIV[NODES].std()
    for e in events:
        if e < 1 or e >= len(p):
            continue
        z = (dIV[NODES].iloc[e] / sd).abs().values
        z = z[np.isfinite(z)]
        if len(z) == 0:
            continue
        sync.append(float(np.mean(z > 1.0)))          # fraction of nodes moving >1σ same day
        w = z / (z.sum() + 1e-9)
        ent.append(float(-np.sum(w * np.log(w + 1e-12)) / np.log(len(z))))  # normalized entropy
        wf.append(float(z.std()))
    return {"synchronization": np.array(sync), "entropy": np.array(ent), "wavefront": np.array(wf)}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_heatmap(cum, out):
    M = np.array([cum[nd] for nd in NODES])
    fig, ax = plt.subplots(figsize=(11, 7))
    lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[-PRE - 0.5, POST + 0.5, len(NODES) - 0.5, -0.5])
    ax.set_yticks(range(len(NODES))); ax.set_yticklabels(NODES, fontsize=6.5)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("event day (0 = shock)"); ax.set_title("Figure 1 — surface propagation heatmap (ATM shock, sign-aligned)")
    ax.grid(False); fig.colorbar(im, ax=ax, label="cumulative IV change vs Day −1", fraction=0.04, pad=0.02)
    fig.savefig(out / "fig1_propagation_heatmap.png"); plt.close(fig)


def fig_trajectories(cum, out):
    fig, ax = plt.subplots(figsize=(11, 5))
    days = np.arange(-PRE, POST + 1)
    show = {"ATM_30d": ACCENT, "25Δp_30d": WARM, "25Δc_30d": GOOD,
            "ATM_14d": "#8250df", "ATM_180d": NEUT}
    for nd, c in show.items():
        ax.plot(days, cum[nd], "-o", ms=3, color=c, label=nd)
    ax.axvline(0, color="k", lw=0.8, ls=":"); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("event day"); ax.set_ylabel("cumulative IV change vs Day −1")
    ax.set_title("Figure 2 — event-study trajectories by region (ATM shock)"); ax.legend(fontsize=8)
    fig.savefig(out / "fig2_event_trajectories.png"); plt.close(fig)


def fig_leadlag(dIV, out):
    pairs = [("25Δp_30d", "ATM_30d", "put→ATM"), ("ATM_14d", "ATM_180d", "short→long"),
             ("25Δp_30d", "25Δc_30d", "put→call")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for a, b, lab in pairs:
        xc = xcorr(dIV[a].values, dIV[b].values)
        ax[0].plot(sorted(xc), [xc[k] for k in sorted(xc)], "-o", ms=4, label=lab)
    ax[0].axvline(0, color="k", lw=0.7, ls=":")
    ax[0].set_title("(a) Cross-correlation (peak at k=0 ⇒ simultaneous)")
    ax[0].set_xlabel("lag k (days)"); ax[0].legend(fontsize=8)
    # Granger p-value matrix among 5 core regions
    core = ["ATM_14d", "ATM_180d", "25Δp_30d", "25Δc_30d", "ATM_30d"]
    P = np.ones((len(core), len(core)))
    for i, a in enumerate(core):
        for j, b in enumerate(core):
            if i != j:
                _, pv = granger(dIV[b].dropna().values, dIV[a].dropna().values)
                P[i, j] = pv
    im = ax[1].imshow(P, cmap="viridis_r", vmin=0, vmax=0.1)
    ax[1].set_xticks(range(len(core))); ax[1].set_xticklabels(core, rotation=40, ha="right", fontsize=7)
    ax[1].set_yticks(range(len(core))); ax[1].set_yticklabels(core, fontsize=7)
    ax[1].set_title("(b) Granger p (row→col); dark ⇒ significant"); ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)
    fig.suptitle("Phase 3 — lead-lag at daily resolution", fontweight="bold")
    fig.savefig(out / "fig3_leadlag_network.png"); plt.close(fig)


def fig_diffusion(C, aniso, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    im = ax[0].imshow(C.values, cmap="viridis", vmin=0.3, vmax=1)
    ax[0].set_title("(a) ΔIV correlation across the 25 surface nodes")
    ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)
    ax[1].bar(["across strikes\n(same maturity)", "across maturities\n(same strike)"],
              [aniso["corr_across_strikes"], aniso["corr_across_maturities"]], color=[ACCENT, WARM])
    ax[1].set_ylim(0, 1)
    ax[1].set_title(f"(b) Diffusion anisotropy\nDirectional Diffusion Ratio = {aniso['directional_diffusion_ratio']:.2f}")
    ax[1].set_ylabel("mean adjacent-node ΔIV correlation")
    fig.suptitle("Phase 7 — surface diffusion: does information travel faster across strikes or maturities?",
                 fontweight="bold")
    fig.savefig(out / "fig4_diffusion_graph.png"); plt.close(fig)


def fig_recovery(cum, hl, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    days = np.arange(0, POST + 1)
    for nd, c in [("ATM_30d", ACCENT), ("25Δp_30d", WARM), ("ATM_14d", "#8250df"), ("ATM_180d", NEUT)]:
        y = np.array([cum[nd][PRE + k] for k in range(POST + 1)])
        y = y / (y[0] + 1e-9)
        ax[0].plot(days, y, "-o", ms=3, color=c, label=nd)
    ax[0].axhline(0.5, color="k", ls="--", lw=0.8); ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_title("(a) Post-shock normalization (fraction of Day-0 response)")
    ax[0].set_xlabel("days after shock"); ax[0].legend(fontsize=8)
    reg = {"10Δp": [], "25Δp": [], "ATM": [], "25Δc": [], "10Δc": []}
    for nd in NODES:
        for k in reg:
            if nd.startswith(k):
                reg[k].append(hl[nd])
    labs = ["10Δp", "25Δp", "ATM", "25Δc", "10Δc"]
    ax[1].bar(labs, [np.nanmean(reg[k]) for k in labs], color=NEUT)
    ax[1].set_title("(b) IV half-life by moneyness (AR1, days)"); ax[1].set_ylabel("half-life (days)")
    fig.suptitle("Phase 4 — recovery dynamics", fontweight="bold")
    fig.savefig(out / "fig5_recovery_curves.png"); plt.close(fig)


def fig_decomp(V, evr, proj, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    names = ["PC1 (level)", "PC2 (skew/slope)", "PC3 (curv)", "PC4 (twist)"]
    im = ax[0].imshow(V[:, :4], cmap="coolwarm", vmin=-0.4, vmax=0.4, aspect="auto")
    ax[0].set_yticks(range(len(NODES))); ax[0].set_yticklabels(NODES, fontsize=6)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels([f"PC{i+1}\n{evr[i]*100:.0f}%" for i in range(4)])
    ax[0].set_title("(a) ΔIV PCA loadings"); ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)
    x = np.arange(len(proj)); w = 0.2
    for k in range(4):
        ax[1].bar(x + (k - 1.5) * w, [proj[nm][k] for nm in proj], w, label=names[k])
    ax[1].set_xticks(x); ax[1].set_xticklabels(list(proj.keys()), rotation=25, ha="right", fontsize=8)
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_title("(b) Shock decomposition by type"); ax[1].legend(fontsize=7)
    fig.suptitle("Phase 5 — shock decomposition (level / skew / curvature / twist)", fontweight="bold")
    fig.savefig(out / "fig6_pca_decomposition.png"); plt.close(fig)


def fig_metrics(nm, out, boot):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for k, (key, lab) in enumerate([("synchronization", "Synchronization Index\n(frac. nodes >1σ, Day 0)"),
                                    ("entropy", "Propagation Entropy\n(0=concentrated,1=dispersed)"),
                                    ("wavefront", "Wavefront width\n(σ of response)")]):
        ax[k].hist(nm[key], bins=30, color=ACCENT, edgecolor="white")
        ax[k].axvline(np.mean(nm[key]), color=WARM, ls="--", lw=1.4,
                      label=f"mean={np.mean(nm[key]):.2f}\n95%CI[{boot[key][0]:.2f},{boot[key][1]:.2f}]")
        ax[k].set_title(lab); ax[k].legend(fontsize=8)
    fig.suptitle("Phase 9 — novel propagation-metric distributions (ATM shocks)", fontweight="bold")
    fig.savefig(out / "fig7_metric_distributions.png"); plt.close(fig)


def fig_regimes(p, dIV, events, out):
    labels = list(REGIMES); metrics = {}
    for rname, (lo, hi) in REGIMES.items():
        mask = (p["date"] >= lo) & (p["date"] <= hi)
        evs = [e for e in events if mask.iloc[e]]
        nm = novel_metrics(p, dIV, evs)
        ani, _ = diffusion(dIV[mask.values])
        metrics[rname] = {"sync": float(np.mean(nm["synchronization"])) if len(nm["synchronization"]) else np.nan,
                          "entropy": float(np.mean(nm["entropy"])) if len(nm["entropy"]) else np.nan,
                          "ddr": ani["directional_diffusion_ratio"], "n_events": len(evs)}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for k, key in enumerate(["sync", "entropy", "ddr"]):
        ax[k].bar(labels, [metrics[r][key] for r in labels],
                  color=[GOOD, "#8250df", WARM])
        ax[k].set_title({"sync": "Synchronization", "entropy": "Propagation entropy",
                         "ddr": "Directional diffusion ratio"}[key])
    fig.suptitle("Phase 8 — propagation by regime (2022 outside sample)", fontweight="bold")
    fig.savefig(out / "fig8_regime_comparison.png"); plt.close(fig)
    return metrics


# --------------------------------------------------------------------------
def main():
    surf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/generated/research_m1/m6_surface.csv")
    quotes = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/generated/research_m1/m8_quotes.csv")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/generated/research_m8")
    figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)

    print("grid"); p, dIV = build_grid(surf)
    print("phase1 shocks"); ev = shocks(p, dIV)
    print("phase2 propagation"); cum, ncnt = event_panel(p, dIV, ev["ATM decile"], sign_by="ATM_30d")
    fig_heatmap(cum, figdir); fig_trajectories(cum, figdir)
    print("phase3 leadlag"); fig_leadlag(dIV, figdir)
    ll = {a + "->" + b: xcorr(dIV[a].values, dIV[b].values)
          for a, b in [("25Δp_30d", "ATM_30d"), ("ATM_14d", "ATM_180d"), ("25Δp_30d", "25Δc_30d")]}
    gr = {}
    for a, b in [("ATM_14d", "ATM_180d"), ("25Δp_30d", "ATM_30d"), ("ATM_180d", "ATM_14d")]:
        gr[a + "->" + b] = granger(dIV[b].dropna().values, dIV[a].dropna().values)[1]
    print("phase4 recovery"); hl = half_lives(p); fig_recovery(cum, hl, figdir)
    print("phase5 decomposition")
    V, evr, proj = decompose(dIV, {k: ev[k] for k in ["ATM decile", "Wing shock", "Term shock", "Surface PCA"]})
    fig_decomp(V, evr, proj, figdir)
    print("phase7 diffusion"); ani, C = diffusion(dIV); fig_diffusion(C, ani, figdir)
    print("phase9 metrics"); nm = novel_metrics(p, dIV, ev["ATM decile"])
    rng = np.random.default_rng(0); boot = {}
    for key in ["synchronization", "entropy", "wavefront"]:
        a = nm[key]; bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(1000)]
        boot[key] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    fig_metrics(nm, figdir, boot)
    print("phase8 regimes"); reg = fig_regimes(p, dIV, ev["ATM decile"], figdir)

    # phase6 quotes (if available)
    q6 = {}
    if quotes.exists():
        q = pd.read_csv(quotes, parse_dates=["date"])
        m = p[["date"]].merge(q, on="date", how="left")
        for col in ["mean_rel_spread", "quote_count"]:
            base = m[col].values
            resp = np.zeros(PRE + POST + 1); cnt = 0
            for e in ev["ATM decile"]:
                if e - PRE < 1 or e + POST >= len(m):
                    continue
                b0 = base[e - 1]
                if not np.isfinite(b0) or b0 == 0:
                    continue
                resp += (base[e - PRE:e + POST + 1] / b0 - 1); cnt += 1
            q6[col] = (resp / max(cnt, 1)).tolist()
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        days = np.arange(-PRE, POST + 1)
        ax[0].plot(days, q6["mean_rel_spread"], "-o", ms=3, color=WARM)
        ax[0].axvline(0, color="k", ls=":", lw=0.8); ax[0].set_title("(a) Relative bid-ask spread around ATM shock")
        ax[0].set_xlabel("event day"); ax[0].set_ylabel("Δ vs Day −1")
        ax[1].plot(days, q6["quote_count"], "-o", ms=3, color=ACCENT)
        ax[1].axvline(0, color="k", ls=":", lw=0.8); ax[1].set_title("(b) Quote count around ATM shock")
        ax[1].set_xlabel("event day")
        fig.suptitle("Phase 6 — market-maker quote behavior around IV shocks", fontweight="bold")
        fig.savefig(figdir / "fig9_quote_behavior.png"); plt.close(fig)

    summary = {"underlying": "SPY",
               "period": {"start": str(p["date"].min().date()), "end": str(p["date"].max().date()), "days": int(len(p))},
               "n_events": {k: len(v) for k, v in ev.items()},
               "phase3_leadlag_peaklag": {k: max(v, key=lambda kk: abs(v[kk])) for k, v in ll.items()},
               "phase3_granger_p": gr,
               "phase4_half_life_by_moneyness": {mlab: float(np.nanmean([hl[nd] for nd in NODES if nd.startswith(mlab)]))
                                                 for mlab in MA_LAB},
               "phase4_half_life_by_maturity": {f"{mb}d": float(np.nanmean([hl[nd] for nd in NODES if nd.endswith(f"_{mb}d")]))
                                                for mb in MB},
               "phase5_decomp": {"evr": np.round(evr[:4], 3).tolist(), "projections": proj},
               "phase7_diffusion": ani,
               "phase9_metrics": {k: {"mean": float(np.mean(nm[k])), "ci": boot[k]} for k in boot},
               "phase8_regimes": reg, "phase6_quotes": q6}
    with open(out / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    p.to_csv(out / "m8_surface_grid.csv", index=False)

    print("\n" + "=" * 90)
    print(f"SPY {summary['period']['start']}..{summary['period']['end']} ({len(p)} days)")
    print("events:", summary["n_events"])
    print("Phase3 lead-lag peak lags:", summary["phase3_leadlag_peaklag"])
    print("Phase3 Granger p:", {k: round(v, 3) for k, v in gr.items()})
    print("Phase4 half-life by moneyness:", {k: round(v, 1) for k, v in summary["phase4_half_life_by_moneyness"].items()})
    print("Phase4 half-life by maturity:", {k: round(v, 1) for k, v in summary["phase4_half_life_by_maturity"].items()})
    print("Phase5 decomp evr:", summary["phase5_decomp"]["evr"])
    print("Phase7 diffusion:", {k: round(v, 3) for k, v in ani.items()})
    print("Phase9 metrics:", {k: (round(v["mean"], 3), [round(x, 3) for x in v["ci"]]) for k, v in summary["phase9_metrics"].items()})
    print("Phase8 regimes:", {r: {k: round(v, 2) for k, v in reg[r].items() if k != "n_events"} for r in reg})
    print("figures ->", figdir)


if __name__ == "__main__":
    main()
