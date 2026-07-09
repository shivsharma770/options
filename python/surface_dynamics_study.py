#!/usr/bin/env python3
"""
surface_dynamics_study.py -- Research Milestone 7.

Moves from "what does the surface look like?" to "how is it moving?". Treats the
sequence of daily surface states (the Milestone-6 geometric state vector) as a
trajectory in state space and asks whether its *dynamics* -- velocity,
acceleration, turning, persistence, memory, entropy, and regime structure --
carry incremental predictive information beyond the static level, smile, Greeks,
and geometry already studied.

Eight phases:
  1 state space           (reuse the M6 geometric state vector)
  2 dynamics              (velocity/acceleration/jerk/turning/path/persistence)
  3 dynamic geometry      (region lead-lag: ATM vs wing, short vs long maturity)
  4 regime detection      (k-means / GMM / hierarchical / HDBSCAN + transitions)
  5 information content    (nested baselines A-E: static -> dynamic; HAC/LR/OOS/DM)
  6 surface memory        (ACF/PACF, half-life, impulse response, continue/reverse)
  7 entropy & complexity  (path efficiency, deformation energy, compression, roughness)
  8 novel questions        (speed vs level; instability before crises; inertia)

Reuses m6_geometry.csv (state), m6_surface.csv (regions), and the HAC-OLS
estimator. No pricing/calibration/interpolation/data changes.

Usage:  surface_dynamics_study.py [m6_geometry.csv] [m6_surface.csv] [out_dir]
"""
from __future__ import annotations

import gzip
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
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from iv_rv_study import ols_hac

TD = 252
STATE = ["atm_iv", "smile_slope", "smile_curv", "ts_slope", "ts_curv", "skew_asym",
         "bf_intensity", "rough1d", "rough2d", "ridge", "CCI", "SSI", "SAI"]
EVENTS = {"2011 EU/US debt": ("2011-07-15", "2011-10-15"),
          "2015 China": ("2015-08-15", "2015-09-30"),
          "2018 Volmageddon": ("2018-02-01", "2018-02-15"),
          "2018 Q4": ("2018-10-01", "2018-12-31"),
          "COVID": ("2020-02-20", "2020-04-30")}
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.grid": True, "grid.alpha": 0.30,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True})
ACCENT, WARM, GOOD, NEUT = "#1f6feb", "#d1495b", "#2a9d8f", "#8d99ae"


# --------------------------------------------------------------------------
# Phase 1-2: state space + dynamics
# --------------------------------------------------------------------------
def build(master_geo, master_surf):
    p = pd.read_csv(master_geo, parse_dates=["date"]).dropna(subset=STATE).reset_index(drop=True)
    Z = ((p[STATE] - p[STATE].mean()) / p[STATE].std()).values          # standardized state
    n = len(p)
    V = np.vstack([np.zeros(len(STATE)), np.diff(Z, axis=0)])           # velocity
    A = np.vstack([np.zeros(len(STATE)), np.diff(V, axis=0)])           # acceleration
    J = np.vstack([np.zeros(len(STATE)), np.diff(A, axis=0)])           # jerk
    speed = np.linalg.norm(V, axis=1)
    accel = np.linalg.norm(A, axis=1)
    jerk = np.linalg.norm(J, axis=1)
    # turning angle between consecutive velocity vectors
    turn = np.zeros(n)
    for i in range(2, n):
        a, b = V[i], V[i - 1]
        d = np.linalg.norm(a) * np.linalg.norm(b)
        turn[i] = np.arccos(np.clip(a @ b / (d + 1e-12), -1, 1)) if d > 0 else 0
    p["speed"], p["accel"], p["jerk"], p["turn_angle"] = speed, accel, jerk, turn
    p["path_len"] = np.cumsum(speed)
    p["traj_curv"] = turn / (speed + 1e-9)
    # deformation energy (20d motion intensity) + path efficiency (displacement/path)
    w = 20
    defo = pd.Series(speed ** 2).rolling(w).sum().values
    disp = np.full(n, np.nan); eff = np.full(n, np.nan)
    for i in range(w, n):
        disp[i] = np.linalg.norm(Z[i] - Z[i - w])
        pl = speed[i - w + 1:i + 1].sum()
        eff[i] = disp[i] / (pl + 1e-9)
    p["deform_energy"], p["path_eff"] = defo, eff
    # compression ratio of the recent standardized-state window (Kolmogorov proxy)
    comp = np.full(n, np.nan)
    for i in range(w, n):
        raw = np.round(Z[i - w:i], 3).astype(np.float32).tobytes()
        comp[i] = len(gzip.compress(raw)) / len(raw)
    p["compression"] = comp
    return p, Z, V


# --------------------------------------------------------------------------
# Phase 3: region lead-lag
# --------------------------------------------------------------------------
def regions(master_surf):
    df = pd.read_csv(master_surf, parse_dates=["date"])
    rows = []
    for d, g in df.groupby("date"):
        rec = {"date": d}
        for b, lab in [(14, "short"), (180, "long")]:
            gb = g[g["mat_bucket"] == b]
            if len(gb) >= 3:
                i0 = gb["moneyness"].abs().values.argmin()
                rec[f"atm_{lab}"] = float(gb.iloc[i0]["implied_volatility"])
        gb = g[g["mat_bucket"] == 30]
        if len(gb) >= 5:
            i0 = gb["moneyness"].abs().values.argmin()
            rec["atm"] = float(gb.iloc[i0]["implied_volatility"])
            wing = gb[gb["moneyness"].abs() >= 0.08]["implied_volatility"]
            rec["wing"] = float(wing.mean()) if len(wing) else np.nan
        rows.append(rec)
    r = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    for c in ["atm", "wing", "atm_short", "atm_long"]:
        r[f"d_{c}"] = r[c].diff()
    return r


def leadlag(r, a, b, maxlag=5):
    x, y = r[f"d_{a}"].dropna(), r[f"d_{b}"].dropna()
    idx = x.index.intersection(y.index)
    x, y = x.loc[idx].values, y.loc[idx].values
    out = {}
    for k in range(-maxlag, maxlag + 1):
        if k < 0:
            out[k] = float(np.corrcoef(x[:k], y[-k:])[0, 1])
        elif k > 0:
            out[k] = float(np.corrcoef(x[k:], y[:-k])[0, 1])
        else:
            out[k] = float(np.corrcoef(x, y)[0, 1])
    return out


# --------------------------------------------------------------------------
# Phase 4: regimes
# --------------------------------------------------------------------------
def regimes(p, Z, k=4):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
    lab = km.labels_
    # order clusters by mean ATM IV (0=calm .. k-1=crisis)
    means = [p["atm_iv"].values[lab == c].mean() for c in range(k)]
    order = np.argsort(means); remap = {c: i for i, c in enumerate(order)}
    lab = np.array([remap[c] for c in lab])
    p["regime"] = lab
    # transition matrix + expected duration
    T = np.zeros((k, k))
    for i in range(1, len(lab)):
        T[lab[i - 1], lab[i]] += 1
    T = T / T.sum(axis=1, keepdims=True)
    dur = {i: float(1 / (1 - T[i, i])) for i in range(k)}
    gmm = GaussianMixture(k, random_state=0).fit(Z)
    return lab, T, dur, float((lab == np.array([remap[c] for c in gmm.predict(Z)])).mean())


# --------------------------------------------------------------------------
# Phase 5: nested information content
# --------------------------------------------------------------------------
def oos(X, y, h=20, n0=504):
    n = len(y); pr = np.full(n, np.nan)
    for i in range(n0, n):
        te = i - h
        if te < 252:
            continue
        b = np.linalg.lstsq(X[:te + 1], y[:te + 1], rcond=None)[0]
        pr[i] = X[i] @ b
    return pr


def dm(y, p1, p2, h):
    m = ~np.isnan(p1) & ~np.isnan(p2) & ~np.isnan(y)
    d = (y[m] - p1[m]) ** 2 - (y[m] - p2[m]) ** 2
    n = len(d); dc = d - d.mean(); var = np.sum(dc ** 2)
    for l in range(1, h):
        var += 2 * (1 - l / h) * np.sum(dc[l:] * dc[:-l])
    var /= n
    s = d.mean() / np.sqrt(var / n)
    return float(s), float(2 * stats.norm.sf(abs(s)))


def info_content(p, figdir):
    tgt = "rv_20"
    blocks = {
        "A ATM": ["atm_iv"],
        "B +HV": ["atm_iv", "hv_20"],
        "C +smile": ["atm_iv", "hv_20", "smile_slope", "smile_curv", "skew_asym", "bf_intensity"],
        "D +geometry": ["atm_iv", "hv_20", "smile_slope", "smile_curv", "skew_asym", "bf_intensity",
                        "ts_curv", "rough2d", "ridge", "SSI", "CCI", "SAI"],
        "E +dynamics": ["atm_iv", "hv_20", "smile_slope", "smile_curv", "skew_asym", "bf_intensity",
                        "ts_curv", "rough2d", "ridge", "SSI", "CCI", "SAI",
                        "speed", "accel", "turn_angle", "stability", "deform_energy", "path_eff"],
    }
    need = sorted(set(sum(blocks.values(), [])) | {tgt})
    sub = p[["date"] + need].dropna().reset_index(drop=True)
    # standardize predictors so inv(X'X) in ols_hac is well-conditioned
    # (predictors span orders of magnitude; this is scale-only, R²/OOS unchanged)
    for c in need:
        if c != tgt:
            sd = sub[c].std()
            sub[c] = (sub[c] - sub[c].mean()) / (sd if sd > 0 else 1.0)
    y = sub[tgt].values; n = len(sub); ones = np.ones(n)
    ybench = np.full(n, np.nan)
    for i in range(504, n):
        te = i - 20
        if te >= 252:
            ybench[i] = y[:te + 1].mean()
    def rob(X):                       # robust (lstsq / min-norm) OLS fit
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ b
        r2 = 1 - (e @ e) / ((y - y.mean()) ** 2).sum()
        adj = 1 - (1 - r2) * (n - 1) / (n - np.linalg.matrix_rank(X))
        return b, float(e @ e), float(r2), float(adj)

    res = {}; preds = {}
    for name, cols in blocks.items():
        X = np.column_stack([ones] + [sub[c].values for c in cols])
        _, _, r2, adj = rob(X)
        pr = oos(X, y, 20)
        m = ~np.isnan(pr) & ~np.isnan(ybench)
        oosr2 = 1 - np.sum((y[m] - pr[m]) ** 2) / np.sum((y[m] - ybench[m]) ** 2)
        res[name] = {"r2": r2, "adj_r2": adj, "oos_r2": float(oosr2),
                     "rmse": float(np.sqrt(np.nanmean((y - pr) ** 2))),
                     "mae": float(np.nanmean(np.abs(y - pr)))}
        preds[name] = pr
    # E vs D: LR (robust RSS) + DM (OOS, robust); primary verdict is DM/OOS
    subD = blocks["D +geometry"]
    dyn = ["speed", "accel", "turn_angle", "stability", "deform_energy", "path_eff"]
    _, rssD, _, _ = rob(np.column_stack([ones] + [sub[c].values for c in subD]))
    _, rssE, _, _ = rob(np.column_stack([ones] + [sub[c].values for c in subD + dyn]))
    lr = n * np.log(rssD / rssE); lrp = float(stats.chi2.sf(max(lr, 0), len(dyn)))
    dmstat, dmp = dm(y, preds["D +geometry"], preds["E +dynamics"], 20)
    # clean, well-conditioned HAC t of dynamics beyond level + HV (no collinear geometry)
    Xd = np.column_stack([ones, sub["atm_iv"], sub["hv_20"]] + [sub[c].values for c in dyn])
    fd = ols_hac(y, Xd, 20)
    seld = list(range(3, 3 + len(dyn)))
    bsel = fd["beta"][seld]; V = fd["cov"][np.ix_(seld, seld)]
    wald = float(bsel @ np.linalg.solve(V, bsel)); waldp = float(stats.chi2.sf(wald, len(dyn)))
    res["_tests_EvsD"] = {"lr": float(lr), "lr_p": lrp, "wald_p": waldp,
                          "dm": dmstat, "dm_p": dmp,
                          "dyn_t": {dyn[i]: float(fd["tstat"][seld[i]]) for i in range(len(dyn))}}

    names = list(blocks.keys())
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [res[nm]["adj_r2"] for nm in names], 0.4, color=NEUT, label="in-sample adj R²")
    ax[0].bar(x + 0.2, [res[nm]["oos_r2"] for nm in names], 0.4, color=GOOD, label="OOS R²")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=20, ha="right")
    ax[0].set_title("(a) Nested baselines A→E, future RV (h=20d)"); ax[0].legend()
    dt = res["_tests_EvsD"]["dyn_t"]
    ax[1].barh(list(dt.keys()), list(dt.values()),
               color=[GOOD if abs(v) > 1.96 else NEUT for v in dt.values()])
    ax[1].axvline(1.96, color=WARM, ls="--", lw=1); ax[1].axvline(-1.96, color=WARM, ls="--", lw=1)
    ax[1].set_title(f"(b) Dynamics HAC t (E vs D)\nWald p={waldp:.2f}  DM p={dmp:.2f}")
    ax[1].set_xlabel("t-stat")
    fig.suptitle("Phase 5 — do surface dynamics add beyond static geometry?", fontweight="bold")
    fig.savefig(figdir / "fig5_info_content.png"); plt.close(fig)
    return res


# --------------------------------------------------------------------------
# Phase 6-7: memory + complexity
# --------------------------------------------------------------------------
def acf(x, nlags):
    x = x - x.mean(); v = x @ x
    return [1.0] + [float((x[k:] @ x[:-k]) / v) for k in range(1, nlags + 1)]


def memory_complexity(p, figdir):
    iv = p["atm_iv"].dropna().values
    sp = p["speed"].dropna().values
    ac_iv = acf(iv, 60); ac_sp = acf(sp, 30)
    ar1 = ac_iv[1]; half = float(np.log(0.5) / np.log(ar1)) if 0 < ar1 < 1 else np.nan
    # velocity autocorrelation (momentum vs reversion of state moves)
    dv = np.diff(iv); ac_dv = acf(dv, 20)
    # continue/reverse: after a big move (top-decile speed), next-5d speed vs return
    q = p[["speed", "rv_5", "ret"]].dropna()
    thr = q["speed"].quantile(0.9)
    big = q[q["speed"] >= thr]; nrm = q[q["speed"] < thr]
    # complexity -> future RV: does path_eff / deform_energy / compression predict RV_20?
    tests = {}
    for c in ["path_eff", "deform_energy", "compression", "turn_angle"]:
        s = p[[c, "rv_20"]].dropna()
        X = np.column_stack([np.ones(len(s)), (s[c] - s[c].mean()) / s[c].std()])
        f = ols_hac(s["rv_20"].values, X, 20)
        tests[c] = {"t": float(f["tstat"][1]), "r2": float(f["r2"])}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].bar(range(len(ac_iv)), ac_iv, color=ACCENT)
    ax[0].axhline(1.96 / np.sqrt(len(iv)), color=WARM, ls="--", lw=1)
    ax[0].set_title(f"(a) ATM-IV ACF  (AR1={ar1:.2f}, half-life={half:.0f}d)")
    ax[0].set_xlabel("lag (days)")
    ax[1].bar(range(len(ac_dv)), ac_dv, color=NEUT)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axhline(-1.96 / np.sqrt(len(dv)), color=WARM, ls="--", lw=1)
    ax[1].axhline(1.96 / np.sqrt(len(dv)), color=WARM, ls="--", lw=1)
    ax[1].set_title("(b) ΔATM-IV ACF (momentum vs reversion)"); ax[1].set_xlabel("lag")
    tv = {k: v["t"] for k, v in tests.items()}
    ax[2].barh(list(tv.keys()), list(tv.values()),
               color=[GOOD if abs(v) > 1.96 else NEUT for v in tv.values()])
    ax[2].axvline(1.96, color=WARM, ls="--", lw=1); ax[2].axvline(-1.96, color=WARM, ls="--", lw=1)
    ax[2].set_title("(c) Complexity → future RV (HAC t)"); ax[2].set_xlabel("t-stat")
    fig.suptitle("Phases 6–7 — surface memory & complexity", fontweight="bold")
    fig.savefig(figdir / "fig6_memory_complexity.png"); plt.close(fig)
    return {"atm_ar1": float(ar1), "atm_half_life_days": half,
            "dv_acf1": float(ac_dv[1]), "speed_acf1": float(ac_sp[1]),
            "big_move_next_rv5": float(big["rv_5"].mean()), "normal_next_rv5": float(nrm["rv_5"].mean()),
            "complexity_tests": tests}


# --------------------------------------------------------------------------
# Figures: trajectory, dynamics, regimes
# --------------------------------------------------------------------------
def fig_trajectory(p, Z, lab, figdir):
    cov = np.cov(Z.T); w, Vv = np.linalg.eigh(cov); o = np.argsort(w)[::-1]
    PC = Z @ Vv[:, o[:2]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sc = ax[0].scatter(PC[:, 0], PC[:, 1], c=p["atm_iv"], cmap="viridis", s=6, alpha=0.5)
    ax[0].set_title("(a) Surface trajectory in state space (PC1–PC2)")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].grid(False)
    fig.colorbar(sc, ax=ax[0], label="ATM IV", fraction=0.046, pad=0.02)
    ax[1].plot(p["date"], p["speed"], color=WARM, lw=0.6)
    for lab_, (lo, hi) in EVENTS.items():
        ax[1].axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color=NEUT, alpha=0.18)
    ax[1].set_title("(b) Surface speed |ΔstateΔt| through time"); ax[1].set_ylabel("speed")
    ax[1].xaxis.set_major_locator(mdates.YearLocator()); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Phase 2 — surface trajectory & speed", fontweight="bold")
    fig.savefig(figdir / "fig1_trajectory.png"); plt.close(fig)


def fig_regimes(p, T, dur, figdir):
    k = T.shape[0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    cmap = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, k))
    ax[0].scatter(p["date"], p["atm_iv"], c=cmap[p["regime"].values], s=5)
    ax[0].set_title("(a) Regime assignment (colored, 0=calm→3=crisis)"); ax[0].set_ylabel("ATM IV")
    ax[0].xaxis.set_major_locator(mdates.YearLocator()); ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    im = ax[1].imshow(T, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            ax[1].text(j, i, f"{T[i, j]:.2f}", ha="center", va="center",
                       color="white" if T[i, j] > 0.5 else "black")
    ax[1].set_xticks(range(k)); ax[1].set_yticks(range(k))
    ax[1].set_title("(b) Regime transition matrix"); ax[1].set_xlabel("to"); ax[1].set_ylabel("from")
    ax[1].grid(False); fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02)
    fig.suptitle(f"Phase 4 — regimes  (expected durations: "
                 + ", ".join(f"R{i}:{dur[i]:.0f}d" for i in range(k)) + ")", fontweight="bold")
    fig.savefig(figdir / "fig4_regimes.png"); plt.close(fig)


def fig_leadlag(r, figdir):
    ll_aw = leadlag(r, "wing", "atm")            # does wing lead ATM?
    ll_sl = leadlag(r, "atm_short", "atm_long")  # does short maturity lead long?
    vols = {c: float(r[f"d_{c}"].std()) for c in ["atm", "wing", "atm_short", "atm_long"]}
    pers = {c: float(pd.Series(r[c]).autocorr(1)) for c in ["atm", "wing", "atm_short", "atm_long"]}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ks = sorted(ll_aw)
    ax[0].plot(ks, [ll_aw[k] for k in ks], "-o", color=ACCENT, ms=4, label="wing→ATM")
    ax[0].plot(ks, [ll_sl[k] for k in ks], "-o", color=WARM, ms=4, label="short→long")
    ax[0].axvline(0, color="k", lw=0.7, ls=":")
    ax[0].set_title("(a) Lead–lag cross-correlation\n(peak at k<0 ⇒ first series leads)")
    ax[0].set_xlabel("lag k (days)"); ax[0].set_ylabel("corr(Δx_t, Δy_{t−k})"); ax[0].legend()
    x = np.arange(4); labs = ["ATM", "wing", "short", "long"]
    ax[1].bar(x - 0.2, [vols[c] for c in ["atm", "wing", "atm_short", "atm_long"]], 0.4,
              color=WARM, label="daily-change vol")
    ax[1].bar(x + 0.2, [pers[c] for c in ["atm", "wing", "atm_short", "atm_long"]], 0.4,
              color=GOOD, label="persistence (AR1)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labs)
    ax[1].set_title("(b) Region volatility & persistence"); ax[1].legend()
    fig.suptitle("Phase 3 — which part of the surface moves first?", fontweight="bold")
    fig.savefig(figdir / "fig3_leadlag.png"); plt.close(fig)
    return {"wing_to_atm": ll_aw, "short_to_long": ll_sl, "region_vol": vols, "region_persist": pers}


# --------------------------------------------------------------------------
def main():
    geo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/generated/research_m6/m6_geometry.csv")
    surf = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/generated/research_m1/m6_surface.csv")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/generated/research_m7")
    figdir = out / "figures"; figdir.mkdir(parents=True, exist_ok=True)

    print("[1-2] state + dynamics"); p, Z, V = build(geo, surf)
    print("[3] regions"); r = regions(surf); ll = fig_leadlag(r, figdir)
    print("[4] regimes"); lab, T, dur, gmm_agree = regimes(p, Z)
    fig_trajectory(p, Z, lab, figdir); fig_regimes(p, T, dur, figdir)
    print("[5] information content"); ic = info_content(p, figdir)
    print("[6-7] memory + complexity"); mc = memory_complexity(p, figdir)
    p.to_csv(out / "m7_dynamics.csv", index=False)

    summary = {"underlying": "SPY",
               "period": {"start": str(p["date"].min().date()), "end": str(p["date"].max().date()),
                          "days": int(len(p))},
               "phase3_leadlag": ll,
               "phase4_regimes": {"transition": T.tolist(), "expected_duration_days": dur,
                                  "gmm_kmeans_agreement": gmm_agree},
               "phase5_info_content": ic, "phase6_7_memory_complexity": mc}
    with open(out / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 92)
    print(f"SPY {summary['period']['start']}..{summary['period']['end']} ({len(p)} days)")
    print("PHASE 5 — nested baselines (adj R² / OOS R²):")
    for nm in ["A ATM", "B +HV", "C +smile", "D +geometry", "E +dynamics"]:
        print(f"   {nm:>13}: adj={ic[nm]['adj_r2']:.3f}  OOS={ic[nm]['oos_r2']:.3f}")
    t = ic["_tests_EvsD"]
    print(f"   E vs D: LR p={t['lr_p']:.1e}  Wald p={t['wald_p']:.2f}  DM p={t['dm_p']:.2f}")
    print(f"PHASE 3 — wing→ATM peak lag: {max(ll['wing_to_atm'], key=lambda k: abs(ll['wing_to_atm'][k]))}, "
          f"short→long peak lag: {max(ll['short_to_long'], key=lambda k: abs(ll['short_to_long'][k]))}")
    print(f"PHASE 6 — ATM AR1={mc['atm_ar1']:.3f} half-life={mc['atm_half_life_days']:.0f}d  "
          f"ΔATM acf1={mc['dv_acf1']:+.3f}  big-move next-RV5={mc['big_move_next_rv5']:.3f} vs {mc['normal_next_rv5']:.3f}")
    print("PHASE 4 — regime durations: " + ", ".join(f"R{i}:{dur[i]:.0f}d" for i in dur)
          + f"  gmm/kmeans agree={gmm_agree:.2f}")
    print("figures ->", figdir)


if __name__ == "__main__":
    main()
