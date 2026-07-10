# Research Milestone 8 — figures (IV surface information propagation, SPY 2010–2021)

Figures for [../RESEARCH_M8_INFORMATION_PROPAGATION.md](../RESEARCH_M8_INFORMATION_PROPAGATION.md),
kept in git for future reference. Full outputs live under
`data/generated/research_m8/` (git-ignored, regenerated on demand). Built on the
Milestone-6 region surface and a compact quote panel from `calibration.csv`.

| File | Phase | Contents |
|---|---|---|
| `fig1_propagation_heatmap.png` | 2 | Surface propagation heatmap (node × event day, ATM shock) |
| `fig2_event_trajectories.png` | 2 | Event-study IV trajectories by region |
| `fig3_leadlag_network.png` | 3 | Cross-correlation + Granger-causality network |
| `fig4_diffusion_graph.png` | 7 | ΔIV correlation matrix + diffusion anisotropy |
| `fig5_recovery_curves.png` | 4 | Post-shock recovery and half-lives by region |
| `fig6_pca_decomposition.png` | 5 | PCA shock decomposition (level/skew/curvature/twist) |
| `fig7_metric_distributions.png` | 9 | Synchronization / entropy / wavefront distributions (bootstrap CIs) |
| `fig8_regime_comparison.png` | 8 | Propagation metrics by regime (calm / 2018 / COVID) |
| `fig9_quote_behavior.png` | 6 | Bid-ask spread and quote count around IV shocks |
| `summary_stats.json` | — | Full machine-readable results |

## Regenerating

```sh
# Phase 6 quote panel (per-day spread/count from calibration.csv):
for d in data/historical/spy/spy_eod_*/; do
  ./build/examples/example_historical_calibration "$d" SPY 4 0
  .venv/bin/python python/build_m8_quotes.py data/generated/research data/generated/research_m1
  rm -f data/generated/research/{calibration,smiles,surface,skew,term_structure}.csv
done
# Phases 1-5,7-10 reuse m6_surface.csv from Milestone 6:
.venv/bin/python python/surface_propagation_study.py
cp data/generated/research_m8/figures/*.png data/generated/research_m8/summary_stats.json docs/figures/research_m8_propagation/
```
