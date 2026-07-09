# Research Milestone 7 — figures (IV surface dynamics, SPY 2010–2021)

Figures for [../RESEARCH_M7_SURFACE_DYNAMICS.md](../RESEARCH_M7_SURFACE_DYNAMICS.md),
kept in git for future reference. Full outputs live under
`data/generated/research_m7/` (git-ignored, regenerated on demand). Built on the
Milestone-6 state vector and region surface — no data regenerated.

| File | Phase | Contents |
|---|---|---|
| `fig1_trajectory.png` | 2 | Surface trajectory in PC1–PC2 state space + speed through time |
| `fig3_leadlag.png` | 3 | Region lead-lag cross-correlation, volatility, and persistence |
| `fig4_regimes.png` | 4 | k-means regime assignment and transition matrix |
| `fig5_info_content.png` | 5 | Nested baselines A→E (in-sample vs OOS) + dynamics HAC t-stats |
| `fig6_memory_complexity.png` | 6–7 | ATM-IV ACF & half-life, ΔIV ACF, complexity → future RV |
| `summary_stats.json` | — | Full machine-readable results (lead-lag, regimes, LR/Wald/DM, memory) |

## Regenerating

```sh
# requires m6_geometry.csv + m6_surface.csv from Milestone 6
.venv/bin/python python/surface_dynamics_study.py
cp data/generated/research_m7/figures/*.png data/generated/research_m7/summary_stats.json docs/figures/research_m7_dynamics/
```
