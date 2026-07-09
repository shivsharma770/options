# Research Milestone 6 — figures (IV surface geometry, SPY 2010–2021)

Figures for [../RESEARCH_M6_SURFACE_GEOMETRY.md](../RESEARCH_M6_SURFACE_GEOMETRY.md),
kept in git for future reference. Full outputs live under
`data/generated/research_m6/` (git-ignored, regenerated on demand).

| File | Contents |
|---|---|
| `fig1_surface_evolution.png` | The IV surface (moneyness × maturity) — calm vs COVID |
| `fig2_curvature_heatmap.png` | Smile and term-structure curvature over time |
| `fig3_ridge_trajectory.png` | Smile ridge (min-IV moneyness) trajectory |
| `fig4_geometry_pca.png` | PCA of the daily surface-geometry descriptor vector |
| `fig5_predictive.png` | OOS R², likelihood-ratio tests, and HAC coefficients (geometry vs benchmark) |
| `summary_stats.json` | Full machine-readable results (LR/HAC-Wald/OOS per target) |

## Regenerating

```sh
cmake -S . -B build -DORE_BUILD_EXAMPLES=ON && cmake --build build -j
for d in data/historical/spy/spy_eod_*/; do
  ./build/examples/example_historical_calibration "$d" SPY 4 0
  .venv/bin/python python/build_m6_surface_panel.py data/generated/research data/generated/research_m1
  rm -f data/generated/research/{calibration,smiles,surface,skew,term_structure}.csv
done
.venv/bin/python python/surface_geometry_study.py
cp data/generated/research_m6/figures/*.png data/generated/research_m6/summary_stats.json docs/figures/research_m6_geometry/
```
