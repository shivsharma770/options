# Research Milestone 2 — figures (smile-shape forecasting, SPY 2010–2021)

Figures for [../RESEARCH_M2_SMILE_FORECASTING.md](../RESEARCH_M2_SMILE_FORECASTING.md),
kept in git for future reference. The compact feature dataset lives under
`data/generated/research_m1/m2_features.csv` and the full outputs under
`data/generated/research_m2/` (git-ignored, regenerated on demand).

| File | Contents |
|---|---|
| `fig1_coef_heatmap.png` | Standardized model-B coefficients by feature × horizon |
| `fig2_feature_importance.png` | Mean \|HAC t\| across horizons (feature ranking) |
| `fig3_correlation_heatmap.png` | Feature correlation matrix (collinearity) |
| `fig4_incremental_power.png` | In-sample adj R² and out-of-sample R² for models A/B/C |
| `fig5_residual_diagnostics.png` | Residuals vs fitted, Q–Q, ACF, histogram (model B, h=20) |
| `m2_regression_results.csv` | adj-R², incremental F, HAC Wald, OOS R² per horizon/model |
| `m2_vif.csv`, `m2_feature_correlation.csv` | Multicollinearity diagnostics |
| `summary_stats.json` | Full machine-readable results |

## Regenerating

```sh
cmake -S . -B build -DORE_BUILD_EXAMPLES=ON && cmake --build build -j
for d in data/historical/spy/spy_eod_*/; do
  ./build/examples/example_historical_calibration "$d" SPY 4 0
  .venv/bin/python python/build_m2_features.py data/generated/research data/generated/research_m1
  rm -f data/generated/research/{calibration,smiles,surface,skew,term_structure}.csv
done
.venv/bin/python python/iv_smile_forecast_study.py
cp data/generated/research_m2/figures/*.png data/generated/research_m2/{summary_stats.json,m2_regression_results.csv,m2_vif.csv,m2_feature_correlation.csv} docs/figures/research_m2_smile/
```
