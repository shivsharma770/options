# Research Milestone 1 — figures (IV → future RV, SPY 2010–2021)

Figures for [../RESEARCH_M1_IV_PREDICTS_RV.md](../RESEARCH_M1_IV_PREDICTS_RV.md),
kept in git for future reference. The compact source dataset
(`daily_underlying.csv`, `atm_term_structure.csv`) and the full-size CSV outputs
live under `data/generated/research_m1/` (git-ignored, regenerated on demand).

| File | Contents |
|---|---|
| `fig1_iv_rv_timeseries.png` | ATM IV vs subsequently realized 20-day vol, 2010–2021 |
| `fig2_scatter_by_horizon.png` | Realized vs implied vol per horizon, OLS fit + 45° line |
| `fig3_coefficients.png` | Slope β (95% HAC bands) and forecast R² (IV vs naive benchmark) |
| `fig4_bias_vrp.png` | Mean forecast bias by horizon + IV−RV premium over time |
| `regression_results.csv` | All coefficients, HAC SEs, tests, and accuracy metrics per horizon |
| `summary_stats.json` | Machine-readable summary |

## Regenerating

```sh
cmake -S . -B build -DORE_BUILD_EXAMPLES=ON && cmake --build build -j
for d in data/historical/spy/spy_eod_*/; do
  ./build/examples/example_historical_calibration "$d" SPY 4 0
  .venv/bin/python python/build_iv_rv_dataset.py data/generated/research data/generated/research_m1
  rm -f data/generated/research/{calibration,smiles,surface,skew}.csv
done
.venv/bin/python python/iv_rv_study.py
cp data/generated/research_m1/figures/*.png data/generated/research_m1/{summary_stats.json,regression_results.csv} docs/figures/research_m1_iv_rv/
```
