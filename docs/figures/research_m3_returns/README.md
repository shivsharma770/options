# Research Milestone 3 — figures (option features → returns, SPY 2010–2021)

Figures for [../RESEARCH_M3_OPTION_RETURN_PREDICTABILITY.md](../RESEARCH_M3_OPTION_RETURN_PREDICTABILITY.md),
kept in git for future reference. The full outputs live under
`data/generated/research_m3/` (git-ignored, regenerated on demand).

| File | Contents |
|---|---|
| `fig1_coef_evolution.png` | 2-year rolling coefficients of key features (20-day excess return) |
| `fig2_rolling_power.png` | 2-year rolling predictive R² by horizon (instability) |
| `fig3_return_scatter.png` | In-sample predicted vs actual excess return by horizon |
| `fig4_crash_events.png` | ATM IV / put skew around the 2011, 2018-Q4, 2020 drawdowns |
| `fig5_predrawdown_distributions.png` | Feature distributions before worst-decile drawdowns vs normal |
| `m3_regression_results.csv` | R², joint Wald p, HAC t per feature (all targets × horizons) |
| `summary_stats.json` | Full results incl. subperiod stability and pre-drawdown shifts |

## Regenerating

```sh
cmake -S . -B build -DORE_BUILD_EXAMPLES=ON && cmake --build build -j
for d in data/historical/spy/spy_eod_*/; do
  ./build/examples/example_historical_calibration "$d" SPY 4 0
  .venv/bin/python python/build_m3_calib.py data/generated/research data/generated/research_m1
  rm -f data/generated/research/{calibration,smiles,surface,skew,term_structure}.csv
done
.venv/bin/python python/option_return_predictability.py
cp data/generated/research_m3/figures/*.png data/generated/research_m3/{summary_stats.json,m3_regression_results.csv} docs/figures/research_m3_returns/
```
