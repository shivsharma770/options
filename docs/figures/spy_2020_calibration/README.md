# SPY 2020 calibration study — figures

Publication-quality figures for [../SPY_2020_CALIBRATION_STUDY.md](../SPY_2020_CALIBRATION_STUDY.md),
kept in git for future reference. The underlying CSVs they are built from live
under `data/generated/research/` (git-ignored, regenerated on demand).

| File | Contents |
|---|---|
| `fig1_calibration_quality.png` | Daily IV-error RMSE/MAE vs provider + RMSE histogram |
| `fig2_term_structure.png` | ATM term structure on representative dates + date×maturity heatmap |
| `fig3_smiles.png` | 1-month OTM smiles (calm/crash/recovery) + curvature time series |
| `fig4_skew_evolution.png` | 25Δ risk reversal & butterfly + skew–vol scatter |
| `fig5_surface.png` | IV surface in log-moneyness space, calm vs crash |
| `summary_stats.json` | Machine-readable summary statistics |

## Regenerating

```sh
cmake -S . -B build -DORE_BUILD_EXAMPLES=ON && cmake --build build -j
./build/examples/example_historical_calibration data/historical/spy/spy_eod_2020-kwe0mi SPY 4 0
.venv/bin/python python/calibration_research.py     # writes to data/generated/research/figures/
cp data/generated/research/figures/*.png data/generated/research/summary_stats.json docs/figures/spy_2020_calibration/
```
