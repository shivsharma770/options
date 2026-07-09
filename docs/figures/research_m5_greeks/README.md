# Research Milestone 5 — figures (Greek information content, SPY 2010–2021)

Figures for [../RESEARCH_M5_GREEK_INFORMATION.md](../RESEARCH_M5_GREEK_INFORMATION.md),
kept in git for future reference. Full outputs live under
`data/generated/research_m5/` (git-ignored, regenerated on demand). Built on the
Milestone-4 smile panel and the validated `higher_order_greeks.py` — no data
regenerated.

| File | Study | Contents |
|---|---|---|
| `fig1_study1_vol_forecast.png` | 1 | OOS RV-forecast R² (ATM vs ATM+HO) + Diebold–Mariano tests |
| `fig2_study2_dealer_flow.png` | 2 | Aggregate Greek exposure and its return/RV predictability |
| `fig3_study3_regime.png` | 3 | ATM IV vs higher-order Greeks; do they precede vol events? |
| `fig4_study4_hedge_attribution.png` | 4 | Share of delta-hedge residual explained per 2nd-order term |
| `fig5_study5_xsection.png` | 5 | Cross-sectional decile option returns (moneyness confound) |
| `fig6_study6_structure.png` | 6 | Correlation, PCA, hierarchical clustering (+ VIF in JSON) |
| `fig7_study7_importance.png` | 7 | LASSO/ElasticNet/RandomForest importance ranking |
| `summary_stats.json` | — | Full machine-readable results (incl. correlations, VIFs, DM tests) |

## Regenerating

```sh
# requires the M4 smile panel (data/generated/research_m1/m4_smile30.csv) and M1 masters
.venv/bin/python python/greek_information_study.py
cp data/generated/research_m5/figures/*.png data/generated/research_m5/summary_stats.json docs/figures/research_m5_greeks/
```
