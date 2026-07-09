# Research Milestone 4 — figures (higher-order Greeks, SPY 2010–2021)

Figures for [../RESEARCH_M4_HIGHER_ORDER_GREEKS.md](../RESEARCH_M4_HIGHER_ORDER_GREEKS.md),
kept in git for future reference. Full outputs live under
`data/generated/research_m4/` (git-ignored, regenerated on demand).

| File | Contents |
|---|---|
| `fig1_greek_profiles.png` | Vanna/Vomma/Charm/Speed/Zomma/Color across the 1-month smile, calm vs COVID |
| `fig2_greek_timeseries.png` | Higher-order Greeks of a rolling 1-month 25Δ SPY put, 2010–2021 |
| `fig3_pnl_attribution.png` | Second-order P&L attribution by regime; the Gamma-convexity residual |
| `fig4_correlation_pca.png` | Cross-Greek correlation matrix and PCA of the 13-Greek vector |
| `fig5_regime_comparison.png` | Greek magnitude by regime (low-vol / normal / recovery / COVID) |
| `fig6_empirical_tests.png` | Vomma→future \|ΔIV\|, and Γ/Vanna→delta-hedge error |
| `m4_greek_correlation.csv`, `summary_stats.json` | Correlations + full machine-readable results |

## Regenerating

```sh
cmake -S . -B build -DORE_BUILD_EXAMPLES=ON && cmake --build build -j
.venv/bin/python python/higher_order_greeks.py           # analytic-vs-FD validation
for d in data/historical/spy/spy_eod_*/; do
  ./build/examples/example_historical_calibration "$d" SPY 4 0
  .venv/bin/python python/build_m4_greeks_panel.py data/generated/research data/generated/research_m1
  rm -f data/generated/research/{calibration,smiles,surface,skew,term_structure}.csv
done
.venv/bin/python python/greeks_empirical_study.py
cp data/generated/research_m4/figures/*.png data/generated/research_m4/{summary_stats.json,m4_greek_correlation.csv} docs/figures/research_m4_greeks/
```
