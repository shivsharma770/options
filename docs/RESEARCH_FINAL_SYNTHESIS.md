# The Implied-Volatility Surface: What Eight Milestones Actually Established

**Capstone Synthesis of Research Milestones 1–8 — SPY Options, 2010–2021**

*An objective review of a completed empirical research program. No new experiments, models, data, or alpha search — only synthesis, evaluation, and honest assessment.*

---

## 1. Executive Summary

Over eight milestones this project interrogated a single object — the SPY
implied-volatility surface, 2010–2021 — from progressively more sophisticated
angles, always asking a version of one question: **does the surface contain
information beyond its ATM level, and does that information survive rigorous,
out-of-sample testing?**

The answer, earned honestly and repeatedly, is: **almost never.**

* **What survived.** ATM implied volatility is a strong, near-unbiased,
  out-of-sample predictor of realized volatility (M1). The surface embeds a
  persistent ~10% volatility risk premium (M1). Skew is strongly mean-reverting
  and that dynamic is predictable out-of-sample (M6). And the surface is a
  tightly-coupled dynamic *system* whose shock-propagation and diffusion have
  real, measurable, previously-undocumented structure (M8).
* **What failed.** Everything meant to *add* forecasting power beyond the ATM
  level: smile shape at long horizons (M2), return direction (M3), higher-order
  Greeks (M5), surface geometry for volatility (M6), and surface dynamics (M7)
  each looked significant in-sample and each **degraded out-of-sample**.
* **The through-line.** The project's most valuable output is not a signal but a
  discipline: a unified protocol — Newey–West HAC standard errors, HAC-Wald and
  likelihood-ratio tests, expanding-window out-of-sample R², and Diebold–Mariano
  tests — that repeatedly **dissolved apparent predictability** the moment it was
  applied. In-sample R², feature importances, and rich models were, again and
  again, a mirage.

The single most defensible sentence the project can assert: **the implied-vol
surface is richly informative about the present — the volatility level, the term
structure, the skew, and how they move right now — and, once honestly tested,
almost uninformative about the future beyond the volatility level itself.**

---

## 2. Research Timeline and Map

Each milestone was a direct response to the previous one's result.

```
M1  Does ATM IV predict realized vol?                  → YES, robustly (OOS R² 0.58→0.14; β≈1 short-horizon)
      │  ("the level works — does anything add to it?")
      ▼
M2  Does smile SHAPE add beyond ATM IV?                → Only at ≤2 weeks; hurts OOS at 20–60d (ATM↔RR corr −0.85)
      │  ("shape is redundant — what about returns?")
      ▼
M3  Do surfaces predict RETURNS (not vol)?             → NO; apparent R² is an overlap artifact; unstable, sign-flipping
      │  ("returns are hard — is the info in the Greeks?")
      ▼
M4  Higher-order Greeks: math + P&L attribution        → Descriptive, not predictive; Greeks are a low-vol signature
      │  ("Greeks describe — do they forecast? [ML included]")
      ▼
M5  Do higher-order Greeks carry information?           → NO; they HURT OOS vol forecasts (DM p≤0.05); massively redundant
      │  ("Greeks are redundant — is the GEOMETRY richer?")
      ▼
M6  Does surface GEOMETRY predict?                     → NO for RV/VRP (LR p 1e-21 but Wald p 0.20, OOS worse);
      │                                                    YES, uniquely, for future SKEW (OOS R² 0→0.33)
      │  ("geometry fails OOS — is the MOTION the signal?")
      ▼
M7  Do surface DYNAMICS add beyond static?             → NO; nested A→E OOS R² FALLS 0.371→0.31; dynamics Wald p=0.18, DM p=0.63
      │  ("nothing forecasts — stop forecasting. How does info MOVE?")
      ▼
M8  How does information PROPAGATE through the surface? → Systems-level structure: same-day surface-wide response,
                                                          region-specific recovery, maturity>strike diffusion anisotropy
```

The arc has a clear shape: **the more sophisticated the feature, the more
convincingly it fails out-of-sample** — until M8 abandons forecasting entirely and
finds that the surface's genuine structure is in its *dynamics as a system*, not
its predictive content.

---

## 3. Master Results Table

| M | Research question | Data | Methodology | Statistical tests | Main quantitative result | Robustness | Conclusion | Survived OOS? |
|---|---|---|---|---|---|---|---|---|
| **1** | Does ATM IV forecast future RV? | SPY 2010–21, 2,994 days; matched-tenor ATM IV | Mincer–Zarnowitz `RV=α+β·IV`, matched tenor | HAC (lag h−1); test β=1, α=0 | OOS R² **0.58→0.14** (5→60d); β **0.95→0.57**, t(β=1) −0.6→−4.1; IV/RV **1.08–1.16** (VRP); corr 0.76→0.37 | log spec, horse-race vs HV, encompassing | IV is a strong, short-horizon-unbiased RV forecast that **subsumes historical vol**; persistent VRP | **Yes** |
| **2** | Does smile shape add beyond ATM IV? | + skew/curvature/slope, TS slopes | Nested A/B/C, HAC-Wald, expanding-window OOS | HAC-Wald, incremental F, VIF | In-sample jointly sig (Wald p≤1e-3); **OOS +0.05 at 5d, −0.06 at 60d**; ATM↔RR **−0.85**, VIF 7.3/5.3; TS-slope-short dominant (t=−6.5) | smile-only≈full; per-horizon | Smile is **statistically** significant but **economically marginal**; helps only ≤2 weeks | **Partial** (short only) |
| **3** | Do surfaces predict returns? | + forward returns/drawdown | 7 features → 4 targets; rolling; subperiods | HAC, rolling R², subperiod stability | Apparent R² **2%→19%** (overlap artifact); rolling R² **0.02–0.47**, collapses 2018–19; subperiod **sign flips**; only ATM IV robust (+, risk premium) | 3 subperiods, rolling, drawdown | Returns **not robustly predictable**; long-horizon R² is mechanical; drawdown = vol in disguise | **No** |
| **4** | Higher-order Greeks: structure & P&L | + analytic Greeks (validated <5e-5 vs FD) | Math survey; 2nd-order P&L attribution; regimes | — (descriptive) + FD validation | Vanna/Vomma vanish ATM; Greeks are a **low-vol signature**; Γ+Vanna corr 0.66 w/ hedge error | March-2020 case; validation | Higher-order Greeks are **attribution, not forecasting** tools | n/a (descriptive) |
| **5** | Do higher-order Greeks carry info? | + daily Greek panel | 7 studies incl. LASSO/RF/permutation | DM, HAC-Wald, VIF, PCA | Greeks **HURT** OOS RV (DM p≤0.05, 10–60d); no return pred; extreme Vanna/Vomma → **calmer** markets (t=−14); **Gamma≈all** hedge residual; decile spreads = **moneyness artifact**; VIF in **hundreds** | 7 independent studies | Higher-order Greeks carry **no incremental OOS info**; massively redundant | **No** |
| **6** | Does surface geometry predict? | + 13 geometric descriptors, 4 novel indices | Incremental HAC, LR, expanding OOS | HAC-Wald, LR, OOS | RV/VRP: LR p **1e-21** but **Wald p 0.20, OOS worse**; **Δskew: Wald p 5e-78, OOS 0→0.33**; geometry ~7 PCs; Stress Index **lower** before vol events | LR-vs-Wald contrast | Geometry fails OOS for RV/VRP; **one real signal — skew mean-reversion** | **No** (except skew) |
| **7** | Do surface dynamics add beyond static? | + velocity/accel/entropy/regimes | Nested A→E; regimes (k-means/GMM); memory | HAC, LR, Wald, OOS, DM | Nested adj R² **↑0.370→0.404** but **OOS ↓0.371→0.31**; dynamics **Wald p 0.18, DM p 0.63**; no daily lead-lag; ΔIV reverses (acf −0.25), speed inertia (acf 0.36) | bootstrap-adjacent; regimes | Dynamics add **nothing OOS**; parsimonious ATM wins | **No** |
| **8** | How does information propagate? | 25-node grid + quote panel | Event study, Granger, recovery, PCA decomp, diffusion | Granger, bootstrap CIs | Same-day surface-wide (**Sync 0.58, Entropy 0.94**); **short→long / put→ATM** Granger lead (p≈0 vs 0.54); recovery region-specific (short 4d, ATM/put/long 12–14d); shocks **level-dominated**; **diffusion anisotropy DDR 0.60** (maturity>strike), **isotropic in COVID (0.93)**; spreads widen 2.4% **with** shock | bootstrap, 3 thresholds, 4 shock defs | Surface is a **tightly-coupled dynamic system**; genuine, novel *descriptive* structure | n/a (not forecasting) |

---

## 4. Cross-Milestone Themes

**Hypotheses that consistently survived robust inference.**
1. **ATM IV → realized volatility** (M1): the only forecasting relationship that
   is strong, unbiased at short horizons, OOS-robust, and stable across
   subperiods. Everything else is measured *relative to this baseline*.
2. **The volatility risk premium** (M1, M6): IV exceeds subsequent RV by ~10%,
   robustly and persistently.
3. **Skew mean-reversion** (M6): the one *incremental* signal that survives
   HAC-Wald and improves OOS R² (0→0.33) — though it is partly mechanical
   (regressing a change on its level).

**Hypotheses that consistently failed.**
- Smile shape beyond ATM (M2, long horizons), return predictability (M3),
  higher-order Greek information (M5), surface geometry for RV/VRP (M6), and
  surface dynamics (M7). In every case the *in-sample* evidence was strong and the
  *out-of-sample* evidence was absent or negative.

**Apparent discoveries that disappeared under HAC / OOS.** This is the project's
signature pattern, and it recurred at least five times:
| Milestone | Looked significant (in-sample) | Verdict (robust) |
|---|---|---|
| M3 returns | R² up to 19% at 60d | Overlap artifact; unstable, sign-flipping |
| M5 Greeks | RF/LASSO flag Vanna/Vomma | DM-significantly *worse* OOS |
| M6 geometry→RV | LR p ≈ 10⁻²¹ | HAC-Wald p = 0.20; OOS worse |
| M7 dynamics | in-sample adj R² rises to 0.40 | Wald p 0.18, DM p 0.63; OOS falls |
| M2 smile→RV (long h) | Wald p ≤ 10⁻³ | OOS negative beyond 2 weeks |

**Findings economically meaningful after all checks.** Realistically only two:
the **ATM-IV level as a volatility forecast** and the **volatility risk premium**.
Skew mean-reversion is statistically robust but of limited standalone economic
value; the M8 propagation results are *structural*, not economic.

**Negative results that are real methodological contributions.** The
demonstration that (i) *long-horizon overlapping-return R²* is mechanically
inflated (M3), (ii) *feature importance ≠ out-of-sample value* under
multicollinearity (M5, M7), and (iii) the *LR–vs–HAC-Wald gap* (a p-value of
10⁻²¹ collapsing to 0.20; M6) is the single cleanest illustration in the project
of why overlap-robust inference is mandatory. These negatives are more useful
than most positives a naive version of the project would have "found."

---

## 5. Discussion

**What did the project actually discover?** Three things. (1) A careful,
OOS-robust *quantification* of ATM-IV's volatility-forecasting power and the
volatility risk premium on a modern liquid ETF — a rigorous replication of
Christensen–Prabhala/Poon–Granger. (2) One genuinely incremental predictive
signal — *skew mean-reversion* (M6). (3) A previously-undocumented *structural*
fact — the implied-vol surface diffuses shocks **more coherently across maturities
than across strikes**, and this anisotropy **collapses to isotropy in crises**
(M8).

**What misconceptions did it overturn?** The pervasive practitioner/undergraduate
intuition that *more surface detail = more predictive power*. Smile shape,
higher-order Greeks, surface geometry, and surface dynamics are each *richer*
descriptions of the surface, and each is *less* useful out-of-sample than the
one-number ATM level. The project is, in effect, a sustained refutation of
feature-richness as a route to volatility/return forecasting.

**What surprised us most?** Two results. First, that higher-order Greeks
*actively degrade* OOS forecasts (M5) — not merely fail to help, but
Diebold–Mariano-significantly hurt. Second, the M6 LR-vs-Wald gap: a likelihood-
ratio p-value of 10⁻²¹ for surface geometry that becomes an insignificant HAC-Wald
p of 0.20 — the same data, two tests, opposite conclusions, and only the robust
one generalizes.

**Findings that contradict naive intuition.** (a) Extreme Vanna/Vomma precede
*calmer*, not more volatile, markets (M5) — high higher-order Greeks are a low-vol
signature. (b) A "stressed-looking" surface (high roughness/curvature) is a
calm-market feature and is *lower* before large-vol events (M6/M7). (c) Big
cross-sectional option-return spreads sorted on Greeks are a *moneyness artifact*,
not a factor (M5).

**Agreement with the literature.** M1 reproduces Christensen–Prabhala (1998) and
the Poon–Granger (2003) consensus; the VRP echoes Bollerslev–Tauchen–Zhou (2009);
M3's instability echoes Welch–Goyal (2008) and the Boudoukh–Richardson–Whitelaw
(2008) overlap critique; the redundancy of surface features is consistent with
Cont–da Fonseca's (2002) low-dimensional surface dynamics.

**Where the project extends the literature.** Modestly but genuinely: the M8
**diffusion anisotropy** (maturity-coherence > strike-coherence, with crisis
isotropy) and the associated propagation metrics (Synchronization, Directional
Diffusion Ratio) do not appear in the standard literature and are a small, honest,
reproducible novel contribution. The project's *breadth of OOS-robust negative
results on a single surface* is itself uncommon.

---

## 6. Contribution Evaluation and Ranking

Each contribution scored on **novelty**, **statistical credibility**, and
**practical usefulness** (H/M/L), ranked strongest → weakest.

| Rank | Contribution | Novelty | Stat. credibility | Practical use | Net |
|---|---|---|---|---|---|
| 1 | **The OOS-robust "predictability audit"** — a unified HAC/Wald/OOS/DM protocol showing surface features add ~nothing beyond ATM IV | **M** (method application at this scope) | **H** | **M–H** (saves you from overfit signals) | **Strongest** |
| 2 | **ATM-IV → RV quantification + VRP** (M1) | L (established) | **H** | **H** | Very strong |
| 3 | **Diffusion anisotropy & crisis isotropy** (M8) | **H** | **M–H** (bootstrap CIs) | L (descriptive) | Strong-novel |
| 4 | **Skew mean-reversion is OOS-predictable** (M6) | M | **H** | M | Solid |
| 5 | **Region-specific recovery half-lives** (M8) | M | M | L | Solid-descriptive |
| 6 | **Higher-order Greeks degrade OOS forecasts** (M5) | M | **H** | M (tells you *not* to use them) | Useful negative |
| 7 | **Validated analytic higher-order Greek library** (M4) | L | H | M (reusable tooling) | Useful engineering |
| 8 | Surface-geometry / dynamics novel indices (M6/M7) | M | L–M (mostly OOS-null) | L | Weakest |

---

## 7. The Single Strongest Contribution

Not the most statistically significant result (that is M6's LR p = 10⁻⁷⁸ for
Δskew, which is partly mechanical). Judged on the combination of **originality,
robustness, importance, and reproducibility**, the strongest contribution is:

> **A reproducible, uniformly-applied out-of-sample "predictability audit" of the
> implied-volatility surface that shows, across eight independent studies, that no
> layer of surface sophistication — smile shape, higher-order Greeks, geometry, or
> dynamics — adds economically meaningful, out-of-sample forecasting power beyond
> the ATM implied-volatility level, and that the apparent additions are in-sample
> and overlap artifacts.**

**Why this, and why it beats the flashier candidates.**
- **Originality.** The novel *finding* is the diffusion anisotropy (M8), but a
  single descriptive fact on one asset is a thin contribution. The audit's
  originality is in the *scope and consistency* of the negative result: applying
  the same overlap-robust machinery to eight escalating hypotheses and reporting
  every failure honestly is rare — most work of this kind stops at the first
  in-sample "success."
- **Robustness.** The contribution *is* robustness — it is the accumulated output
  of HAC-Wald, Diebold–Mariano, and expanding-window OOS. It cannot be an
  overfit, because it is the demonstration that the alternatives *are*.
- **Importance.** It answers the actual question a practitioner or academic cares
  about — "is there tradeable/forecasting information in the surface beyond the
  VIX-like level?" — with a defensible *no*. A credible negative on a heavily
  data-mined object is worth more than another fragile positive.
- **Reproducibility.** Every step is scripted (`iv_rv_study.py` … `surface_dynamics_study.py`),
  reuses one calibration pipeline, one HAC-OLS estimator, and one OOS protocol,
  and regenerates from the raw archive by documented commands.

The audit is the project's spine; the diffusion anisotropy (M8) is its best single
*novel* result, and the ATM-IV/VRP quantification (M1) its best *practical* one.

---

## 8. Negative Results That Matter

Three negatives are, on reflection, worth more than any positive the project could
plausibly have manufactured:

1. **Long-horizon return R² is mechanical (M3).** The 2%→19% rise with horizon is
   the Boudoukh–Richardson–Whitelaw overlap artifact, not signal — a concrete
   reproduction that should inoculate the reader against long-horizon
   predictability claims forever.
2. **Feature importance ≠ out-of-sample value (M5, M7).** Random-Forest importance
   and LASSO flag Vanna/Vomma, and those exact features degrade OOS forecasts. In
   a field increasingly driven by ML feature attributions on collinear inputs,
   this is a clean cautionary example.
3. **In-sample significance evaporates under HAC (M6).** A likelihood-ratio
   p-value of 10⁻²¹ and a HAC-Wald p-value of 0.20 on the *same* data is the
   single most compact argument in the project for why overlap-robust inference
   is not optional.

---

## 9. Methodological Lessons

1. **Always report OOS, and make it expanding-window with a look-ahead guard.**
   Overlapping targets (h-day RV/returns) make in-sample R² and LR tests
   systematically over-optimistic.
2. **Use HAC-Wald, not LR, for nested tests on autocorrelated targets.** The two
   disagreed by ~20 orders of magnitude in M6.
3. **Diebold–Mariano is the right adjudicator** when two models' forecasts must be
   compared under overlap.
4. **Collinearity is the silent killer.** VIFs of the surface features reach the
   hundreds; feature importances split credit arbitrarily and coefficients flip
   sign across subperiods. Standardize, check VIF, and distrust individual
   coefficients.
5. **Watch for confounds masquerading as factors** (the moneyness/leverage
   confound in M5's cross-sectional sorts).
6. **A negative result, rigorously established, is a result.** Reporting it is the
   difference between research and a backtest brochure.

---

## 10. Limitations (and why they matter)

* **Daily end-of-day data.** The binding constraint. All propagation/lead-lag
  results (M7, M8) are *day-level*; genuine intraday information flow — who ticks
  first within a day — is invisible. The "same-day, surface-wide" propagation
  finding is a statement about daily data, and the Granger short→long edge is the
  finest resolution EOD permits. *Why it matters:* the most interesting
  microstructure questions (dealer hedging latency, wing-vs-ATM lead) are simply
  unanswerable here.
* **SPY-only sample.** One underlying, the most liquid and efficient index-option
  chain in the world. *Why it matters:* results may not generalize to single
  names (where skew *does* predict cross-sectional returns; Xing–Zhang–Zhao),
  other indices, or less liquid markets; SPY is the *hardest* place to find
  predictability, which biases the project toward negatives.
* **Sample period 2010–2021.** Post-GFC, ZIRP-heavy, one major crisis (COVID) plus
  two smaller ones. *Why it matters:* the 2022 rate-driven bear market is **outside
  the sample**; regime conclusions rest on few crisis episodes; the VRP and
  IV→RV relationship may differ in a high-rate or inflationary regime.
* **European-Black–Scholes, r = q = 0 IVs.** SPY options are American; the
  calibrator uses European BS with zero rate and dividend. *Why it matters:* this
  biases IV *levels* (mean |IV error| vs provider ≈ 5 vol points, RMSE ≈ 20 in the
  low-vega wings) and contaminates deep-ITM/LEAP IVs; near-ATM, short-tenor
  results are largely insulated (absorbed by regression intercept/slope), but
  wing and long-dated descriptors carry a systematic bias.
* **Missing intraday information.** No opening/overnight decomposition, no
  realized-kernel RV; realized vol is close-to-close only, a noisier estimator
  that lowers R² (though it does not bias β).
* **Measurement error.** IVs come from mid-quotes of calibrated contracts;
  illiquid strikes are dropped; ridge/roughness descriptors are the noisiest and
  least individually significant.
* **No transaction costs, no capacity, no signed positioning.** The one
  semi-tradeable signal (skew mean-reversion) is never subjected to a P&L test
  net of the bid-ask spread (which M8 shows widens on shock days); the dealer-flow
  hypothesis (M5) cannot be tested without OI/positioning data. *Why it matters:*
  "statistically significant" is a long way from "profitable after costs," and the
  project deliberately never crosses that line.
* **Multiple testing.** Across eight milestones, dozens of hypotheses were tested;
  no family-wise or FDR correction was applied. *Why it matters:* the *positives*
  (skew MR, diffusion anisotropy) should be read with this in mind — though the
  dominant pattern being *negative* makes the project robust to this critique in
  aggregate.

---

## 11. External Critical Evaluation

*Four reviewers, asked to be honest rather than kind.*

**A Jane Street quant.**
- *Strengths:* the OOS discipline and the willingness to publish negatives; the
  validated Greeks; clean reuse. This is how a real desk thinks about signal.
- *Weaknesses:* no intraday, no latency, no transaction costs, no live/holdout
  period, single asset. Nothing here is tradeable and the project correctly never
  claims otherwise.
- *Impressed by:* the M6 LR-vs-Wald gap and the M5 "importance ≠ OOS value"
  result — exactly the traps that sink junior researchers.
- *Would do differently:* one intraday dataset and one honest P&L test (even a
  paper straddle/variance-swap) of skew mean-reversion, net of the spread.
- *Vs typical undergrad:* far above. Most undergrad projects report a Sharpe-3
  backtest that is pure overfit; this one is the opposite failure mode, which is
  the *right* failure mode.

**A Citadel researcher.**
- *Strengths:* systematic robustness; the propagation/diffusion angle is a genuine
  idea; reproducibility.
- *Weaknesses:* no cross-sectional dimension (single names are where option
  signals live), no signed dealer flow (the GEX hypothesis is untestable here),
  the one positive (skew MR) is partly mechanical and untraded.
- *Impressed by:* diffusion anisotropy and its crisis isotropy — a legitimately
  novel descriptive result.
- *Would do differently:* push to a single-name cross-section and get OI/flow
  data; formalize skew MR as a tradeable with a transaction-cost model.
- *Vs typical undergrad:* clearly stands out; approaches the low end of a strong
  first-year-analyst research note, minus the tradeable payoff.

**An options-market academic reviewer.**
- *Strengths:* sound econometrics (HAC, DM, expanding OOS); honest negatives;
  faithful replication of the IV→RV literature.
- *Weaknesses:* SPY-only limits external validity; the European/r=q=0 IV bias is a
  real measurement concern for the shape descriptors; no multiple-testing
  correction; several "novel indices" (Surface Stress, Curvature Concentration)
  are ad hoc and mostly OOS-null; the skew-MR result needs cleaner identification
  (it is close to an AR(1) tautology).
- *Impressed by:* the diffusion-anisotropy result, which could be a short
  empirical note if extended to more indices.
- *Would do differently:* multiple underlyings; a term-structure-consistent IV
  (SVI or a documented existing surface) to remove the level bias; a formal
  encompassing framework across all feature families.
- *Vs typical undergrad:* well above; publishable as a rigorous replication plus
  one modest novel descriptive finding, not as a top-journal contribution.

**An experienced software engineer.**
- *Strengths:* genuine reuse (one calibration pipeline, one HAC-OLS estimator, one
  OOS protocol reused across eight studies with almost no duplication); the
  C++/Python separation (pricing/calibration in C++, measurement/analysis in
  Python) is exactly right; per-year ETL that bounds disk to the compact panels;
  validated Greeks with a finite-difference self-check; every figure and dataset
  reproducible from the raw archive.
- *Weaknesses:* the research scripts occasionally re-implement small helpers
  rather than import them; no CI and no unit tests on the analysis layer (the C++
  has GoogleTest, the Python does not); a bruising CRLF/line-ending episode and
  pre-existing numerical fragility in a few C++ tests; the studies are scripts,
  not a packaged library.
- *Impressed by:* the discipline of never modifying the pricing/calibration core
  across eight research milestones, and the honesty of the git history.
- *Would do differently:* factor the shared HAC/OOS/DM utilities into one imported
  module; add a thin test suite and CI for the analysis; add a `.gitattributes`
  earlier (it eventually was).
- *Vs typical undergrad:* substantially above; the infrastructure hygiene and
  reproducibility are the project's most unambiguous strength.

**Consensus.** This is an *excellent, unusually honest, reproducible empirical
audit* — well above a typical undergraduate quantitative-finance project — whose
main value is methodological rigor and negative results rather than novel alpha.
It would stand out in an undergraduate/portfolio context and hold up as a rigorous
replication in a professional one; it is *not*, and does not claim to be,
research-scientist-level novel contribution.

---

## 12. Future Research Directions

Within the honest boundaries the project established:
1. **Intraday data** to resolve the propagation/lead-lag questions M7–M8 could
   only bound (does the wing/short end *really* lead within the day?).
2. **Cross-section of single names**, where option-implied signals (skew, IV
   innovations) are documented to predict returns — the one place the negative
   might flip positive.
3. **Remove the IV level bias**: recalibrate with real rate/dividend curves and an
   American pricer, or a documented SVI surface, and re-run the shape descriptors.
4. **A transaction-cost-aware economic test** of the two survivors (VRP capture;
   skew mean-reversion) — the step the project deliberately never took.
5. **Signed dealer positioning / GEX data** to test the flow hypotheses M5 could
   only reject in weak form.
6. **Multiple indices and a longer, higher-rate sample** (2022+) to test external
   validity and regime robustness.

---

## 13. Final Conclusions — Three Things to Remember in Five Years

If nothing else from this program survives, these three should:

1. **The ATM implied-volatility level is the whole forecasting story.** On SPY,
   over twelve years, no smile, Greek, geometric, or dynamic elaboration of the
   surface adds robust out-of-sample forecasting power beyond the single ATM-IV
   number (and its historical-vol companion). Sophistication did not help; it
   overfit. The one-number VIX-style level, plus the ~10% volatility risk premium
   it embeds, is what an honest forecaster keeps.

2. **In-sample significance is not evidence.** Time and again — long-horizon
   return R² (M3), ML feature importance (M5), surface geometry (M6), surface
   dynamics (M7) — a relationship that was overwhelmingly significant in-sample
   (LR p-values to 10⁻²¹) vanished or reversed under HAC-Wald, Diebold–Mariano,
   and expanding-window out-of-sample testing. The project's most transferable
   lesson is a protocol, not a signal: *if it hasn't survived overlap-robust OOS
   evaluation, it hasn't survived.*

3. **The surface is a system, informative about the present, not the future.**
   What the surface *does* contain is rich structure in the here-and-now: a
   forecastable volatility level, mean-reverting skew, and — the project's best
   novel result — a shock-propagation geometry that is anisotropic (information
   diffuses more coherently across maturities than strikes) and that tightens into
   crisis-time isotropy. The implied-vol surface is a beautifully-coupled dynamic
   object to *describe*; it is a poor crystal ball to *forecast* with, beyond its
   own level.

*The most valuable thing this project produced is not a discovery but a
disposition: to distrust the in-sample mirage, to test out-of-sample, and to
report the negative result the evidence demands.*

---

*Companion reports: [M1](RESEARCH_M1_IV_PREDICTS_RV.md) · [M2](RESEARCH_M2_SMILE_FORECASTING.md) · [M3](RESEARCH_M3_OPTION_RETURN_PREDICTABILITY.md) · [M4](RESEARCH_M4_HIGHER_ORDER_GREEKS.md) · [M5](RESEARCH_M5_GREEK_INFORMATION.md) · [M6](RESEARCH_M6_SURFACE_GEOMETRY.md) · [M7](RESEARCH_M7_SURFACE_DYNAMICS.md) · [M8](RESEARCH_M8_INFORMATION_PROPAGATION.md). All figures under `docs/figures/research_m*/`; all pipelines under `python/`; all datasets regenerable from the git-ignored `data/`.*
