# Final report — Algo Trading return regression

Horizon: **21 trading days**. Universe: **98 US large-caps** (100 configured, 2
unavailable). Panel: **483,869 rows**, 2006-01-17 → 2026-06-29.

Development holdout: 2023-05-31 → 2026-06-29. It is reported below but was not
used to select anything.

---

## 1. What changed

| Area | Before | After |
| --- | --- | --- |
| Trainers | `train.py` (LSTM) + `train_xgboost.py` importing from it | `train_lstm.py` and `train_xgboost.py`, both importing `src/training_common.py`; neither imports the other |
| Target | market-excess (`r − r_bench`) | unchanged — beta-neutral was implemented, tested on folds and **rejected** (see §3a); an explicit stage-1 market model was added |
| LSTM loss | Huber + whole-batch correlation | `0.40·MSE + 0.40·Huber + 0.20·(1 − per-date IC)` |
| Batching | random rows | whole date cross-sections |
| Checkpoint metric | `cross_sectional_ic` | `IC + 0.25·MSE-skill` (requires both) |
| XGBoost rounds | `max(50, best_iteration+1)` with `best_iteration = 0` | iteration-ladder evaluation on folds, **no floor** |
| Feature importance | one unused reference booster | mean gain over the actual bootstrap ensemble + permutation importance |
| Features | 53 | **104** |
| Calibration | single affine fit on validation | affine / ridge / isotonic + shrinkage, selected out of fold, rank-preservation guarded |
| Backtest | ~17–31% invested vs 100% buy-and-hold | fully invested top-k, turnover costs, **all 21 rebalance offsets**, sector- and beta-neutral variants |
| Baselines | 6 | **13** |
| Acceptance | none | 9 pre-registered gates |
| Tests | 63 | **176** |

---

## 2. Files created, renamed, deleted

**Renamed:** `train.py` → `train_lstm.py`.

**Created (12):** `src/training_common.py`, `src/losses.py`, `src/market_model.py`,
`src/panel_features.py`, `src/calibration.py`, `src/blending.py`, `src/boosting.py`,
`src/portfolio.py`, `src/acceptance.py`, `src/experiments.py`, `src/evaluation.py`,
`blend_models.py`.

**Created (tests):** `tests/test_trainer_refactor.py`, `tests/test_modeling_core.py`,
`tests/test_portfolio_and_blend.py`.

**Substantially rewritten:** `train_lstm.py`, `train_xgboost.py`, `src/dataset.py`,
`src/model.py`, `src/regression.py`, `src/features.py`, `src/pipeline.py`,
`src/uncertainty.py`, `predict.py`, `compare_results.py`, `README.md`,
`docs/modeling_methodology.md`.

**Deleted:** none (`train.py` was renamed, not removed).

---

## 3. XGBoost results

### Walk-forward (3 purged folds, selection data only)

| metric | mean | std | min | max |
| --- | --- | --- | --- | --- |
| cross-sectional IC | **+0.0403** | 0.0077 | +0.0350 | +0.0491 |
| ICIR | +0.7921 | 0.2370 | +0.5993 | +1.0567 |
| IC t-statistic | +1.3437 | 0.4018 | +1.0172 | +1.7924 |
| MSE | 0.00470 | 0.00110 | 0.00360 | 0.00580 |
| MAE | 0.04880 | 0.00710 | 0.04190 | 0.05610 |
| RMSE | 0.06790 | 0.00810 | 0.05990 | 0.07600 |
| direction accuracy | 0.5243 | 0.0192 | 0.5032 | 0.5409 |
| MSE skill vs historical mean | +0.0031 | 0.0010 | +0.0024 | +0.0043 |
| long-short spread (annualised) | +0.1322 | 0.0141 | +0.1166 | +0.1442 |

**IC positive in all 3 folds.** Previous run: +0.0374 ± 0.0101.

### Development holdout

IC **+0.0437**, ICIR +0.871, t-stat +1.524, IC > 0 on **62.2%** of dates,
long-short spread **+15.31%** annualised, direction accuracy 52.8%.

MSE **0.006865**, MAE **0.060157**, RMSE 0.082856, R² +0.0034.
**MSE skill vs historical mean +0.0079; MAE skill +0.0081** — both now positive,
where MAE previously sat at parity with the historical mean.

### 3a. Target mode: beta-neutral was implemented and rejected

A beta-weighted residual target (`r − β·r_bench`) is the theoretically better
choice and removes more variance (30.3% vs 27.7%). Compared against plain
market-excess on three purged folds:

| target | mean IC | std across folds | per-fold IC | selection score |
| --- | --- | --- | --- | --- |
| **market_excess** (shipped) | +0.0411 | **0.0039** | 0.0449 / 0.0375 / 0.0431 | **+0.0380** |
| beta_neutral_residual | +0.0411 | 0.0147 | 0.0243 / 0.0462 / 0.0523 | +0.0262 |

Mean skill is identical; the beta-weighted target is simply far less stable,
because the rolling 60-day beta is itself a noisy estimate and multiplying it
into the label injects that noise. The `mean − std` rule selects market-excess.

**No change was made to the shipped target.** The capability exists
(`regression_target.mode: "beta_neutral_residual"`) and the evidence says not to
use it. Artifact: `reports/experiments/experiment_target_mode_h21.md`.

### Boosting-round selection

Folds preferred **5, 40 and 160** rounds; the consistency rule selected **80**.
No minimum floor was applied (`minimum_round_floor_applied: false`). The
grid winner was `shallow_depth3` (`max_depth: 3`), selected on folds.

### Reproducibility

The whole XGBoost pipeline was run twice, independently, from the same seeds.
Every reported figure was identical: 80 rounds, `shallow_depth3` grid winner,
walk-forward IC +0.0403 ± 0.0077, holdout IC +0.0437, MSE skill +0.0079, MAE
skill +0.0081, 8/9 gates, portfolio Sharpe 1.555, and the same two-feature
permutation blocklist recommendation. Seeds, fold boundaries and library versions
are recorded in every experiment payload.

---

## 3b. LSTM results

### Walk-forward (3 purged folds)

| metric | mean | std | min | max |
| --- | --- | --- | --- | --- |
| cross-sectional IC | **+0.0327** | 0.0115 | +0.0214 | +0.0444 |
| ICIR | +0.6383 | 0.2435 | +0.3582 | +0.7994 |
| IC t-statistic | +1.0828 | 0.4128 | +0.6079 | +1.3559 |
| MSE | 0.00690 | 0.00180 | 0.00490 | 0.00800 |
| MAE | 0.06020 | 0.00840 | 0.05070 | 0.06680 |
| direction accuracy | 0.5284 | 0.0058 | 0.5228 | 0.5344 |
| MSE skill vs historical mean | **−0.0049** | 0.0147 | −0.0135 | +0.0121 |
| long-short spread (annualised) | +0.1343 | 0.0208 | +0.1190 | +0.1580 |

IC is positive in all three folds. Previous run: +0.0248 ± 0.0139, so the
walk-forward IC improved (+0.0327).

### Development holdout — and a clear generalisation gap

| metric | value |
| --- | --- |
| cross-sectional IC | **+0.0118** |
| ICIR | +0.252 |
| IC t-statistic | +0.441 |
| IC > 0 on | 55.2% of dates |
| long-short spread (annualised) | +9.79% |
| MSE / MAE | 0.008154 / 0.065706 |
| MSE skill vs historical mean | **−0.0370** |
| direction accuracy | 0.5049 |

**The LSTM's holdout IC (+0.0118) is well below its walk-forward IC (+0.0327).**
That gap is the headline finding for this model, and it is not flattering: the
2023–2026 period was materially harder for it than any of the earlier folds. On
the holdout it also fails to beat simple baselines — sector-relative momentum
reaches +0.0211 and a ridge regression +0.0361 on identical rows.

### Acceptance gates: 6 of 9

| gate | LSTM | XGBoost |
| --- | --- | --- |
| walk-forward mean IC > 0.03 | PASS | PASS |
| IC positive in every fold | PASS | PASS |
| IC t-statistic > 2.0 | **FAIL** (+1.08) | **FAIL** (+1.34) |
| MSE/RMSE/MAE beat historical mean | **FAIL** | PASS |
| positive information ratio after costs | PASS | PASS |
| positive spread in every fold | PASS | PASS |
| stable calibration across folds | PASS | PASS |
| no material regime collapse | **FAIL** | PASS |
| interval coverage and width | PASS | PASS |

The LSTM fails the magnitude gate outright — its MSE skill is negative, meaning
its reported percentages are worse than predicting the training-window mean. For
a product that reports an expected percentage return, that is the more serious of
the two extra failures.

### Where the LSTM is genuinely better

Two things go the other way, and both are real:

| metric | LSTM | XGBoost |
| --- | --- | --- |
| portfolio Sharpe (mean over 21 offsets) | **+1.765** | +1.555 |
| portfolio IR vs universe (mean) | **+0.898** | +0.770 |
| interval coverage (nominal 0.80) | **0.820** | 0.785 |
| **does confidence filtering help?** | **yes, +0.0206** | no, −0.0165 |
| selected `min_z_score` | **0.20** | 0.00 |

The last two rows are the interesting ones. For the LSTM the uncertainty estimate
**does** carry decision-relevant information: filtering by confidence raises IC,
and validation tuning selected a non-zero `min_z_score`, so the risk-adjusted
rule is actually doing work. For XGBoost neither is true. The two families
disagree about whether their own uncertainty is useful, and the framework
detected that separately for each rather than assuming one answer.

The portfolio Sharpe advantage should be read cautiously: the top-20 portfolio
only depends on the extreme top of the ranking, and with a holdout IC of +0.0118
the LSTM's *overall* ordering is weak. A better Sharpe on a weaker IC is more
likely to be the top of the book behaving well over one period than a durable
edge.

### Verdict

**XGBoost is the better model on this data**, on walk-forward IC (+0.0403 vs
+0.0327), holdout IC (+0.0437 vs +0.0118), magnitude skill (positive vs
negative), and acceptance gates (8/9 vs 6/9). The LSTM is retained because the
comparison is the point of the project and because its uncertainty estimate is
the more useful of the two — not because it is competitive on accuracy.

---

## 3c. Loss selection: the composite objective was validated, not assumed

Phase 2 required the four loss variants to be compared on purged walk-forward
folds rather than chosen by argument. They were, at an identical 5-epoch budget
per candidate on identical folds:

| loss | fold 1 | fold 2 | fold 3 | mean IC | std | selection score |
| --- | --- | --- | --- | --- | --- | --- |
| **MSE + Huber + date-IC** (shipped) | +0.0195 | +0.0573 | +0.0344 | **+0.0371** | 0.0214 | **+0.0148** |
| pure MSE | +0.0163 | +0.0539 | +0.0324 | +0.0342 | 0.0222 | +0.0113 |
| MSE + Huber | +0.0134 | +0.0572 | +0.0302 | +0.0336 | 0.0255 | +0.0067 |
| pure Huber | +0.0040 | +0.0545 | +0.0153 | +0.0246 | 0.0299 | −0.0089 |

**The shipped configuration wins, and wins on every fold** — it has the highest
IC in folds 1 and 3 and ties fold 2. It is also the *most stable* (std 0.0214,
the lowest of the four).

Three things worth reading off this table:

1. **The date-IC term is what earns the win.** Adding it to MSE + Huber lifts the
   selection score from +0.0067 to +0.0148 — the largest single improvement in
   the table. That is the term matching the training objective to the metric the
   model is evaluated on.
2. **Pure Huber is clearly the worst**, and the only candidate with a negative
   selection score. Capping outliers without any magnitude or ranking pressure is
   not enough on its own.
3. **Pure MSE is a surprisingly strong second.** MSE alone beats MSE + Huber
   here. This is a useful corrective to the intuition that robust losses are
   automatically better on fat-tailed data — the evidence says Huber only helps
   once the ranking term is present to stop the collapse it would otherwise allow.

**Cost:** the composite is about 5× slower per fold (700s versus 138s for the
other three), because the per-date grouped correlation is computed on every
batch. It is worth it here, but the trade is real and should be stated.

**Caveat:** this comparison ran at a 5-epoch cap so that four candidates × three
folds would finish in reasonable time. The ranking is like-for-like — every
candidate got an identical budget and identical folds — but the ordering at 40
epochs was not verified, and the IC term's benefit could plausibly grow or shrink
with a longer budget. Artifact:
`reports/experiments/experiment_lstm_loss_h21.md`.

---

## 4. Baselines (development holdout, identical rows, component space)

| forecast | IC | MSE | MAE | direction |
| --- | --- | --- | --- | --- |
| **XGBoost** | **+0.0437** | **0.006865** | **0.060157** | 0.5276 |
| ridge regression (same features) | +0.0361 | 0.006903 | 0.060560 | 0.4970 |
| sector-relative momentum | +0.0211 | 0.011530 | 0.079400 | 0.5102 |
| excess momentum | +0.0195 | 0.012688 | 0.083610 | 0.5142 |
| residual momentum | +0.0159 | 0.011708 | 0.080610 | 0.5080 |
| sector historical mean | +0.0100 | 0.006922 | 0.060680 | 0.4820 |
| momentum | +0.0076 | 0.032861 | 0.132050 | 0.4954 |
| ticker historical mean | +0.0018 | 0.006962 | 0.060870 | 0.4806 |
| zero return | 0.0000 | 0.006889 | 0.060200 | — |
| historical mean | 0.0000 | 0.006920 | 0.060650 | 0.4786 |
| rolling historical mean | 0.0000 | 0.006909 | 0.060350 | 0.4860 |
| market drift | 0.0000 | 0.006943 | 0.060870 | 0.4786 |
| reversal | −0.0076 | 0.033956 | 0.132000 | 0.5046 |

The model beats every baseline on IC, MSE and MAE simultaneously.

**Honest caveat:** a plain ridge regression on the same features reaches
**+0.0361 IC**, against the boosted trees' +0.0437. Most of the available signal
is linear; the non-linear model adds roughly 0.008 IC. That is a real gain but a
modest one, and it should temper any claim that gradient boosting is essential
here.

---

## 5. Portfolio backtest (fully invested top-20, all 21 rebalance offsets)

| metric | mean | median | worst | best |
| --- | --- | --- | --- | --- |
| total return | +134.4% | +132.7% | +109.7% | +152.4% |
| annualised return | +32.0% | +32.5% | +28.0% | +35.0% |
| Sharpe | **+1.555** | +1.525 | +1.341 | +1.777 |
| information ratio vs universe | **+0.770** | +0.803 | +0.432 | +0.970 |
| excess return vs universe | +52.5% | +52.4% | +28.3% | +68.3% |
| max drawdown | −15.2% | −14.8% | −21.3% | −8.2% |
| annualised turnover | 9.03× | 8.98× | 8.30× | 9.57× |

Every offset is profitable and beats the equal-weight universe; the worst
calendar still delivers IR +0.432. Both the strategy and the benchmark are 100%
invested, so this is a like-for-like comparison — unlike the previous report.

**Neutral variants:**

| variant | Sharpe (mean) | IR vs universe (mean) |
| --- | --- | --- |
| long-only top-20 | +1.555 | +0.770 |
| **sector-neutral** | **+1.887** | **+1.084** |
| beta-neutral | +1.592 | +0.827 |

Sector-neutralisation *improves* the result. That is the strongest single piece
of evidence in this report that the edge is genuine **stock selection** rather
than a standing bet on a sector: removing the sector tilt makes it better, not
worse.

---

## 5b. Blending the two families

Weights fitted on **124,506 / 124,651 out-of-fold rows**, inner-joined on
`(ticker, date)` to 124,506 rows across three purged folds. Constraints:
non-negative, summing to one, fitted on out-of-fold data only.

**Fitted weights: LSTM 0.30 / XGBoost 0.70** — not equal, and not the pure model.

Out-of-fold selection scores (`mean − std` across folds, objective
`0.5·MSE-skill + 0.5·normalised IC`):

| weights (LSTM / XGB) | selection score | mean | std |
| --- | --- | --- | --- |
| 0.30 / 0.70 | **+0.6598** | +0.8086 | 0.1488 |
| 0.35 / 0.65 | +0.6548 | +0.7776 | 0.1228 |
| 0.25 / 0.75 | +0.6544 | +0.8343 | 0.1799 |
| 0.00 / 1.00 (XGBoost alone) | +0.4310 | | |
| 1.00 / 0.00 (LSTM alone) | +0.3432 | | |

The blend beat the better standalone model by **+0.229** out of fold, comfortably
above the 0.010 retention bar, so it was retained.

### The holdout disagrees, and that is the finding

| model | IC | ICIR | MSE | MSE skill |
| --- | --- | --- | --- | --- |
| LSTM | +0.0118 | +0.252 | 0.007955 | −0.0116 |
| XGBoost | **+0.0437** | **+0.871** | 0.007802 | +0.0078 |
| Blend (0.30 / 0.70) | +0.0205 | +0.430 | **0.007784** | **+0.0101** |

The blend has the **best MSE and the best magnitude skill of the three**, but its
IC (+0.0205) is less than half XGBoost's (+0.0437).

This is worth stating plainly rather than smoothing over. The blend was selected
on out-of-fold evidence that was genuinely favourable, and on the holdout that
evidence turned out to be misleading for *ranking* — because the LSTM's own
holdout IC collapsed from +0.0327 (folds) to +0.0118. Mixing 30% of a component
that degraded in the final period dragged the combined ranking down with it,
even though averaging two independent error sources still improved squared error
exactly as the theory predicts.

Two honest conclusions:

1. **Out-of-fold selection is not a guarantee.** It is the best available
   protection against selecting on the test set, and it is still only as reliable
   as the assumption that the future resembles the folds. Here that assumption
   held for magnitude and broke for ranking.
2. **For ranking, XGBoost alone remains the better choice on this data.** The
   blend is reported and its artifact saved, but nothing in this report claims it
   improved stock selection, because it did not.

The blend weights were frozen before the holdout was scored; the holdout was
never used to choose them.

---

## 6. Calibration

Isotonic with shrinkage 1.0 won on the folds, but was **rejected when re-fitted
on the validation window** by the rank-preservation guard, so the identity
transform shipped. The report records both facts
(`fold_selection_overridden_on_validation: true`) rather than claiming a
calibration is active that is not.

Decile table (development holdout, total return):

| decile | n | predicted | realised | std err | hit rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 7,565 | +0.36% | +1.14% | 0.08% | 55.9% |
| 5 | 7,565 | +0.63% | +1.20% | 0.09% | 55.1% |
| 8 | 7,564 | +0.86% | +2.36% | 0.10% | 60.6% |
| 9 | 7,565 | +1.03% | +2.81% | 0.11% | 60.7% |
| 10 | 7,565 | +1.66% | +2.55% | 0.16% | 56.3% |

Rank correlation between predicted and realised decile means: **0.903**. The
fitted slope is **1.40**, i.e. realised spread is *wider* than predicted — the
model is under-confident in magnitude, not over-confident. Decile 10 breaks
monotonicity (realised +2.55% below decile 9's +2.81%), so the very top of the
book is not reliably the best part of it.

---

## 7. Uncertainty — including where it does not help

| metric | value |
| --- | --- |
| coverage (PICP), nominal 80% | 0.785 |
| mean width (MPIW) | 20.15% |
| normalised width | 2.43 |
| Winkler score | 0.298 |
| worst conditional coverage error — by volatility | **0.136** |
| worst conditional coverage error — by magnitude | 0.049 |
| epistemic share of total variance | 0.0011 |

Marginal coverage is close to nominal (78.5% vs 80%). **Conditional coverage is
not**, and the table shows the failure mode exactly:

| volatility bucket | n | coverage | error | mean width |
| --- | --- | --- | --- | --- |
| 1 (calmest) | 15,130 | 0.664 | **−0.136** | 10.6% |
| 2 | 15,129 | 0.731 | −0.069 | 14.4% |
| 3 | 15,129 | 0.792 | −0.008 | 17.9% |
| 4 | 15,129 | 0.844 | +0.044 | 22.5% |
| 5 (most volatile) | 15,130 | 0.896 | **+0.096** | 35.4% |

This is the textbook case for why marginal coverage is not enough: the interval
is **too narrow for calm stocks and too wide for volatile ones**, and the two
errors cancel to a respectable-looking 78.5% overall. A single global conformal
multiplier cannot fix a miscalibration that changes sign across the volatility
distribution — that is precisely what Mondrian (per-regime) conformal
calibration is for, and it is implemented in `src/uncertainty.py`.

The filter test is equally clear:

| keep | rows | IC | direction accuracy |
| --- | --- | --- | --- |
| 100% | 75,647 | **+0.0437** | 0.5276 |
| 75% | 56,735 | +0.0405 | 0.5329 |
| 50% | 37,824 | +0.0341 | 0.5298 |
| 25% | 18,912 | +0.0272 | 0.5344 |

IC falls **monotonically** as the filter tightens. Confidence, as this system
measures it, is anti-correlated with ranking skill. (Directional accuracy rises
slightly, from 52.8% to 53.4%, so sigma is not completely uninformative — but it
does not help the metric the model is actually selected on.)

Two findings reported plainly rather than buried:

1. **Filtering by confidence does not help.** Keeping only the most confident
   forecasts *reduced* IC by 0.017. The `|prediction| / sigma` ordering carries
   no useful decision information on this data.
2. **The risk-adjusted decision rule collapsed to the point rule.** Validation
   tuning selected `min_z_score = 0.00`, meaning the uncertainty input to the
   decision is doing nothing.

One thing the interval does get right: it responds to the volatility *state*. A
live inference run on 2026-07-30 returned a ±15pp band for AAPL (20-day realised
volatility 1.83% daily) and a ±31pp band for MSFT (4.07% daily, after a single
17% move inside the window). The band is not a fixed width dressed up as a
forecast — it tracks the conditional dispersion of the underlying, which is the
aleatoric component doing its job.

Together these say the uncertainty estimate is worth reporting — it honestly
communicates how wide the outcome distribution is, and the epistemic share of
0.0011 correctly says almost all of that width is irreducible market noise — but
it is **not earning its keep as a filter**. The decision layer could be
simplified to the point rule with no measured loss.

---

## 8. The market model: a negative result

Fitted on one validation window, the stage-1 market model wanted a shrinkage of
**0.741** on the strength of **+0.141 validation MSE skill**. That same setting
scored **−0.111 MSE skill on the test period**.

Chosen instead by fold agreement, the per-fold optima were **0.000, 0.228,
0.000** → mean 0.076, std 0.132 → **selected 0.000**.

The market leg therefore reduces to the constant historical drift exactly
(verified: test MSE identical to the drift baseline, skill +0.0000). **A
month-ahead index forecast from these features does not generalise.** The
hierarchical machinery is implemented, tested and working; it reports that this
component is not forecastable, and the shrinkage rule stops that non-skill from
contaminating every stock's forecast.

---

## 9. Acceptance gates — XGBoost 8/9, LSTM 6/9

| gate | XGBoost | LSTM |
| --- | --- | --- |
| walk-forward mean IC > 0.03 | **PASS** (+0.0403) | **PASS** (+0.0327) |
| IC positive in every fold | **PASS** (3/3) | **PASS** (3/3) |
| IC t-statistic > 2.0 | **FAIL** (+1.52) | **FAIL** (+0.44) |
| MSE, RMSE, MAE beat historical mean | **PASS** | **FAIL** |
| positive information ratio after costs | **PASS** (+0.770) | **PASS** (+0.898) |
| positive top-minus-bottom spread every fold | **PASS** (3/3) | **PASS** (3/3) |
| stable calibration across folds | **PASS** | **PASS** |
| no material regime collapse | **PASS** | **FAIL** |
| interval coverage and width | **PASS** | **PASS** |
| **total** | **8 / 9** | **6 / 9** |

**The t-statistic gate fails for both models and is reported as failed.** With
overlap-discounted effective sample size the aggregate IC t-stat is +1.52 for
XGBoost and +0.44 for the LSTM, against a required 2.0. The XGBoost edge is real
but not yet statistically decisive at this sample length; the LSTM's holdout edge
is not established at all.

Neither was tuned towards. Doing so would have destroyed the only thing a
pre-registered gate is for.

---

## 10. Feature importance

Averaged over the 15 bootstrap members actually used for inference:

| feature | mean gain | std across members | members using it | new? |
| --- | --- | --- | --- | --- |
| `amihud_illiquidity_20d` | 0.0231 | 0.0048 | 15/15 | **new** |
| `idiosyncratic_volatility_20d_z252` | 0.0213 | 0.0056 | 15/15 | |
| `idiosyncratic_volatility_20d` | 0.0213 | 0.0042 | 15/15 | |
| `volatility_60d` | 0.0206 | 0.0051 | 15/15 | |
| `xs_rank_price_to_sma_50` | 0.0203 | 0.0044 | 15/15 | **new** |
| `sector_beta_60d` | 0.0200 | 0.0028 | 15/15 | **new** |
| `sector_relative_volatility_20d` | 0.0200 | 0.0086 | 14/15 | **new** |
| `dollar_volume_trend_20d` | 0.0197 | 0.0030 | 15/15 | **new** |

**The feature work paid for itself.** 12 of the top 20 features are Phase 7
additions, and the new features account for **55% of total gain**. The single
most important feature is a new one — Amihud illiquidity, the price move required
to absorb a unit of volume. Sector features, cross-sectional ranks, liquidity
measures and regime interactions are all represented in the top 20.

The dispersion column matters too: `dispersion_x_volatility` has a high mean gain
but a standard deviation of 0.0098 across members and is used by only 13 of 15,
so it is being relied on by a minority of the ensemble rather than being a stable
signal. That is exactly what a single reference model's importance would have
hidden.

Out-of-fold permutation importance flagged **2 features as harmful in ≥2 folds**:
`high_low_range_20d` and `regime_x_beta`. This is a **recommendation only** and
was deliberately not applied automatically — dropping features on a noisy
estimate is its own form of overfitting. To act on it, add them to
`feature_blocklist` in `configs/config.yaml`.

---

## 11. Tests and verification

```text
python3 -m unittest discover -s tests   ->  Ran 176 tests   OK
python3 -m compileall -q .              ->  exit 0
git diff --check                        ->  exit 0
streamlit run app.py (headless)         ->  HTTP 200, 0 error lines
```

Additional end-to-end verification:

* both saved artifacts reload for inference at schema 4, with their calibration,
  market model and frozen decision rule intact;
* `compare_results.py` regenerates the head-to-head table, the baseline table and
  the per-fold walk-forward table from the saved payloads;
* the experiment runner was exercised on a controlled candidate set and produced
  JSON, CSV and Markdown with fold boundaries, seeds and library versions — and
  its `mean − std` rule demonstrably changed the ranking (a candidate with a
  slightly lower mean but half the dispersion won);
* the XGBoost pipeline was run twice from the same seeds and reproduced every
  reported figure exactly.

Test coverage added for: trainer split and import direction, absence of obsolete
`train.py` references, both CLIs, MSE in every payload, the selection score, the
composite loss and each preset, per-date grouped correlation (including equal
date weighting), date-grouped batching, recurrent dropout, auxiliary heads,
beta-neutral target and total-return reconstruction, market-model degeneration,
round selection with no floor, calibration rank-preservation, cross-sectional
centring, decile reports, purged splits, top-k exposure and turnover costs,
multi-offset backtesting, blend weight constraints, `(ticker, date)` alignment,
point-in-time feature construction, conditional coverage, Mondrian calibration,
filter-benefit, acceptance gates, and inference compatibility.

Smoke training: both families were run end-to-end at reduced settings before the
full runs.

**Three real bugs were found by these tests and fixed:**

1. `MarketReturnModel.raw_predict` raised inside a matmul when a metadata payload
   named features but carried no standardisation moments.
2. The calibration rank guard compared against the *outcome*, so when the raw
   forecast was negatively correlated in the fitting window, collapsing to a
   constant looked like an improvement and was accepted. It now tests
   monotonicity against the model's own output.
3. `sort_index()` in the panel stage was not stable, so with ~100 duplicate index
   labels per date the within-date row order was non-deterministic between runs.

A fourth issue was found while wiring inference: cross-sectional centring applied
to a single-ticker request would return exactly **zero**, silently destroying
every single-stock forecast. The calibration now falls back to a stored centring
offset when no cross-section is available.

---

## 12. Known limitations

* **Survivorship bias.** The universe is 100 companies large *today*, held fixed
  across 2006–2026. Names that failed or were acquired are absent, so returns are
  biased upward. Read the numbers as relative comparisons on identical data, not
  as an achievable live return.
* **IC t-statistic 1.52 < 2.0.** The edge is not yet statistically decisive.
* **A linear ridge reaches +0.0361 IC** against XGBoost's +0.0437 — most of the
  signal is linear. On the holdout the LSTM (+0.0118) does **not** beat the ridge
  baseline, or even sector-relative momentum (+0.0211).
* **The LSTM does not generalise as well as its folds suggested** — walk-forward
  IC +0.0327 against holdout IC +0.0118 — and it fails the magnitude gate with
  negative MSE skill.
* **The blend was retained out of fold but did not improve ranking on the
  holdout** (IC +0.0205 against XGBoost's +0.0437), because it carries 30% of the
  component that degraded. It does improve MSE and magnitude skill.
* **Conditional coverage by volatility misses by 0.136** despite correct marginal
  coverage; Mondrian calibration is implemented but not yet the default.
* **Uncertainty does not improve decisions** (filtering costs 0.017 IC; tuned
  `min_z_score = 0`).
* **Decile 10 is not the best decile** — the extreme top of the book is unreliable.
* **The market leg is not forecastable** at this horizon from these features.
* The backtest ignores market impact, order-book dynamics and borrow costs.
* Fundamental/earnings features remain disabled: only a present-day snapshot is
  available, which would introduce hindsight bias.
* **The loss matrix was run** and the shipped composite objective won on all
  three folds (§3c), but at a 5-epoch cap rather than the full 40-epoch budget.
  The **architecture matrix** (`--architecture-experiment`: sequence length,
  hidden size, depth, dropout variants) is implemented and wired but was **not
  run** — eleven variants × three folds at a comparable budget is several hours
  on this hardware. The shipped architecture is therefore the documented default,
  not a validated optimum.

---

## 13. Exact commands

**Retrain:**

```bash
python3 train_lstm.py    --horizon 21 --walk-forward
python3 train_xgboost.py --horizon 21 --walk-forward
python3 train_all_models.py --horizon 21 --compare --walk-forward
```

**With the full selection machinery:**

```bash
python3 train_xgboost.py --horizon 21 --walk-forward --grid-search --permutation-importance
python3 train_lstm.py    --horizon 21 --walk-forward --loss-experiment --architecture-experiment
```

**Blend and compare:**

```bash
python3 blend_models.py     --horizon 21
python3 compare_results.py  --horizon 21
```

**Launch the website:**

```bash
python3 -m streamlit run app.py
```

Then open <http://localhost:8501>.

---

_Academic research and simulation only. Not financial advice._
