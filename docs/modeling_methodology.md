# Modelling methodology

This document explains *why* the system is built the way it is. Every design
choice below replaced something that measurably did not work, and the reasoning
matters more than the code.

---

## 1. The problem, stated honestly

Predicting a stock's percentage return one month ahead is close to
unforecastable. On a panel of ~100 large US names the realistic ceiling for a
technical/statistical feature set is a cross-sectional information coefficient of
roughly 0.02–0.05. Anything much larger should be treated as a bug or a leak
rather than a discovery.

That has two consequences that drive the whole design:

1. **A constant forecast is a strong baseline.** Predicting the training-window
   mean for every stock achieves nearly the variance of the target. Any
   optimisation criterion that rewards squared error alone will drive the model
   towards that useless optimum.
2. **Most of the target is not the model's to predict.** The market move common
   to every stock dominates the variance of a pooled panel, and no indicator
   computed on one stock tells you much about next month's index return.

---

## 2. Target: hierarchical decomposition

### What it is

```text
expected_stock_return = beta * expected_market_return   <- one number per date
                      + expected_sector_return          <- not separately forecast
                      + expected_stock_residual         <- what the model learns
```

The user-facing output is the total. Internally the model only learns the
residual leg, and the market leg is supplied by a separate stage-1 model.

### Beta-neutral residual: implemented, measured, and not selected

The theoretical argument for a beta-weighted target is sound. Plain
`future_return − benchmark_future_return` assumes every stock has beta 1, which
is false: for a high-beta semiconductor name it leaves market exposure inside the
"excess" target, and for a low-beta utility it over-subtracts and inserts
exposure with the opposite sign. The beta-weighted residual
`future_return − beta × benchmark_future_return` (rolling beta observable at the
prediction date, clipped to `[0, 3]`) is fully implemented as
`regression_target.mode: "beta_neutral_residual"`, and it does remove more
variance: **30.3%** against **27.7%**.

It was then compared against plain market-excess on three purged folds, and it
**lost**:

| target | mean IC | std across folds | per-fold IC | selection score |
| --- | --- | --- | --- | --- |
| **market_excess** (shipped) | +0.0411 | **0.0039** | 0.0449 / 0.0375 / 0.0431 | **+0.0380** |
| beta_neutral_residual | +0.0411 | 0.0147 | 0.0243 / 0.0462 / 0.0523 | +0.0262 |

The mean skill is identical to four decimal places. The difference is entirely
**stability**: the beta-weighted target swings from +0.024 to +0.052 across folds
while market-excess stays within +0.037 to +0.045.

The explanation is that removing variance from a target is not automatically
useful. The rolling 60-day beta is itself a noisy estimate, and multiplying it
into the label injects that estimation noise into every training row. What looks
like a cleaner target is partly a noisier one, and the `mean − std` rule
correctly refuses to buy variance reduction at the price of instability.

This is recorded here rather than quietly dropped because it is the clearest
example in the project of the framework doing its job: a better-motivated idea
was implemented in full, measured honestly, and rejected on the evidence.
See `reports/experiments/experiment_target_mode_h21.md`.

### Why the sector leg is reported as zero

Sector information enters through features — sector composites, sector beta,
sector-relative momentum and volatility — rather than as a separately forecast
term. The leg is reported as `0.0` so the composition remains an exact identity.
Reporting a fabricated sector forecast would be worse than reporting none.

### Stage 1: the market-return model, and why it earns almost no authority

A ridge regression on date-level market-state features (breadth, dispersion,
average correlation, volatility regime, benchmark drawdown, VIX, treasury yield,
USD index) predicts the benchmark's forward return, with **one observation per
date** — about 3,600 training points, not 350,000 copies of the same 3,600.

Its prediction is shrunk towards the training-window historical drift:

```text
prediction = drift + shrinkage * (ridge_prediction - drift)
```

The shrinkage weight is **not** fitted on one validation window. Doing that gave
a weight of 0.74 on the strength of +0.14 validation MSE skill — and that same
setting then scored **−0.11 MSE skill on the test period**. The fit was a
property of the window, not of the market.

The weight is instead chosen from agreement across purged folds:

```text
shrinkage = clip(mean(fold optima) - std(fold optima), 0, cap)
```

**Measured outcome:** fold optima were 0.000, 0.228, 0.000 → mean 0.076, std
0.132 → **selected weight 0.000**. The market model therefore contributes
nothing, and the composition reduces to the constant historical drift exactly
(verified: test MSE identical to the drift baseline, skill +0.0000).

This is a real negative result and it is reported as one: **a month-ahead index
forecast from these features does not generalise.** The machinery is implemented
and validated; it simply reports that the market leg is not forecastable, and the
shrinkage rule prevents that non-skill from contaminating every stock's forecast.

---

## 3. Objective: why MSE alone is wrong, and why IC alone is also wrong

### The composite loss

```text
total = 0.40 * MSE + 0.40 * Huber + 0.20 * (1 - mean per-date correlation)
```

| term | job | what goes wrong without it |
| --- | --- | --- |
| MSE | makes the magnitude meaningful | the output is a rank, not a percentage return |
| Huber | caps extreme return events | a handful of crash days dominate every batch on a fat-tailed panel |
| per-date IC | rewards correct ordering | the model collapses to the constant-mean solution |

MSE genuinely contributes to the gradient. It is not computed after the fact and
printed — the product promises an expected percentage return, so magnitude has to
be optimised, not merely measured.

**This weighting was measured, not argued.** All four variants were compared on
three purged folds at an identical epoch budget:

| loss | mean IC | std | selection score |
| --- | --- | --- | --- |
| **MSE + Huber + date-IC** (shipped) | **+0.0371** | **0.0214** | **+0.0148** |
| pure MSE | +0.0342 | 0.0222 | +0.0113 |
| MSE + Huber | +0.0336 | 0.0255 | +0.0067 |
| pure Huber | +0.0246 | 0.0299 | −0.0089 |

The composite wins on every fold and is also the most stable. Adding the date-IC
term to MSE + Huber is the single largest improvement in the table (+0.0067 →
+0.0148), which is the clearest available evidence that matching the training
objective to the evaluation metric is what does the work. Pure Huber is the only
candidate with a negative score — capping outliers is not sufficient by itself.
Notably pure MSE beats MSE + Huber, so Huber only earns its place once the
ranking term is present.

### Why the correlation term is *per date*

The reported headline metric is the mean per-date rank correlation. A correlation
computed over a batch of randomly mixed rows measures something different: it is
inflated by the market factor common to all names on a date, so a model that only
tracks "the market was up that day" scores well on it while adding no
cross-sectional skill. Grouping by date makes the training objective measure the
same thing as the evaluation metric.

Each qualifying date contributes **equally**, regardless of how many names it
holds. Weighting by group size would let the few dates holding the full universe
dominate the gradient, and the evaluation metric averages dates, not rows.

Pearson correlation within each date is the differentiable surrogate for the
Spearman IC that is reported; ranks are not differentiable.

### Date-grouped batching

The ranking term needs a real cross-section to work with. Sampling rows
independently scatters each date across many batches and leaves two or three
names per date per batch, from which no ordering can be learned. Batches are
therefore built from **whole date cross-sections**; only the order of dates is
shuffled each epoch, so batches stay de-correlated.

### Why early stopping is not based on MSE

Selecting on MSE alone rewards the constant forecast — earlier runs of this
project peaked at **epoch 1** every single time for exactly this reason.
Selecting on IC alone has the opposite failure: a model that orders stocks
correctly while emitting wildly mis-scaled magnitudes.

The checkpoint criterion requires both:

```text
validation_selection_score = cross_sectional_ic + 0.25 * mse_skill_vs_historical_mean
```

**This is not theoretical.** In the shipped run the LSTM's validation IC kept
climbing to +0.0573 at epoch 9 while its MSE skill fell to **−0.0999** — i.e. its
reported percentages became *worse than predicting the historical mean* even as
its ranking improved. Pure-IC selection would have shipped that checkpoint. The
combined criterion peaked earlier, where both properties held.

The magnitude term is skill against the *training-window* mean rather than R²,
because R² measures against the evaluation set's own mean, which no forecaster
could have known in advance.

---

## 4. Validation: the test set is a development holdout

The test period has been inspected repeatedly across the life of this project.
Every look leaks a little information into the next design decision, and after
enough looks "test performance" measures the analyst as much as the model.

It is therefore treated as a **development holdout**: reported, never used to
select anything. In code, each trainer computes

```python
history = full_df[full_df.index <= validation_df.index.max()]
```

and every selection decision — features, hyperparameters, loss, boosting rounds,
calibration, blend weights, uncertainty calibration, decision thresholds — runs on
that slice only.

### Purging

Splits are separated by a gap of the full horizon. Without it a training row
dated just before a boundary carries a label built from prices inside the next
segment, which leaks the answer.

### The selection rule

Candidates are ranked by **`mean − std` across folds**, not by mean alone. A
configuration that wins on average while swinging between folds has not
demonstrated an edge; it has demonstrated sensitivity to the period. The same
rule selects the loss, the architecture, the boosting rounds, the calibration
family, the market-model shrinkage and the blend weights.

---

## 5. Features

104 stock-level features, up from 53. The additions are:

* **Sector**: composite returns (5/20/60d), sector volatility, rolling sector
  beta, sector-relative momentum and volatility, sector residual momentum.
* **Beta-neutral residual momentum** (20d, 60d).
* **Cross-sectional ranks by date** for 13 core factors — scale-free, immune to
  the level drift that makes a raw momentum number mean different things in 2009
  and 2021, and robust to the outliers a fat-tailed panel produces.
* **Liquidity**: dollar-volume z-score and trend, Amihud illiquidity,
  high–low range.
* **Regime interactions**: momentum × volatility regime, beta × regime,
  momentum × dispersion, beta × average correlation, and others.

### Sector composites from the universe, not sector ETFs

XLC launched in June 2018 and XLRE in October 2015. ETF-based sector returns
would be missing for a third of the sample and would silently delete every
pre-2018 row for the communication-services names when incomplete rows are
dropped. A composite of the actual universe exists from the first date and is the
more precise benchmark for these particular stocks. Sectors with fewer than three
members fall back to the whole-universe composite — a one-stock sector would make
sector-relative momentum identically zero, which is a constant, not a feature.

### Market-wide features are used, but not as direct stock inputs

Features identical for every ticker on a date (`benchmark_*`, `macro_*`,
`marketstate_*`) carry no cross-sectional ranking information by themselves, and
they let a sequence model recognise *which date* a window came from and memorise
that date's noise. Measured effect on this panel: including them cost roughly
0.05 of test IC and drove validation error up from epoch 1.

They are not discarded. They are used in two places where they cannot leak a date
fingerprint into a per-stock ranking:

1. the **market-return model**, which predicts one number per date and is
   therefore entitled to date-level features;
2. **regime interactions**, whose stock-specific leg makes them vary across the
   cross-section — this is what lets the model express "momentum works
   differently when dispersion is high", a conditional structure that a purely
   additive feature set cannot represent.

### Point-in-time guarantee

Every engineered feature is computed from information available at or before its
own row's date. This is enforced by a test that recomputes the entire panel stage
on a truncated prefix of history and requires every overlapping value to be
bit-for-bit identical — the strongest available check for look-ahead.

Fundamental and earnings features remain **disabled**: the available source is a
present-day snapshot, not a point-in-time historical feed, and using it would
introduce hindsight bias.

---

## 6. XGBoost model selection

### The bug that was fixed

The previous trainer early-stopped on pooled Spearman correlation, read back
`best_iteration`, then did:

```python
tuned_rounds = max(50, best_iteration + 1)
```

The run reported `best_iteration = 0` — the validation objective never improved
after the first tree — and the code silently overrode that with 50 trees. The
number recorded in the metadata described nothing that had happened. Separately,
pooled Spearman is the wrong objective: it mixes "did the market rise?" with
"which stock beat which?".

### What replaced it

Boosters are fitted once to a generous number of rounds, then **evaluated at a
ladder of iteration counts** (`1, 5, 10, 20, 40, 80, 160, 320, 640`) using
`iteration_range`, scored by per-date cross-sectional IC plus magnitude skill on
purged folds. One fit yields the entire curve.

**Measured:** the three folds preferred **5, 40 and 160** rounds respectively —
a spread that no single fixed floor could have represented. The consistency rule
selected **80**. No floor is applied.

Feature importance is averaged over the **actual bootstrap ensemble used for
inference**, with dispersion across members, replacing a throwaway reference
model that was never used to predict anything. Out-of-fold **permutation
importance** is also computed, shuffling values *within each date* so the
permuted column keeps that day's cross-sectional distribution and only the
assignment to stocks is destroyed.

---

## 7. Calibration

The models produce reliable ordering skill and weak magnitude skill. Calibration
closes that gap under four rules.

### Rule 1: it may not destroy the ranking

A least-squares fit on a low-signal panel will return a near-zero or negative
slope, because flattening every prediction genuinely improves squared error — and
deletes the only thing the model produced.

The guard tests monotonicity **with respect to the model's own output**, not
agreement with the outcome. This distinction matters and was found by a test:
when the raw forecast happens to be *negatively* correlated with the truth in the
fitting window, collapsing everything to a constant *improves* agreement with the
outcome, so an outcome-based guard waves the collapse through as an upgrade.
Requiring rank correlation ≈ 1 against the raw forecast makes a calibration what
it is supposed to be — a monotone rescaling — and rejects anything else
regardless of how it scores.

### Rules 2–4

* **No common intercept on the residual leg**, and **cross-sectional centring**
  so the alpha sums to roughly zero across the universe. A level on a given date
  is a market call, and the market call belongs to stage 1; leaving it in the
  residual would count it twice.
* **Shrinkage towards zero** — the honest default for an alpha forecast is "no
  view", not "the average view".
* **Affine, ridge and isotonic compared out of fold**, selected by the same
  `mean − std` consistency rule.

### The decile table

`reports/decile_calibration_*.csv` reports mean predicted return, mean realised
return, count, directional hit rate and standard error per predicted decile. This
is the single most revealing diagnostic for a return model: a model with real
magnitude skill produces a realised column near the predicted column, while a
model with ranking skill but no magnitude skill produces a monotone realised
column with a much flatter slope — visible immediately here and invisible in an
aggregate MAE.

---

## 8. Uncertainty, and whether it is useful

Epistemic uncertainty comes from MC dropout across the seed ensemble (LSTM,
combined by the law of total variance) or a moving-block bootstrap (XGBoost).
Aleatoric uncertainty is a multiple of the stock's trailing volatility scale.
Intervals come from normalised split-conformal calibration fitted on validation
only, which attains the requested coverage without assuming Gaussian errors.

Coverage alone is not enough to call an interval good. An interval can hit 80%
coverage overall while being far too narrow for volatile stocks and too wide for
calm ones — averaging to the right answer by being wrong in both directions.
Therefore also reported:

* **conditional coverage** by volatility regime and by forecast magnitude;
* **Mondrian (per-regime) conformal calibration**, available for when the
  conditional tables show a global multiplier is insufficient;
* a **filter-benefit test**: rank forecasts by `|prediction| / sigma`, keep the
  most confident fraction, and check whether IC actually improves.

**Measured, and reported honestly:** on the XGBoost run, filtering by confidence
*reduced* IC by 0.039, and the validation-tuned decision rule chose a minimum
z-score of 0.00 — meaning the risk-adjusted rule collapsed to the plain point
rule. Both facts say the same thing: **on this data the sigma estimate carries
little decision-relevant information.** The interval is still worth reporting —
it correctly communicates how wide the outcome distribution is — but it is not
earning its keep as a *filter*, and the report says so rather than presenting the
uncertainty machinery as though it were adding value it has not demonstrated.

---

## 9. Backtesting: what was unfair before

The signal backtest holds a full unit in every name clearing a hurdle and cash
otherwise. That left roughly 17–31% of capital invested, and its total return was
printed next to a 100%-invested buy-and-hold number. That comparison is
meaningless: a strategy two-thirds in cash *should* return less, and its Sharpe
ratio is flattered because cash has no volatility.

The fully invested **top-k portfolio** replaces it. On each rebalance date the
universe is ranked, the top `k` are bought, and weights are renormalised so
invested exposure sums to 1. Now the strategy and the equal-weight universe are
both 100% invested and the difference is attributable to *selection*.

* **Costs from realised turnover** (`sum |w_new − w_old|`), not a flat per-signal
  fee. Holding the same names costs nothing to keep holding.
* **Every rebalance offset evaluated.** A 21-day rebalance has 21 valid calendars
  and the choice moves the result materially. Reporting one is reporting one draw
  from a distribution and calling it the result; mean, median, worst and
  dispersion are reported instead.
* **Sector-neutral and beta-neutral variants** answer whether the edge is genuine
  stock selection or a standing bet on one sector or on high beta.

---

## 10. Blending

```text
blend = w_lstm * lstm + w_xgboost * xgboost,   w >= 0,   sum(w) = 1
```

* **Non-negative**: a negative weight means "predict the opposite of this model",
  which on out-of-fold data is almost always noise-fitting.
* **Sums to one**: keeps the blend in return units. Weights summing to more than
  one silently inflate every magnitude and corrupt both the reported percentage
  and the interval built around it.
* **Fitted out of fold only**, on rows where both families produced a forecast
  (an inner join on `(ticker, date)` — the two do not cover identical rows,
  because the sequence model needs a full look-back window).
* **Retained only if it beats the better standalone model** on the
  `mean − std` score. A blend that wins by a hair on the mean while being less
  stable is not an improvement, it is a more complicated way to get the same
  answer.
* **Equal weights are not assumed** — 50/50 is one candidate among 21 on the
  simplex grid and wins only if it earns it.

**Measured outcome.** The fitted weights were **LSTM 0.30 / XGBoost 0.70**, and
the blend beat the better standalone model by +0.229 out of fold, so it was
retained. On the development holdout it produced the best MSE and the best
magnitude skill of the three, but an IC of +0.0205 against XGBoost's +0.0437.

That gap is instructive rather than embarrassing. Out-of-fold selection is the
best available protection against selecting on the test set, but it is only as
reliable as the assumption that the future resembles the folds. Here that
assumption held for magnitude — averaging two independent error sources reduced
squared error exactly as predicted — and broke for ranking, because the LSTM's
own holdout IC collapsed from +0.0327 to +0.0118 and the blend inherited 30% of
that collapse. The honest conclusion is that for stock *selection*, XGBoost alone
remains the better choice on this data.

---

## 11. Acceptance gates

Nine pre-registered criteria are reported with every run. They are **reported,
never optimised against** — a failing gate is a finding about the model, not a
target to tune towards. Optimising the system to reach them would destroy the
only thing they are for.

---

## Post-mortem: what was actually wrong before

| Symptom | Root cause | Fix |
| --- | --- | --- |
| Every run peaked at epoch 1 | MSE-driven selection rewards a constant forecast | selection score requiring IC *and* magnitude skill |
| Target dominated by noise | total return contains the unforecastable market factor | beta-neutral residual target + stage-1 market model |
| `best_iteration = 0` but 50 trees used | `max(50, best+1)` overrode the selection silently | ladder evaluation on folds, no floor |
| MAE ≈ historical mean | nothing in the objective optimised magnitude | MSE term contributing to the gradient |
| Correlation loss over mixed rows | did not match the per-date metric it was checkpointed on | date-grouped batches + per-date IC loss |
| Importance from an unused model | reference booster was never used for inference | mean gain over the actual bootstrap ensemble |
| 17%-invested vs 100% buy-and-hold | exposure mismatch made the comparison meaningless | fully invested top-k portfolio |
| One arbitrary rebalance phase | one draw reported as the result | all 21 offsets, with dispersion |
| Constant market drift | ignores that expected market return is not constant | stage-1 model, shrunk to 0 when folds disagree |
