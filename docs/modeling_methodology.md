# Regression Modeling Methodology

## 1. Objective

Every model in this project is a **regressor**. For a horizon of `h` trading
sessions the reported prediction is the future percentage return

```text
future_return_h = adjusted_close[t + h] / adjusted_close[t] - 1
```

together with a calibrated **prediction interval** for that return. Neither model
is ever trained on BUY / HOLD / SELL classes; the discrete signal is derived
afterwards from the predicted return and its uncertainty (Section 7).

---

## 2. Why the first version scored so poorly

The earlier pipeline reported test correlation of 0.037, rank IC of 0.053 and a
backtest that executed a single trade. The diagnosis was not "the model needs
more tuning" — three specific defects made a good result impossible.

**2.1 The target was dominated by an unforecastable factor.**
Over a one-month horizon, most of the variance of a pooled multi-stock panel is
the *market* move common to every name. Technical indicators on a single stock
carry essentially no information about next month's index return, so including
that component in the target adds a large block of pure noise to the label.
The learnable part — which stock beats which — was a small fraction of the
signal being optimised.

**2.2 The checkpoint metric rewarded a constant forecast.**
Validation MAE, and any pooled score dominated by it, is minimised by predicting
a near-constant value on a low signal-to-noise target. The training logs show
this directly: the best epoch was **epoch 1** for every horizon, after which
validation MAE rose monotonically while training loss fell.

**2.3 A calibration step silently collapsed XGBoost to a constant.**
`fit_return_calibration` fitted a least-squares slope on validation, clipped it
to `[0, 3]`, and kept the result whenever MAE improved. When the raw slope came
out negative it was clipped to exactly `0.0`, producing `prediction = intercept`
for every row. MAE did improve, so the calibration was retained. That is why the
saved XGBoost model reported a return correlation of *exactly* 0.0.

Secondary issues also present: adjacent horizon labels overlapped across split
boundaries, LSTM training windows ended at `t-1` while inference ended at `t`,
absolute price levels let a pooled model identify tickers by scale, the trading
threshold was untuned, and backtests compounded overlapping `h`-day returns as
if they were daily.

---

## 3. Target: a market-excess decomposition

The label is decomposed into a market leg and a stock-specific leg:

```text
future_return = benchmark_future_return + future_excess_return
```

The model learns **only the excess leg**. At inference the market leg is supplied
by a drift constant estimated on the training window alone:

```text
predicted_total_return = market_drift_train + predicted_excess_return
```

This is the single most important change. It removes the large, unforecastable
component from the optimisation target while leaving the user-facing output
unchanged in meaning. Both components are stored in the model metadata and shown
separately in the UI, so the decomposition is transparent rather than hidden.

For optimisation, the excess leg is divided by a past-only volatility scale
(trailing idiosyncratic volatility times `sqrt(h)`) and clipped. This makes one
loss scale valid across stocks and horizons. The scale is stored per row and the
network output is multiplied back by it, so the decoded prediction is always in
percentage-return units.

Configured under `regression_target` in `configs/config.yaml`; `raw_return` and
`volatility_scaled` remain available for ablation.

---

## 4. Features

Roughly 70 stationary features per row: multi-horizon returns, price-to-moving
-average ratios, RSI, MACD ratios, ATR ratio, Bollinger position and width,
volume ratios and z-scores, realised and downside volatility, market-relative
excess returns, rolling beta, and idiosyncratic volatility.

Two feature rules follow from the target choice:

**4.1 Regime normalisation.** Selected unbounded features additionally get a
**252-session trailing z-score computed per ticker** (`*_z252`). Only past
observations enter each z-score, so there is no look-ahead. This makes a feature
comparable across stocks and adaptive to volatility regimes, which a single
global scaler fitted once on the training period cannot do.

**4.2 Market-wide features are excluded** (`exclude_market_wide_features: true`).
`benchmark_*` and `macro_*` columns take the *same value for every ticker on a
given date*. Against a market-excess target they cannot contribute anything to a
cross-sectional ranking, but they do give a sequence model a way to identify
which date a window came from and memorise that date's cross-sectional noise.
Removing them removes a memorisation channel without removing usable signal.
Stock-specific market-relative features (`excess_return_*`, `market_beta_60d`,
`idiosyncratic_volatility_20d`) are kept, so market context still reaches the
model — through the stock's *relationship* to the market rather than through the
market's level.

Absolute price, moving-average, MACD, ATR and volume **levels** stay excluded:
pooled across stocks they let a model identify a ticker by its price scale.
Present-day earnings snapshots are disabled by default because they are not a
point-in-time historical feed.

---

## 5. Evaluation: cross-sectional, not pooled

A pooled correlation over a stacked panel conflates two different questions:
*"did the market go up?"* and *"which stock beat which?"*. Only the second is
learnable from stock-level features, and it is measured **per date**.

The headline metric is therefore the **cross-sectional information coefficient**:
the mean, across test dates, of the Spearman rank correlation between the
forecast and the realised return within that date's cross-section. Reported
alongside it:

| Metric | Meaning |
| --- | --- |
| `mean_ic` | Average per-date rank correlation. In equity research 0.02–0.05 is a normal, usable value |
| `icir` | IC information ratio, `mean_ic / std(ic)`, annualised by `252/h` independent periods |
| `ic_t_statistic`, `ic_p_value` | Significance of the mean IC, with the sample size discounted for forward-window overlap |
| `ic_positive_rate` | Fraction of dates with positive IC |
| `long_short_spread` | Realised return of the top forecast quintile minus the bottom quintile, per period and annualised |

Point-forecast metrics (MAE, median AE, RMSE, R², directional accuracy, Pearson
correlation, RMSE skill versus a zero forecast) are still reported, in **both**
spaces: the market-excess component the model actually learns, and the total
return the user sees.

Because `predictive_score`-style pooled composites were part of the original
failure, the default early-stopping metric is now `cross_sectional_ic` directly.

---

## 6. Uncertainty estimation

Two sources are estimated separately and recombined.

**Epistemic (model) uncertainty** — how much the forecast would move if the model
had been fitted slightly differently.

* *LSTM*: **Monte Carlo dropout** across a **seed ensemble**. Dropout layers are
  kept stochastic at inference while normalisation layers stay in eval mode, and
  `mc_dropout_passes` forward passes are drawn per member. Within-member and
  between-member variance are combined exactly by the law of total variance,
  `Var = E_m[Var_pass] + Var_m[E_pass]`.
* *XGBoost*: a **moving-block bootstrap ensemble**. Each member is fitted on a
  resample of contiguous *date blocks* — resampling individual rows would ignore
  the serial dependence of a price panel and produce far too narrow a band.
  Bagging is also a genuine accuracy device here, not only an uncertainty device.

**Aleatoric (irreducible) uncertainty** — the noise in the return itself, which
dominates for equities. Estimated as a multiple of the row's trailing volatility
scale, so the band widens for volatile names and volatile regimes.

**Calibration.** The two are combined into a single sigma and turned into an
interval by **normalised split-conformal calibration**: the interval multiplier
is the finite-sample empirical quantile of the standardised absolute validation
errors, `|y - ŷ| / σ`. Under exchangeability this attains the requested marginal
coverage without assuming the errors are Gaussian, which a plain `± 1.96σ`
interval does assume and does not attain on fat-tailed returns.

Interval quality is then measured out of sample:

| Metric | Meaning |
| --- | --- |
| `coverage_picp` | How often the realised return actually fell inside the band. Should sit near the nominal level |
| `mean_interval_width_mpiw` | Average band width. Narrower is better **only given correct coverage** |
| `winkler_score` | Proper scoring rule combining width and coverage; lower is better |

A wide band is not a defect. Over 21 sessions a large-cap stock has a return
standard deviation near 8–10%; an honest 80% interval must be roughly ±11%.
A narrow band would simply be a miscalibrated one — which is exactly what
`coverage_picp` is there to detect.

---

## 7. The BUY / HOLD / SELL layer

The signal layer is kept, because it is what makes the forecast *testable*: it
turns a number into a decision a backtest can charge transaction costs against.
But the rule is risk-adjusted rather than a fixed percentage.

A fixed "+3% means BUY" rule is arbitrary and not comparable across stocks:
+3% expected on a low-volatility utility is a far stronger claim than +3% on a
high-beta semiconductor name. The default rule instead scores each forecast by
its edge per unit of forecast uncertainty,

```text
z = (predicted_return - cost_hurdle) / sigma
```

and acts only when `z` clears a floor. `cost_hurdle` is never below the
round-trip trading cost, so a signal must survive costs before it can be emitted.
Both parameters are tuned **on validation only**, by net non-overlapping Sharpe
with a small penalty for trading almost everything, and frozen before the test
set is touched.

`decision.rule: "point"` restores the classic fixed-threshold rule. Every run
also reports a **point-rule ablation** — the identical forecasts scored through
the threshold rule with no uncertainty input — so the contribution of the
risk-adjusted rule is measurable rather than assumed.

---

## 8. Temporal validation

Splits are chronological, never random. A **purge** of `h` sessions is inserted
before validation and before test, so a training label cannot be built from
prices that fall inside a later segment.

Two schemes are available:

* **Purged chronological hold-out** — 70 / 15 / 15, used for the shipped model.
* **Purged walk-forward** (`--walk-forward`) — `folds` successive out-of-sample
  windows, each refitted from scratch on data strictly before its own test
  window, with mean ± std reported across folds. This is what shows a result is
  not an artefact of one arbitrary cut of the timeline.

Results are additionally broken into consecutive test-regime blocks, so one
favourable period cannot hide instability.

Each horizon gets its own model and artifacts. A 21-session model is never
reused for a 5- or 252-session objective.

---

## 9. Baselines

A model that cannot beat a trivial forecast has demonstrated nothing. Every run
scores these on identical test rows, in both return spaces:

* zero return;
* global historical mean return;
* per-ticker historical mean return;
* trailing momentum, and its reverse (reversal);
* market drift (total-return space) and excess momentum (excess space).

---

## 10. Backtesting

Event-based and deliberately simple. Each prediction is one trade held for the
full horizon. Overlapping forward windows are removed by sampling every `h`-th
rebalance date, so an `h`-day outcome is never compounded as a daily return.
Costs are charged per unit of traded exposure (`2 × (commission + slippage)`),
so partial positions pay partial cost. Annualisation uses `252 / h` periods.

Reported: total return, equal-weight buy-and-hold comparison, excess versus
buy-and-hold, Sharpe, Sortino, information ratio against the universe, maximum
drawdown, win rate, activation rate and average gross exposure.

---

## 11. How to read the result honestly

No implementation can guarantee high accuracy in liquid equity return
prediction, and directional accuracy above 50% is not by itself evidence of
anything — it can be produced by a positive market drift alone.

A credible result here means, in order of importance:

1. positive **cross-sectional IC** out of sample, with a t-statistic that is not
   trivially explained by noise and a positive IC on a majority of dates;
2. beating the baselines in Section 9 on the same rows;
3. **interval coverage close to the nominal level** — an accurate point forecast
   with a miscalibrated band is not a usable forecast;
4. surviving costs in the backtest with a validation-frozen decision rule;
5. stability across walk-forward folds and regime blocks.

If those do not hold, the academically correct conclusion is that the tested
feature set has not demonstrated predictive edge for that horizon — and the
reporting in this project is built so that conclusion is visible rather than
obscured. Reporting a negative or insignificant IC is a valid outcome; hiding it
behind a favourable-looking MAE is not.
