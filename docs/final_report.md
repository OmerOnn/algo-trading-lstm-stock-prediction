# Final Project Report

## 1. Project goal

Build an academic machine learning system that predicts the **future percentage
return** of a stock over a chosen horizon, quantifies **how uncertain that
prediction is**, and evaluates the result as a trading strategy under realistic
costs.

## 2. Problem definition

For a horizon of `h` trading sessions the supervised target is

```text
future_return_h = adjusted_close[t + h] / adjusted_close[t] - 1
```

This is a **regression** problem. BUY / HOLD / SELL is never a supervised
target; it is derived after regression from the predicted return and its
uncertainty, so that the forecast can be charged transaction costs and tested as
a strategy (Section 8).

The deliverable for each prediction is a triple: a point forecast, a calibrated
interval, and an explicit confidence statement.

## 3. Data

| Source | Content |
| --- | --- |
| Yahoo Finance | Daily OHLCV for ~100 large-cap US stocks, 2005 to present |
| SPY | Benchmark for market-relative features and the market leg of the target |
| Macro / alternative | VIX, 10-year Treasury yield proxy, USD index |

For a 21-session horizon this yields **483,775 supervised rows**: 330,579 train,
73,488 validation, 75,613 test after purging.

## 4. What went wrong in the first version

The first implementation reported test correlation of 0.037, rank IC of 0.053
and a backtest that executed **one** trade. Three concrete defects made a good
result impossible; each is documented in `docs/modeling_methodology.md`.

**4.1 The target was dominated by an unforecastable factor.** Most of the
variance of a pooled multi-stock panel over a month is the market move common to
every name. Technical indicators on one stock carry almost no information about
next month's index return, so training on the total return meant optimising
mostly noise.

**4.2 The checkpoint metric rewarded a constant forecast.** Validation MAE is
minimised by predicting a near-constant on a low signal-to-noise target. The
training logs confirm it directly: the best epoch was **epoch 1** for every
horizon, after which validation MAE rose monotonically while training loss fell.

**4.3 A calibration step silently collapsed XGBoost to a constant.**
`fit_return_calibration` fitted a least-squares slope on validation, clipped it
to `[0, 3]`, and retained the result whenever MAE improved. A negative raw slope
clipped to exactly `0.0`, making `prediction = intercept` for every row. MAE did
improve, so it was kept — which is why the saved XGBoost model reported a return
correlation of *exactly* 0.0.

Two further defects were found during this work's own verification: a
`Timestamp` / `datetime64` **hash mismatch** that made every bootstrap resample
empty, and an unconditional Windows certificate path that created a literal
`C:` directory inside the repository on macOS.

## 5. Method

### 5.1 Target decomposition

```text
future_return = benchmark_future_return + future_excess_return
                └── market drift, from ──┘  └── what the model learns ──┘
                    training data only
```

The model learns only the **market-excess** leg; the market leg is supplied by a
drift estimated on the training window alone (+0.76% over 21 sessions). The
user-facing output is unchanged in meaning, and both components are reported
separately. For optimisation the excess leg is divided by a past-only volatility
scale and clipped, then decoded back into return units.

### 5.2 Features

~70 stationary features are engineered; **53** survive after the market-wide
exclusion below. Multi-horizon returns, price-to-moving-average ratios, RSI,
MACD ratios, ATR ratio, Bollinger position and width, volume ratios and
z-scores, realised and downside volatility, market-relative excess returns,
rolling beta and idiosyncratic volatility.

Two rules follow from the target choice:

* **Trailing regime normalisation** — selected features additionally get a
  252-session z-score computed per ticker from past observations only, making
  them comparable across stocks and adaptive to volatility regimes.
* **Market-wide features are excluded** — `benchmark_*` and `macro_*` take the
  same value for every ticker on a date. They cannot rank stocks, but they let a
  sequence model identify *which date* a window came from and memorise its
  cross-sectional noise. Stock-specific market-relative features are kept.

### 5.3 Models

| | LSTM | XGBoost |
| --- | --- | --- |
| Input | 30-session sequences | tabular features |
| Ensemble | 3 seed members | 15 moving-block bootstrap members |
| Uncertainty | Monte Carlo dropout | bootstrap disagreement |
| Loss / objective | Huber + batch correlation term | squared error, early stopping on rank correlation |

Both use the same panel, purged split, target and decision rule, so the
comparison isolates the model.

### 5.4 Validation

Chronological only, with a **purge** of `h` sessions before validation and before
test. Two schemes: a 70/15/15 purged hold-out for the shipped model, and purged
**walk-forward** (3 expanding-origin folds, each refitted from scratch) to show
the result is not an artefact of one cut of the timeline.

## 6. Evaluation methodology

The headline metric is the **cross-sectional information coefficient** — the mean
per-date Spearman rank correlation between forecast and realised return. A
pooled correlation conflates *"did the market go up?"* with *"which stock beat
which?"*, and only the second is learnable from stock-level features.
Directional accuracy above 50% is explicitly **not** treated as evidence, since
positive market drift alone produces it.

Reported per run: `mean_ic`, `icir`, `ic_t_statistic`, `ic_p_value`,
`ic_positive_rate`, top-minus-bottom quintile spread, MAE, median AE, RMSE, R²,
directional accuracy, Pearson correlation, RMSE skill — in both the market-excess
and total-return spaces — plus interval coverage (PICP), width (MPIW), the
Winkler score, six baselines, regime blocks, walk-forward folds and a cost-aware
backtest.

## 7. Uncertainty estimation

Epistemic uncertainty (MC dropout across the seed ensemble for the LSTM,
bootstrap spread for XGBoost) is combined with aleatoric uncertainty (a multiple
of the stock's trailing volatility scale) and turned into an interval by
**normalised split-conformal calibration** fitted on validation only. The
multiplier is the finite-sample empirical quantile of standardised absolute
validation errors, so the stated coverage does not assume Gaussian errors.

A notable finding: the **epistemic share of predictive variance is 0.1%**.
Essentially all forecast uncertainty is irreducible return noise, not model
disagreement. This is the correct answer for liquid equities and explains why an
honest 80% interval for a 21-day horizon is roughly ±10%.

## 8. Decision layer

The signal rule is risk-adjusted rather than a fixed percentage:

```text
z = (predicted_return - cost_hurdle) / sigma      →  act only when z clears a floor
```

A fixed "+3% means BUY" is not comparable across stocks. `cost_hurdle` is never
below round-trip cost, and both parameters are tuned on validation only and
frozen before test. Each run also reports a **point-rule ablation** on identical
forecasts to measure the uncertainty rule's contribution.

## 9. Results — horizon 21 trading days

Test period 2023-05-30 to 2026-06-26, never used for training, early stopping,
calibration or threshold selection.

### 9.1 Predictive skill (market-excess space)

| Metric | LSTM ensemble | XGBoost bootstrap | Best baseline |
| --- | ---: | ---: | ---: |
| Cross-sectional IC | +0.0328 | **+0.0552** | +0.0199 (excess momentum) |
| IC t-statistic | +1.27 | **+2.03** | — |
| ICIR | +0.73 | **+1.16** | — |
| Dates with positive IC | 58.5% | **65.8%** | — |
| Quintile spread (annualised) | +15.6% | **+20.2%** | — |
| Directional accuracy (total return) | 0.5719 | 0.5713 | 0.4784 |
| MAE | 0.0638 | **0.0603** | 0.0601 (zero forecast) |
| RMSE | 0.0879 | **0.0828** | 0.0829 (zero forecast) |

Both models beat every baseline on cross-sectional IC by a factor of ~1.6-2.8.
XGBoost is the stronger model on this horizon, and is the only one whose IC
t-statistic reaches conventional significance (p = 0.050).

Note that MAE and RMSE are barely distinguishable from a zero forecast. This is
expected and is precisely why MAE was the wrong metric to optimise: on a target
this noisy, error magnitude is nearly uninformative about ranking skill.

### 9.2 Interval quality

| Metric | LSTM | XGBoost |
| --- | ---: | ---: |
| Nominal level | 80% | 80% |
| Empirical coverage (PICP) | 81.7% | 78.2% |
| Mean interval width (MPIW) | 23.4% | 20.1% |
| Winkler score | — | 0.297 |
| Epistemic share of variance | <1% | 0.1% |

Both intervals are well calibrated out of sample, within ~2 points of nominal.

### 9.3 Stability

XGBoost cross-sectional IC across consecutive test-regime blocks:
**+0.106, +0.007, +0.084, +0.024** — positive in all four, but clearly
time-varying, as the IC time-series plot shows.

Purged walk-forward (3 expanding folds, refitted from scratch):

| Model | Cross-sectional IC | Folds positive | Per fold |
| --- | ---: | ---: | --- |
| XGBoost | **+0.0374 ± 0.0101** | 3 / 3 | +0.026, +0.040, +0.046 |
| LSTM | +0.0248 ± 0.0139 | 3 / 3 | +0.036, +0.029, +0.009 |

Both models are positive in every fold, which is the main stability claim. Their
trends run opposite: XGBoost improves on later folds while the LSTM decays, and
the LSTM's weakest fold (2023-2026) is the same period as the hold-out test set.
This is consistent with the LSTM being the more variance-prone of the two on this
feature set.

### 9.4 Cost-aware backtest (non-overlapping horizon dates)

| Metric | LSTM | XGBoost |
| --- | ---: | ---: |
| Total return | +16.57% | **+29.98%** |
| Equal-weight buy-and-hold | +87.66% | +87.66% |
| Sharpe ratio (net of costs) | **1.827** | 1.724 |
| Max drawdown | **-2.01%** | -2.38% |
| Win rate | **58.32%** | 57.26% |
| Active trades | 631 | 1,109 |
| Point-rule ablation Sharpe | 1.827 | 1.673 |

The strategies have strong risk-adjusted profiles but **trail buy-and-hold in
total return**, because they are long-only and invested in only ~17% (LSTM) and
~31% (XGBoost) of opportunities during a strong bull market. Reporting this
openly matters: a favourable Sharpe does not imply the strategy beat the market.

The point-rule ablation is informative in both directions. For XGBoost the
risk-adjusted rule raised Sharpe (1.673 → 1.724) by declining marginal trades.
For the LSTM the tuned floor was zero, so the two rules coincide exactly and the
uncertainty input contributed nothing — an honest null result for that model.

The strategy has a strong risk-adjusted profile but **trails buy-and-hold in
total return**, because it is long-only and invested in only ~31% of
opportunities during a strong bull market. Reporting this openly matters: a
favourable Sharpe does not imply the strategy beat the market.

## 10. Conclusions

1. The dominant problem was **problem formulation, not model capacity**. Changing
   the target from total return to market-excess return, and the checkpoint
   metric from MAE to cross-sectional IC, moved the LSTM from a negative test IC
   (-0.024) to a positive one (+0.033), and XGBoost from exactly 0.0 to +0.055.
2. **Gradient boosting outperformed the LSTM** on predictive skill (IC +0.055 vs
   +0.033, and the only model reaching t > 2), though the LSTM edged it on
   risk-adjusted backtest metrics by trading less often. The sequence model's
   extra capacity was mostly used to memorise, which is why shortening the window
   and removing date-identifying features helped it most.
3. **A cross-sectional IC of 0.04-0.06 with t ≈ 2 is a normal, usable result** in
   equity research. It is not a money machine, and the reporting is built so that
   this is legible rather than obscured.
4. **Uncertainty is almost entirely irreducible.** The most defensible thing the
   system tells a user is not the point forecast but the width of the band around
   it, and the calibration metrics prove that width is honest.

## 11. Limitations

* Yahoo Finance data quality varies by ticker and date range; one ticker (`FI`)
  failed to download and was skipped.
* The backtest ignores market impact, order-book dynamics, borrow costs and
  capacity constraints.
* The universe is survivorship-biased: it is today's large-cap list applied
  retrospectively, which flatters any long-only result.
* The market drift used to rebuild total return is a single train-period
  constant, not a conditional forecast.
* Results vary substantially across regimes, as the IC time series shows.

## 12. Future work

* Address survivorship bias with a point-in-time index membership source.
* Add genuinely orthogonal data: point-in-time fundamentals, analyst revisions,
  short interest, news sentiment.
* Model the market leg explicitly rather than as a constant drift.
* Quantile regression or a distributional head as an alternative to conformal
  intervals, so the model learns its own heteroskedasticity.
* Confidence-weighted position sizing and turnover-aware portfolio construction.
