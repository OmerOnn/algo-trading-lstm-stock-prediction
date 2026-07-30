# Demo Script

Target length: 10-12 minutes. The through-line is: *this is a regression system
that predicts a return **and** how much it trusts that return.*

---

## 1. Opening (1 min)

This is an academic simulation that predicts the **future percentage return** of
a stock over a chosen horizon, reports a **calibrated uncertainty range** with
every prediction, and evaluates the result as a trading strategy.

State the framing up front: the models are regressors. BUY / HOLD / SELL is never
a training target — it is derived afterwards from the predicted return and its
uncertainty, purely so the forecast can be tested against transaction costs.

---

## 2. Data (1 min)

* ~100 large-cap US stocks, 2005 to present, from Yahoo Finance.
* SPY as the benchmark.
* Macro / alternative indicators: VIX, 10-year Treasury yield proxy, USD index.
* ~483,000 supervised rows for a 21-session horizon.

---

## 3. The key modelling decision (2 min) — *the most important slide*

Show the decomposition:

```text
expected_stock_return = beta * expected_market_return   <- stage-1 model
                      + expected_stock_residual         <- what the model learns
```

Explain: over a month, most of the variance of a multi-stock panel is the market
move common to every name, and technical indicators on one stock say essentially
nothing about next month's index return. Training on the total return therefore
optimises mostly noise. The model learns only the **residual** leg. The user
still sees one total number, and the UI shows both components.

Two points worth making explicitly:

* The subtraction is **beta-weighted**, not a plain benchmark subtraction. A
  plain subtraction assumes every stock has beta 1, which leaves market exposure
  inside the target for high-beta names and inserts it with the wrong sign for
  low-beta ones.
* The market leg has its own model — and it **earned a weight of zero**. Fitted
  on one validation window it wanted 0.74; across purged folds the per-fold
  optima were 0.000 / 0.228 / 0.000, so the consistency rule set it to 0 and the
  leg reduces to the historical drift. That is a negative result, and showing it
  is a better slide than hiding it: the framework detected non-generalising skill
  and refused to use it.

Then show the feature rule that follows from it: `benchmark_*` and `macro_*`
columns are identical for every ticker on a date, so they cannot rank stocks —
but they let a sequence model identify *which date* a window came from and
memorise its noise. They are excluded. Stock-specific market-relative features
(`excess_return_*`, `market_beta_60d`, `idiosyncratic_volatility_20d`) are kept.

If asked "how do you know that mattered": before the fix, validation error rose
from epoch 1 and test IC was negative; after it, validation error is flat and IC
is positive across every walk-forward fold.

---

## 4. Models (1 min)

| | LSTM | XGBoost |
| --- | --- | --- |
| Input | 30-session sequences | engineered tabular features |
| Ensemble | 3 independently seeded members | 15 moving-block bootstrap members |
| Uncertainty | Monte Carlo dropout | bootstrap disagreement |

Same panel, same purged split, same target, same decision rule — so the
comparison is about the model, not the setup.

---

## 5. Uncertainty (2 min)

Two sources, separated then recombined:

* **Epistemic** — how much the forecast moves if the model is fitted slightly
  differently (MC dropout / bootstrap spread).
* **Aleatoric** — the noise in the return itself, scaled by the stock's trailing
  volatility so the band widens for volatile names.

The interval is then set by **normalised split-conformal calibration** on
validation only, so the coverage is empirical rather than assuming Gaussian
errors.

Show the headline honesty check: **out-of-sample coverage 78.5% against a
nominal 80%**. The epistemic share of variance is **0.1%** — almost all the
uncertainty is irreducible market noise, which is the correct and honest answer
for equities.

Then show the part most projects would leave out. **Conditional** coverage by
volatility bucket misses by up to 0.136, and filtering by confidence *reduced*
IC by 0.017. So the interval honestly communicates how wide the outcome
distribution is, but it is **not** useful as a filter — and the validation-tuned
decision rule agreed, selecting `min_z_score = 0`, which collapses the
risk-adjusted rule to a plain threshold. Saying that out loud is the point.

Pre-empt the obvious question: *"why is the band ±10%?"* Because a large-cap
stock's 21-day return has a standard deviation near 9%. A narrow band would
simply be a miscalibrated one — which is exactly what the coverage metric is
there to detect.

---

## 6. Evaluation (2 min)

Lead with the **cross-sectional information coefficient**, not accuracy. Explain
why: a pooled correlation conflates "did the market go up?" with "which stock
beat which?", and only the second is learnable here. Directional accuracy above
50% can be produced by market drift alone.

Show for XGBoost at horizon 21:

* purged walk-forward IC **+0.0403 ± 0.0077**, positive in **all three folds**;
* holdout IC **+0.0437**, positive on **62.2%** of dates, quintile spread
  **+15.3% annualised**;
* **MSE and MAE now beat the historical mean** (skill +0.0079 / +0.0081) — they
  were at parity before;
* best simple baseline (sector-relative momentum) IC **+0.0211**.

Then the backtest — and stress that this is the *fair* version. The old report
compared a ~17%-invested strategy to 100% buy-and-hold, which is not a
comparison. The fully invested top-20 portfolio, averaged over **all 21 rebalance
offsets**: Sharpe **+1.56** (worst offset +1.34), information ratio vs the
equal-weight universe **+0.77** (worst +0.43), annualised turnover 9×, costs
charged on realised turnover.

The single best evidence slide: **sector-neutralising improves it** (Sharpe
+1.89, IR +1.08). Removing the sector tilt makes the strategy better, which is
what you would expect if the edge is genuine stock selection rather than a
standing sector bet.

Be equally direct about the weak points: the IC t-statistic is **1.52**, below
the pre-registered bar of 2.0 — so one of the nine acceptance gates **fails**,
and it is reported as failing. And a plain ridge regression on the same features
reaches IC +0.0361 against the model's +0.0437, so most of the signal is linear.

---

## 7. Live demo (3 min)

```bash
# pre-run before the demo
python3 train_lstm.py    --horizon 21 --walk-forward
python3 train_xgboost.py --horizon 21 --walk-forward --grid-search --permutation-importance
python3 blend_models.py  --horizon 21

# live
python3 evaluate_saved_model.py --horizon 21 --model xgboost
python3 -m streamlit run app.py
```

In the UI:

1. Enter `AAPL, MSFT, NVDA`, horizon = 1 month, Run prediction.
2. Point at the headline: **Expected movement**, then immediately at
   **Estimated range** and the confidence chip.
3. Show the decomposition row: market baseline vs the model's own view.
4. Show the comparison chart ranking tickers with their intervals.
5. Open the **Model quality** tab — cross-sectional IC, coverage, the baseline
   table and the walk-forward folds are all in the app, not just in a report.
6. Point out the low-confidence warning banner: the UI says outright when a
   forecast is inconclusive rather than dressing it up as a call.

---

## 8. Closing (1 min)

The honest summary: a cross-sectional IC around 0.04 with a t-statistic of 1.5 is
a normal, usable, **not yet decisive** result in equity research — not a money
machine. The value of the project is that it measures that correctly.

Three things it reports that it could easily have hidden: the market-return model
earned a weight of zero, the uncertainty estimate does not improve decisions, and
one acceptance gate fails. Every selection decision was made on purged
walk-forward folds and the test period was treated as a development holdout,
because it has been looked at too many times to be anything else. A system that
reports its own null results is the deliverable.

Academic research and simulation only. Not financial advice.
