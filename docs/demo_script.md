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
future_return = benchmark_future_return + future_excess_return
                └── market drift, from ──┘  └── what the model learns ──┘
                    training data only
```

Explain: over a month, most of the variance of a multi-stock panel is the market
move common to every name, and technical indicators on one stock say essentially
nothing about next month's index return. Training on the total return therefore
optimises mostly noise. The model learns the **market-excess** leg; the market
leg comes from a train-only drift estimate. The user still sees one total number,
and the UI shows both components.

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

Show the headline honesty check: **out-of-sample coverage 78.2% against a
nominal 80%**. And note that the epistemic share of variance is under 1% — almost
all the uncertainty is irreducible market noise, which is the correct and honest
answer for equities.

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

* cross-sectional IC **+0.055**, t-statistic **+2.03**, positive on **65.8%** of
  test dates;
* top-minus-bottom quintile spread **+20.2% annualised**;
* best baseline (excess momentum) IC **+0.020** — the model beats it on identical
  rows;
* purged walk-forward: **+0.037 ± 0.010**, positive in all three folds.

Then the backtest: Sharpe 1.72 net of costs and slippage on non-overlapping
horizon dates, max drawdown -2.4%. Be honest that it trails equal-weight
buy-and-hold in total return because it is long-only and selectively invested.

---

## 7. Live demo (3 min)

```bash
python3 train_all_models.py --horizon 21 --compare --walk-forward   # pre-run
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

The honest summary: a cross-sectional IC around 0.04-0.06 with a t-statistic near
2 is a normal, usable result in equity research — not a money machine. The value
of the project is that it *measures* that correctly, quantifies its own
uncertainty, and would show a null result just as clearly if there were one.

Academic research and simulation only. Not financial advice.
