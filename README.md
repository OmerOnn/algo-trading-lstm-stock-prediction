# Algo Trading Stock Predictor

Academic final project for Track 2: a machine learning system that predicts the
**future percentage return** of one or more stocks over a chosen horizon,
reports a **calibrated uncertainty range** with every prediction, and evaluates
the result as a trading strategy.

Two regression model families share the same data pipeline, target, splits and
decision rule, so they are directly comparable:

1. **`LSTMEnsembleRegressor`** — seed ensemble of LSTMs over sequential windows,
   with Monte Carlo dropout for uncertainty.
2. **`XGBoostBootstrapRegressor`** — gradient-boosted trees over engineered
   tabular features, with a moving-block bootstrap ensemble for uncertainty.

> This project is for academic research and simulation only. It is not financial advice.

---

## What the models predict

The reported output is always the total future percentage return:

```text
future_return_h = Close[t + horizon] / Close[t] - 1
```

Internally the label is decomposed and only the stock-specific leg is learned:

```text
future_return = benchmark_future_return + future_excess_return
                └── market drift, from ──┘  └── what the model learns ──┘
                    training data only
```

Most of the variance of a pooled multi-stock panel is the market move common to
every name, and technical indicators on one stock carry essentially no
information about next month's index return. Training on the total return
therefore optimises mostly noise. Learning the **market-excess** leg and adding
back a train-only drift keeps the user-facing output identical in meaning while
making the learning problem tractable. Both components are shown separately in
the UI.

`BUY` / `HOLD` / `SELL` is **never** a supervised target. It is derived after
regression from the predicted return *and its uncertainty* — see
[Decision layer](#decision-layer).

Full rationale, including a written post-mortem of why the first version scored
poorly, is in [`docs/modeling_methodology.md`](docs/modeling_methodology.md).

---

## Every prediction has an uncertainty range

A point forecast alone is not actionable for equity returns, where irreducible
noise is an order of magnitude larger than any attainable edge. The system
separates two sources of uncertainty and recombines them:

| Source | LSTM | XGBoost |
| --- | --- | --- |
| **Epistemic** (model) | Monte Carlo dropout across the seed ensemble, combined by the law of total variance | Moving-block bootstrap ensemble (blocks of contiguous dates, to respect serial correlation) |
| **Aleatoric** (irreducible) | Multiple of the stock's trailing volatility scale, so the band widens for volatile names and regimes | same |

The combined sigma is turned into an interval by **normalised split-conformal
calibration**, fitted on validation data only: the interval multiplier is the
empirical quantile of the standardised absolute validation errors. Under
exchangeability this attains the requested coverage without assuming Gaussian
errors, which `± 1.96σ` does assume and does not attain on fat-tailed returns.

Interval quality is then reported out of sample as **PICP** (coverage), **MPIW**
(width) and the **Winkler score**, so a band that is too narrow or
uninformatively wide is visible rather than hidden.

---

## Decision layer

The discrete signal is kept because it is what makes the forecast *testable*: it
turns a number into a decision a backtest can charge costs against. But the rule
is **risk-adjusted**, not a fixed percentage:

```text
z = (predicted_return - cost_hurdle) / sigma        →  act only when z clears a floor
```

A fixed "+3% means BUY" rule is not comparable across stocks — +3% expected on a
low-volatility utility is a much stronger claim than +3% on a high-beta
semiconductor name. `cost_hurdle` is never below the round-trip trading cost, and
both parameters are tuned **on validation only** and frozen before the test set
is touched.

Set `decision.rule: "point"` in the config to restore the classic fixed
threshold. Every run also reports a **point-rule ablation** on identical
forecasts, so the contribution of the uncertainty-aware rule is measured rather
than assumed.

---

## Evaluation

The headline metric is the **cross-sectional information coefficient**: the mean
per-date Spearman rank correlation between forecast and outcome. A pooled
correlation over a stacked panel conflates *"did the market go up?"* with
*"which stock beat which?"*, and only the second is learnable from stock-level
features.

Reported for every run, on an untouched purged test period:

* `mean_ic`, `icir`, `ic_t_statistic`, `ic_p_value`, `ic_positive_rate`
* top-minus-bottom quintile spread, per period and annualised
* MAE, median AE, RMSE, R², directional accuracy, Pearson correlation, RMSE skill
  — in **both** the market-excess space and the total-return space
* interval coverage (PICP), width (MPIW), Winkler score
* six baselines on identical rows: zero, historical mean, per-ticker mean,
  momentum, reversal, market drift / excess momentum
* consecutive test-regime blocks, and optional purged walk-forward folds
* cost-aware backtest: total return, buy-and-hold comparison, Sharpe, Sortino,
  information ratio vs the universe, max drawdown, win rate, activation rate

---

## Project structure

```text
algo_trading_advanced_model/
|-- app.py                     # Streamlit UI with uncertainty display
|-- train.py                   # LSTM ensemble training
|-- train_xgboost.py           # XGBoost bootstrap ensemble training
|-- train_all_models.py        # both families, all horizons
|-- predict.py                 # inference with intervals + confidence
|-- compare_results.py         # LSTM vs XGBoost vs baselines report
|-- evaluate_saved_model.py    # full evaluation summary for a saved model
|-- check_device.py
|-- configs/config.yaml
|-- docs/
|   |-- modeling_methodology.md
|   |-- demo_script.md
|   `-- final_report.md
|-- src/
|   |-- backtest.py            # cost-aware, non-overlapping event backtest
|   |-- data_download.py
|   |-- dataset.py
|   |-- decision.py            # risk-adjusted BUY / HOLD / SELL
|   |-- device.py
|   |-- features.py            # indicators + trailing regime z-scores
|   |-- model.py
|   |-- pipeline.py
|   |-- plots.py
|   |-- regression.py          # target decomposition + cross-sectional metrics
|   |-- training_logger.py
|   |-- uncertainty.py         # MC dropout, bootstrap, conformal calibration
|   `-- validation.py          # purged hold-out and walk-forward splits
|-- tests/
|-- models/
|-- reports/
`-- requirements.txt
```

---

## Setup

```bash
cd algo_trading_advanced_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Check acceleration:

```bash
python3 check_device.py
```

Device selection is controlled by `configs/config.yaml`:

```yaml
device: "auto"
```

`auto` uses CUDA on NVIDIA GPUs, MPS/Metal on Apple Silicon when available, and
CPU otherwise. XGBoost supports GPU only through CUDA, so Apple Metal does not
accelerate it; set `xgboost.device: "cuda"` on an NVIDIA machine.

---

## Dataset cache

The processed panel is cached under `data/cache/` as Parquet and reused by later
runs for the same horizon.

```yaml
use_dataset_cache: true
force_rebuild_dataset_cache: false
```

The cache is versioned (`DATASET_SCHEMA_VERSION`). If tickers, dates, macro
sources or feature logic change, it rebuilds automatically.

---

## Training

```bash
python3 train.py --horizon 21                  # LSTM ensemble
python3 train_xgboost.py --horizon 21          # XGBoost bootstrap ensemble
python3 train_all_models.py --horizon 21 --compare
```

Add purged walk-forward validation (slower, refits per fold):

```bash
python3 train.py --horizon 21 --walk-forward
python3 train_all_models.py --horizon 21 --compare --walk-forward
```

Override the ensemble size for a fast run:

```bash
python3 train.py --horizon 21 --ensemble-size 1
```

Horizons are in **trading days**: 5 = one week, 21 = one month, 63 = one
quarter, 126 = six months, 252 = one year. The UI can only offer a horizon that
has been trained.

---

## Inference

```bash
python3 predict.py --ticker AAPL --horizon 21
python3 predict.py --tickers AAPL,MSFT,NVDA --horizon 21
```

Output per ticker:

```text
AAPL - 21 trading days ahead
--------------------------------------------------------
Latest market date      : 2026-07-28
Expected movement       : +1.84%
Estimated range (80%)   : -9.42% to +13.10%
Confidence              : Low (P[direction correct] = 56.3%)
  market baseline       : +0.76%   model view vs market: +1.08%
Signal (risk_adjusted)  : HOLD
Cost hurdle             : 0.87%
```

---

## Run the UI

```bash
python3 -m streamlit run app.py
```

Then open <http://localhost:8501>.

The dashboard shows, for each ticker:

* **Expected movement** as the headline number,
* **Estimated range** with a zero-anchored visual interval bar,
* a **confidence chip** (High / Moderate / Low) plus the probability the
  direction is right,
* the **decomposition** into market baseline and the model's stock-specific view,
* forecast sigma and the cost hurdle,
* the derived BUY / HOLD / SELL signal with a plain-English explanation.

A **Model quality** tab surfaces the saved out-of-sample evidence — cross-sectional
IC, directional accuracy, interval coverage, backtest Sharpe, the baseline table
and the walk-forward folds — so the numbers on screen can be judged against how
the model actually performed.

Tickers whose band is wider than their expected move are flagged explicitly as
inconclusive rather than presented as directional calls.

---

## Reports

```text
models/stock_advanced_model_h21.pt        # ensemble state dicts
models/xgboost_regressor_h21.joblib       # bootstrap ensemble
models/model_metadata_h21.json            # calibrations + frozen decision rule
reports/metrics_h21.json                  # full evaluation payload
reports/metrics_xgboost_h21.json
reports/test_predictions_h21.csv          # per-row forecast, bounds, signal
reports/backtest_results_h21.csv
reports/model_comparison_h21.md
reports/baseline_comparison_h21.csv
reports/feature_importance_xgboost_h21.csv
reports/plots/h21/                        # scatter, deciles, intervals, IC series
logs/training_run_*.log                   # full console transcript per run
```

Inspect a saved model:

```bash
python3 evaluate_saved_model.py --horizon 21 --model lstm
python3 compare_results.py --horizon 21
```

---

## Tests

```bash
python3 -m unittest discover -s tests
```

Covers the target decomposition, cross-sectional metrics, calibration guard
rails, conformal interval coverage, MC dropout behaviour, block bootstrap,
decision rules, purged splits and the backtest cost model.

---

## Limitations

* This is a research simulation, not a production trading system.
* Yahoo Finance data can be delayed, incomplete, or unavailable for some tickers.
* The backtest ignores market impact, order-book dynamics and borrow costs.
* Equity returns are close to unforecastable at these horizons. A cross-sectional
  IC of 0.02–0.05 is a normal, usable result; anything much larger on this kind
  of feature set should be treated as a bug or a leak, not a discovery.
* Results vary across horizons and market regimes, which is why regime blocks and
  walk-forward folds are reported rather than a single headline number.

---

## Academic disclaimer

The system predicts simulated future returns using historical price data,
technical indicators, benchmark context and macro data. It is evaluated using
regression metrics, uncertainty-calibration metrics and backtesting, for academic
comparison only. It is not investment advice.
