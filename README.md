# Algo Trading Stock Predictor

Academic final project for Track 2: a machine learning system that predicts the
**future percentage return** of one or more stocks over a chosen horizon,
reports a **calibrated uncertainty range** with every prediction, and evaluates
the result as a trading strategy.

Two regression model families share the same data pipeline, target, splits,
calibration, uncertainty treatment and decision rule, so they are directly
comparable:

1. **`LSTMEnsembleRegressor`** — seed ensemble of LSTMs over sequential windows,
   trained on date-grouped batches, with Monte Carlo dropout for uncertainty.
2. **`XGBoostBootstrapRegressor`** — gradient-boosted trees over engineered
   tabular features, with a moving-block bootstrap ensemble for uncertainty.

Both are **regression models**. `BUY` / `HOLD` / `SELL` is never a training
target — it is derived after regression from the predicted return, its
uncertainty, transaction costs and a validation-selected rule.

> This project is for academic research and simulation only. It is not financial advice.

---

## Commands

```bash
python3 train_lstm.py    --horizon 21          # LSTM ensemble
python3 train_xgboost.py --horizon 21          # XGBoost bootstrap ensemble
python3 train_all_models.py --horizon 21 --compare
```

Add purged walk-forward validation. **This is the recommended way to train**: it
is what selects the boosting rounds, the return calibration and the blend
weights, and it is what the acceptance gates are scored on.

```bash
python3 train_lstm.py    --horizon 21 --walk-forward
python3 train_xgboost.py --horizon 21 --walk-forward
python3 train_all_models.py --horizon 21 --compare --walk-forward
```

Blend the two families (needs a `--walk-forward` run of both, which is what
writes the out-of-fold predictions the weights are fitted on):

```bash
python3 blend_models.py    --horizon 21
python3 compare_results.py --horizon 21
```

Everything in one command:

```bash
python3 train_all_models.py --horizon 21 --walk-forward --compare --blend \
        --grid-search --permutation-importance
```

Optional experiment matrices (slower; results written to `reports/experiments/`):

```bash
python3 train_xgboost.py --horizon 21 --walk-forward --grid-search --permutation-importance
python3 train_lstm.py    --horizon 21 --walk-forward --loss-experiment --architecture-experiment
```

Launch the website:

```bash
python3 -m streamlit run app.py      # then open http://localhost:8501
```

---

## What the models predict

The reported output is always the total future percentage return:

```text
future_return_h = Close[t + horizon] / Close[t] - 1
```

Internally that total is **decomposed**, and only the stock-specific leg is
learned:

```text
expected_stock_return = beta * expected_market_return   <- one number per date
                      + expected_sector_return          <- see note below
                      + expected_stock_residual         <- what the model learns
```

Most of the variance of a pooled multi-stock panel is the market move common to
every name, and technical indicators on one stock carry essentially no
information about next month's index return. Training on the total return
therefore spends nearly all of the model's capacity on noise.

**Why the target is market-excess, and what beta-neutral cost.** A plain
benchmark subtraction assumes every stock has beta 1. A beta-weighted residual
(`r − β·r_bench`) is the theoretically better target and is fully implemented
(`regression_target.mode: "beta_neutral_residual"`), and it removes more variance
— 30.3% versus 27.7%. It was compared against plain market-excess on three purged
folds and **lost**:

| target | mean IC | std across folds | selection score |
| --- | --- | --- | --- |
| **market_excess** (shipped) | +0.0411 | **0.0039** | **+0.0380** |
| beta_neutral_residual | +0.0411 | 0.0147 | +0.0262 |

Identical mean skill, but the beta-weighted version is far less stable — its
rolling beta estimate is itself noisy, and multiplying a noisy beta into the
target injects that noise. The `mean − std` rule therefore selects market-excess.
The better-looking idea was implemented, measured and rejected on the evidence.
(`reports/experiments/experiment_target_mode_h21.md`)

**The sector leg is not separately forecast.** Sector information enters through
the feature set — sector composites, sector beta, sector-relative momentum,
sector-relative volatility — rather than as its own predicted term. The leg is
reported as zero so the composition stays an exact identity, and the report says
so explicitly rather than implying a component that does not exist.

Full rationale is in [`docs/modeling_methodology.md`](docs/modeling_methodology.md).

---

## How each model trains

### LSTM (`train_lstm.py`)

* **Date-grouped batches.** Each batch is made of whole date cross-sections, so
  the ranking term of the loss has a real cross-section to work with. Sampling
  rows independently would scatter each date across many batches and leave two or
  three names per date, from which no ordering can be learned.
* **Composite objective**, configurable in `configs/config.yaml`:

  ```yaml
  regression_loss:
    mse_weight: 0.40
    huber_weight: 0.40
    cross_sectional_ic_weight: 0.20
    huber_beta: 0.5
  ```

  MSE genuinely contributes to the gradient — it is not merely reported
  afterwards. It is what makes the *magnitude* meaningful, so the output can
  honestly be called an expected percentage return. Huber caps the gradient of
  extreme return events on a fat-tailed panel. The IC term is the only one that
  rewards getting the ordering right. Pearson correlation within each date is the
  differentiable surrogate for the Spearman IC that is reported.

  **These weights were selected, not asserted.** All four variants were compared
  on three purged folds at an identical budget, and this one won on every fold:

  | loss | mean IC | std | selection score |
  | --- | --- | --- | --- |
  | **MSE + Huber + date-IC** (shipped) | **+0.0371** | **0.0214** | **+0.0148** |
  | pure MSE | +0.0342 | 0.0222 | +0.0113 |
  | MSE + Huber | +0.0336 | 0.0255 | +0.0067 |
  | pure Huber | +0.0246 | 0.0299 | −0.0089 |
* **Architecture**: input dropout, optional variational recurrent dropout, dual
  pooling (last state + sequence mean), LayerNorm, GELU head, and optional
  auxiliary regression heads on other horizons.

### XGBoost (`train_xgboost.py`)

* `reg:squarederror` by default; `reg:pseudohubererror` is in the grid and is
  compared across folds.
* **No early stopping.** Boosting rounds are chosen afterwards by scoring one
  fitted booster at a ladder of iteration counts (`1, 5, 10, 20, 40, 80, …`) via
  `iteration_range`, on purged folds. One fit yields the whole curve.
* **No minimum-round floor.** The previous version reported `best_iteration = 0`
  and then silently used 50 trees, so the number in the metadata described
  nothing that had happened. The selected count is now the count that is used.
* Feature importance is averaged over the **actual bootstrap ensemble** used for
  inference, with the dispersion across members, not from a throwaway reference
  model. Out-of-fold **permutation importance** is also reported.

### Why early stopping is not based on MSE

Selecting on MSE alone rewards a constant forecast. On a target this close to
unforecastable, predicting the mean is a strong squared-error solution and
carries zero stock-selection information — it is why earlier runs of this project
peaked at epoch 1 every time. Selecting on IC alone has the opposite failure: a
model that orders stocks correctly while emitting wildly mis-scaled magnitudes,
which is fatal when the product reports a percentage.

The checkpoint criterion therefore requires **both**:

```text
validation_selection_score = cross_sectional_ic + 0.25 * mse_skill_vs_historical_mean
```

The same shape of score selects the XGBoost round count, so both families are
chosen on one definition of "better".

---

## Validation methodology

**The test period is a development holdout, not a pristine test set.** It has
been inspected repeatedly across the life of this project, and after enough looks
"test performance" measures the analyst as much as the model. It is reported but
**never used to select anything**.

Every selection decision — features, hyperparameters, loss, boosting rounds,
return calibration, blend weights, uncertainty calibration and decision
thresholds — is made on **purged walk-forward out-of-fold** data only. In code,
each trainer slices `history = full_df[full_df.index <= validation_end]` and all
selection runs on that slice.

Splits are purged by the full horizon, so no training row carries a label built
from prices inside the next segment. The current 21-day split:

| split | dates |
| --- | --- |
| train | 2006-01-17 → 2020-04-03 |
| validation | 2020-05-06 → 2023-04-28 |
| test (development holdout) | 2023-05-31 → 2026-06-29 |

**Selection rule.** Candidates are ranked by `mean − std` across folds, not by
mean alone. A configuration that wins on average while swinging between folds has
not demonstrated an edge, it has demonstrated sensitivity to the period.

---

## Calibration

The models produce reliable *ordering* skill and weak *magnitude* skill. Closing
that gap is what calibration does, under four rules:

1. **It may not destroy the ranking.** A least-squares fit on a low-signal panel
   will happily return a near-zero slope, because flattening every prediction
   improves squared error while deleting the only thing the model produced. Every
   candidate is checked for monotonicity **with respect to the model's own
   output** — not agreement with the outcome, because when the raw forecast is
   negatively correlated in the fitting window, collapsing to a constant
   *improves* agreement and would be waved through as an upgrade.
2. **The residual leg carries no common intercept**, and is **cross-sectionally
   centred** so the alpha sums to roughly zero across the universe. A level on a
   given date is a market call, and the market call belongs to the market model.
3. **Shrinkage towards zero**, the honest default for an alpha forecast.
4. Affine, ridge and isotonic are compared **out of fold**.

`reports/decile_calibration_*.csv` reports, per predicted decile: mean predicted
return, mean realised return, count, directional hit rate and standard error.

---

## Uncertainty

| Source | LSTM | XGBoost |
| --- | --- | --- |
| **Epistemic** | MC dropout across the seed ensemble, combined by the law of total variance | Moving-block bootstrap ensemble |
| **Aleatoric** | Multiple of the stock's trailing volatility scale | same |

Intervals come from **normalised split-conformal calibration** fitted on
validation only. Reported out of sample: PICP (coverage), MPIW (width), Winkler
score, **conditional coverage by volatility regime and by forecast magnitude**,
and a **filter-benefit test** that asks whether discarding low-confidence
forecasts actually improves IC. Mondrian (per-regime) conformal calibration is
implemented in `src/uncertainty.py`.

---

## Signals and backtesting

`BUY` / `HOLD` / `SELL` is derived from the regression output. Separated
explicitly: expected total return, expected residual alpha, uncertainty, and the
transaction-cost hurdle.

Two backtests are reported:

1. **Signal backtest** — one trade per qualifying prediction, held for the
   horizon, overlapping windows removed.
2. **Fully invested top-k portfolio** — the honest test. On each rebalance date
   the universe is ranked, the top `k` are bought, and weights are renormalised
   so invested exposure sums to 1. Both the strategy and the equal-weight
   universe are 100% invested, so the difference between them is attributable to
   *selection* rather than to exposure.

   * Costs are charged on **realised turnover** (`sum |w_new − w_old|`), so
     holding the same names costs nothing to keep holding.
   * **Every rebalance offset is evaluated.** A 21-day rebalance has 21 possible
     calendars and the choice moves the result materially; mean, median, worst
     and dispersion are reported instead of one lucky alignment.
   * **Sector-neutral and beta-neutral variants** answer whether the edge is real
     stock selection or a standing bet on one sector or on high beta.

The old comparison of a ~17%-invested strategy against 100%-invested
buy-and-hold has been removed. It was not a fair test.

---

## Baselines and acceptance gates

Every model is scored on identical rows against: zero return, historical mean,
rolling historical mean, market drift, per-ticker mean, per-sector mean,
momentum, reversal, excess momentum, residual momentum, sector-relative momentum,
a market-only forecast (`beta × drift`, i.e. no stock view at all) and a ridge
regression on the same features.

Nine pre-registered acceptance gates are **reported, never optimised against**. A
failing gate is a finding about the model, not a target to tune towards.

---

## Project structure

```text
algo_trading_advanced_model/
|-- app.py                     # Streamlit UI with uncertainty display
|-- train_lstm.py              # LSTM ensemble training
|-- train_xgboost.py           # XGBoost bootstrap ensemble training
|-- train_all_models.py        # both families, all horizons
|-- predict.py                 # inference with intervals + confidence
|-- compare_results.py         # LSTM vs XGBoost vs baselines report
|-- evaluate_saved_model.py
|-- configs/config.yaml
|-- docs/
|   |-- modeling_methodology.md
|   |-- final_report.md
|   `-- demo_script.md
|-- src/
|   |-- acceptance.py          # pre-registered acceptance gates
|   |-- backtest.py            # cost-aware signal backtest
|   |-- blending.py            # constrained non-negative LSTM/XGB blend
|   |-- boosting.py            # round selection + ensemble/permutation importance
|   |-- calibration.py         # out-of-fold calibration + decile report
|   |-- dataset.py             # lazy sequence dataset + date-grouped sampler
|   |-- data_download.py
|   |-- decision.py            # risk-adjusted BUY / HOLD / SELL
|   |-- evaluation.py          # the shared evaluation both trainers call
|   |-- experiments.py         # reproducible walk-forward experiment runner
|   |-- features.py            # per-ticker indicators + liquidity
|   |-- losses.py              # MSE + Huber + per-date IC objective
|   |-- market_model.py        # stage-1 market-return model, shrunk by folds
|   |-- model.py
|   |-- panel_features.py      # sector, breadth, dispersion, ranks, interactions
|   |-- pipeline.py
|   |-- plots.py
|   |-- portfolio.py           # top-k fully invested backtest, multi-offset
|   |-- regression.py          # targets + metrics + baselines
|   |-- training_common.py     # shared trainer orchestration
|   |-- uncertainty.py         # MC dropout, bootstrap, conformal, Mondrian
|   `-- validation.py          # purged hold-out and walk-forward splits
|-- tests/
|-- models/  reports/  logs/
`-- requirements.txt
```

---

## Setup

```bash
cd algo_trading_advanced_model
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python3 check_device.py
```

`device: "auto"` uses CUDA on NVIDIA, MPS/Metal on Apple Silicon, CPU otherwise.

The two model families do not share a compute backend, and on a Mac they will
not agree — this is expected, not a misconfiguration:

| | LSTM (PyTorch) | XGBoost |
|---|---|---|
| Apple Silicon | GPU via MPS/Metal | **CPU only** |
| NVIDIA | GPU via CUDA | GPU via CUDA, if the wheel was built with it |

XGBoost's `device` parameter accepts only `cpu`, `cuda`, and `cuda:<ordinal>`;
`mps` and `metal` are rejected by the library itself, and no release contains a
Metal, MPS, or OpenCL tree builder. The `mps`/`metal`/`mac`/`apple` aliases in
`configs/config.yaml` therefore resolve to CPU with an explanation rather than
an error. `auto` selects CUDA only when the installed xgboost was *compiled*
with CUDA **and** an NVIDIA GPU is present — the build flag is checked because
`device="cuda"` on a CPU-only wheel does not fail, it warns and silently trains
on CPU.

`python3 check_device.py` reports both backends, and every training run prints
the resolved XGBoost device with the build flags behind it and records them
under `execution_backend` in the run metadata.

The processed panel is cached under `data/cache/` as Parquet and versioned by
`DATASET_SCHEMA_VERSION`; it rebuilds automatically when tickers, dates, macro
sources or feature logic change.

---

## Inference

```bash
python3 predict.py --ticker AAPL --horizon 21
python3 predict.py --tickers AAPL,MSFT,NVDA --horizon 21
```

---

## Artifact locations

```text
models/stock_advanced_model_h21.pt        # LSTM ensemble state dicts
models/xgboost_regressor_h21.joblib       # bootstrap ensemble
models/model_metadata_h21.json            # calibrations + frozen decision rule
models/xgboost_metadata_h21.json
reports/metrics_h21.json                  # full LSTM evaluation payload
reports/metrics_xgboost_h21.json
reports/test_predictions_h21.csv          # per-row forecast, bounds, components
reports/decile_calibration_h21.csv
reports/portfolio_h21.csv
reports/backtest_results_h21.csv
reports/feature_importance_xgboost_h21.csv
reports/experiments/                      # experiment JSON / CSV / Markdown
reports/plots/h21/  reports/plots/xgboost_h21/
logs/training_run_*.log                   # full console transcript per run
```

---

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q .
git diff --check
```

---

## Limitations

* This is a research simulation, not a production trading system.
* **Survivorship bias.** The universe is 100 companies that are large *today*,
  held fixed across 2006–2026. Names that were large in 2006 and later failed or
  were acquired are absent, so the sample is conditioned on survival. Returns are
  biased upward, and the results should be read as relative comparisons between
  models on identical data rather than as an achievable live return. Removing
  this would need a point-in-time constituent history, which is not available
  from the free data source used here.
* Yahoo Finance data can be delayed, incomplete, or unavailable for some tickers.
* The backtest ignores market impact, order-book dynamics and borrow costs.
* Equity returns are close to unforecastable at these horizons. A cross-sectional
  IC of 0.02–0.05 is a normal, usable result; anything much larger on this kind
  of feature set should be treated as a bug or a leak, not a discovery.
* Fundamental / earnings features are disabled: the available source is a
  present-day snapshot rather than a point-in-time historical feed, and using it
  would introduce hindsight bias.

---

## Academic disclaimer

The system predicts simulated future returns using historical price data,
technical indicators, benchmark context and macro data. It is evaluated using
regression metrics, uncertainty-calibration metrics and backtesting, for academic
comparison only. It is not investment advice.
