# Algo Trading Stock Predictor

Academic final project for Track 2, building a machine learning system that predicts stock movement and evaluates the result as a trading strategy.

The repo now contains two model families that share the same chronological data pipeline:

1. LSTM multi-task model over 60-day sequential windows.
2. XGBoost classifier/regressor over engineered tabular financial features.

Both models predict:

- `BUY`
- `HOLD`
- `SELL`
- expected future return

The feature set includes historical prices, technical indicators, benchmark market context, earnings features, and macro / alternative market data such as VIX, treasury yield proxy, and USD index.

> This project is for academic research and simulation only. It is not financial advice.

---

## Track 2 Coverage

The implementation is designed to satisfy the course requirements:

- At least 3 years of historical market data.
- Alternative / macro data joined to each asset.
- Two model families for comparison.
- EDA, training, inference, and evaluation.
- Backtest metrics, including alpha-style comparison against buy-and-hold.
- Clean repository outputs for demo and presentation use.

---

## Project structure

```text
algo_trading_advanced_model/
|-- app.py
|-- train.py
|-- train_xgboost.py
|-- predict.py
|-- check_device.py
|-- compare_models.py
|-- compare_results.py
|-- evaluate_saved_model.py
|-- configs/
|   `-- config.yaml
|-- docs/
|   |-- demo_script.md
|   `-- final_report.md
|-- notebooks/
|   `-- final_project_colab.ipynb
|-- src/
|   |-- backtest.py
|   |-- baselines.py
|   |-- data_download.py
|   |-- dataset.py
|   |-- device.py
|   |-- features.py
|   |-- model.py
|   |-- pipeline.py
|   `-- plots.py
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

Use `python` instead of `python3` on Windows if needed.

---

## Training

### Train the LSTM

This trains the LSTM model for the configured horizon set:

```bash
python3 train.py
```

The configured horizon set is:

| Meaning | Trading-day horizon |
|---|---:|
| 1 day | 1 |
| 1 week | 5 |
| 1 month | 21 |
| 6 months | 126 |
| 1 year | 252 |
| 5 years | 1260 |
| 10 years | 2520 |

To train one horizon only:

```bash
python3 train.py --horizon 21
```

### Train the XGBoost model

```bash
python3 train_xgboost.py --horizon 21
```

### Run the full comparison

```bash
python3 compare_models.py --horizon 21
```

This runs:

```bash
python train.py --horizon 21
python train_xgboost.py --horizon 21
python compare_results.py --horizon 21
```

---

## Outputs

Training and evaluation create horizon-specific artifacts under `models/`, `reports/`, and `data/processed/`.

Examples include:

```text
models/stock_advanced_model_h21.pt
models/xgboost_classifier_h21.joblib
models/xgboost_regressor_h21.joblib
models/model_metadata_h21.json
reports/metrics_h21.json
reports/metrics_xgboost_h21.json
reports/test_predictions_h21.csv
reports/test_predictions_xgboost_h21.csv
reports/backtest_results_h21.csv
reports/backtest_results_xgboost_h21.csv
reports/model_comparison_h21.csv
reports/model_comparison_h21.md
```

---

## Backtesting Methodology

The backtest is intentionally simple and transparent for academic use:

- Each prediction is treated as one simulated trading decision.
- Transaction costs and slippage are subtracted from active trades.
- Performance is reported against buy-and-hold.
- The report includes alpha-style excess return, drawdown, Sharpe, Sortino, win rate, and average trade return.

---

## Inference

Single ticker:

```bash
python3 predict.py --ticker AAPL --horizon 21
```

Multiple tickers:

```bash
python3 predict.py --tickers AAPL,MSFT,NVDA --horizon 252
```

---

## Run the UI

```bash
python3 -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

In the UI, the user can:

- Enter one stock ticker or many tickers
- Choose the prediction horizon
- Run prediction
- View BUY / HOLD / SELL
- View expected movement
- View confidence
- View class probabilities
- Download predictions as CSV

---

## Limitations

- This is a research simulation, not a production trading system.
- Yahoo Finance data can be delayed, incomplete, or unavailable for some tickers.
- The backtest is not a substitute for live execution, order-book modeling, or risk management.
- Model output can be unstable across horizons and market regimes.

---

## Important note about horizons

The UI can only predict a horizon after a model was trained for that horizon.

For example, if you choose 1 month / 21 trading days in the UI, you need this file:

```text
models/stock_advanced_model_h21.pt
```

If it does not exist, run:

```bash
python3 train.py --horizon 21
```

---

## Academic Disclaimer

The dashboard and reports should be described as a research simulation system, not as investment advice.

Suggested wording:

> The system predicts simulated trading signals using historical price data, technical indicators, benchmark features, macro data, and earnings-related features. It is evaluated using classification metrics and backtesting for academic comparison only.
