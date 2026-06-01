# Algo Trading Stock Predictor

Academic deep-learning project for simulated stock signal prediction.

The project trains an LSTM-based model that predicts:

- `BUY`
- `HOLD`
- `SELL`
- expected percentage movement

The model uses historical prices, technical indicators, volatility features, benchmark-market features, and earnings-related features.

> This project is for academic research and simulation only. It is not financial advice.

---

## Main changes in this version

This version includes the requested fixes:

1. The project uses **LSTM** by default.
2. Training stops early if validation loss does not improve for **7 straight epochs**.
3. The UI no longer shows the internal model type, GPU, CUDA, MPS, or device information to the user.
4. The UI lets the user choose the prediction horizon using a bar/slider.
5. The project supports several prediction horizons: `5`, `10`, `20`, and `30` trading days.
6. The UI displays expected movement consistently with the selected signal.
   - `BUY` is shown as positive movement.
   - `SELL` is shown as negative movement.
   - `HOLD` is shown as neutral/raw expected movement.

---

## Project structure

```text
algo_trading_advanced_model/
├── app.py
├── train.py
├── predict.py
├── check_device.py
├── compare_models.py
├── evaluate_saved_model.py
├── configs/
│   └── config.yaml
├── src/
│   ├── backtest.py
│   ├── baselines.py
│   ├── data_download.py
│   ├── dataset.py
│   ├── device.py
│   ├── features.py
│   ├── model.py
│   ├── pipeline.py
│   └── plots.py
├── models/
├── reports/
└── requirements.txt
```

---

## Setup on Mac

```bash
cd algo_trading_advanced_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check acceleration:

```bash
python3 check_device.py
```

On Apple Silicon Mac, it should show `mps`.

---

## Training

### Train all configured horizons

This trains separate LSTM models for all horizons in `configs/config.yaml`:

```bash
python3 train.py
```

By default, it trains models for:

```text
5, 10, 20, 30 trading days ahead
```

This creates files like:

```text
models/stock_advanced_model_h5.pt
models/stock_advanced_model_h10.pt
models/stock_advanced_model_h20.pt
models/stock_advanced_model_h30.pt

models/feature_scaler_h5.pkl
models/feature_scaler_h10.pkl
models/feature_scaler_h20.pkl
models/feature_scaler_h30.pkl

models/model_metadata_h5.json
models/model_metadata_h10.json
models/model_metadata_h20.json
models/model_metadata_h30.json
```

### Train only one horizon

For faster testing:

```bash
python3 train.py --horizon 10
```

or:

```bash
python3 train.py --horizon 5
```

---

## Early stopping

The model trains up to the maximum number of epochs:

```yaml
epochs: 50
```

But it stops earlier if validation loss does not improve for 7 straight epochs:

```yaml
early_stopping_patience: 7
```

So training stops when either:

```text
1. It reaches the maximum number of epochs
2. Validation loss does not improve for 7 consecutive epochs
```

---

## Prediction from terminal

Single ticker:

```bash
python3 predict.py --ticker AAPL --horizon 10
```

Multiple tickers:

```bash
python3 predict.py --tickers AAPL,MSFT,NVDA --horizon 20
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

## Important note about horizons

The UI can only predict a horizon after a model was trained for that horizon.

For example, if you choose 20 trading days in the UI, you need this file:

```text
models/stock_advanced_model_h20.pt
```

If it does not exist, run:

```bash
python3 train.py --horizon 20
```

---

## Academic interpretation

The dashboard should be described as a research simulation system, not as a real investment-advice system.

A good project explanation:

> The system predicts simulated trading signals using sequential market data, technical indicators, volatility features, benchmark data, and earnings-related features. The output is evaluated academically using classification metrics and backtesting, and should not be interpreted as financial advice.
