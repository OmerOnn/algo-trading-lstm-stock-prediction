# Algo Trading Stock Predictor

This project is an academic final-project pipeline for stock movement prediction in the context of AI and Innovation in Capital Markets. It combines financial data, feature engineering, machine learning, and a simple trading backtest in a research-oriented workflow.

## Project goal
The goal is to predict short-horizon stock movement as a simulated classification and regression task. The system produces BUY / HOLD / SELL signals and an expected future return for a selected horizon.

## Dataset
The pipeline downloads daily OHLCV data from Yahoo Finance for selected stocks and benchmark market data. It also adds alternative and macro information, including VIX, 10-year Treasury yields, and the US dollar index, as well as earnings-related features.

## Feature engineering
The model uses technical indicators, benchmark-market features, macro features, and earnings features. The feature pipeline is chronological and avoids random time-based splitting.

## Models
The repository keeps the existing LSTM flow and adds XGBoost as a second model:
- LSTM: sequence-based deep learning model for classification and return regression
- XGBoost: gradient-boosted decision tree model for the same targets

## Backtesting methodology
The project evaluates predictions with a simple event-based backtest that uses realized future returns and includes transaction costs and slippage. It reports total return, drawdown, Sharpe ratio, Sortino ratio, win rate, and alpha relative to buy-and-hold.

## Output files
Training and evaluation produce artifacts in the models and reports folders, including:
- trained LSTM and XGBoost models
- metrics JSON files
- test prediction CSV files
- backtest CSV files
- model comparison reports

## How to run
```bash
pip install -r requirements.txt
python train.py --horizon 10
python train_xgboost.py --horizon 10
python compare_results.py --horizon 10
python evaluate_saved_model.py --model lstm --horizon 10
python evaluate_saved_model.py --model xgboost --horizon 10
streamlit run app.py
```

## Limitations
This project is a research simulation only. It is not financial advice, and results can be affected by data quality, label design, changing market regimes, and transaction costs.

## Academic disclaimer
This repository is intended for academic study and demonstration in a final project setting. It should not be interpreted as investment advice.
