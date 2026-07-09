# Demo Script

## 1. Opening
This project is an academic simulation for predicting stock movement using machine learning and evaluating the result with a trading backtest.

## 2. Data
Explain that the system uses at least three years of Yahoo Finance historical data, plus benchmark and macro indicators such as VIX, treasury yield proxy, and USD index.

## 3. Feature Engineering
Show that the feature pipeline includes technical indicators, benchmark features, macro features, and earnings features.

## 4. Models
Explain that the project compares two model families:
- LSTM for sequential price windows
- XGBoost for engineered tabular features

## 5. Evaluation
Show both statistical and financial evaluation:
- Accuracy and return MAE
- Total return, drawdown, Sharpe, Sortino
- Buy-and-hold comparison
- Win rate and average trade return

## 6. Live Demo Flow
1. Run `python train.py --horizon 10`
2. Run `python train_xgboost.py --horizon 10`
3. Run `python compare_results.py --horizon 10`
4. Open the Streamlit app
5. Enter a ticker and show the prediction card and probabilities

## 7. Closing
Stress that the output is for academic research only and not financial advice.
