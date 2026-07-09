# Final Project Report

## Project Goal
Build an academic machine learning system for stock movement prediction and trading-style evaluation.

## Problem Definition
The project estimates whether a financial asset will move up, down, or remain neutral over a fixed trading horizon. The output is evaluated both as a classification problem and as a simulated trading strategy.

## Data Sources
- Historical OHLCV data from Yahoo Finance
- Benchmark data from SPY
- Macro / alternative data: VIX, 10-year Treasury yield proxy, USD index
- Earnings-related features where available

## Feature Engineering
- Technical indicators: returns, momentum, volatility, moving averages, MACD, RSI, Bollinger Bands, ATR
- Market context: benchmark returns and benchmark volatility
- Macro context: macro close, macro returns, and macro volatility
- Earnings features: event proximity and surprise-related fields

## Models
### LSTM
A multi-task sequential model using 60-day windows.

### XGBoost
A tabular model trained on the same chronological split and feature set.

## Evaluation
### Statistical Metrics
- Accuracy
- Return MAE
- Confusion matrix
- Classification report

### Financial Metrics
- Total return
- Buy-and-hold return
- Excess return vs buy-and-hold
- Maximum drawdown
- Sharpe ratio
- Sortino ratio
- Win rate
- Average trade return
- Trade activation rate

## Backtest Methodology
Each prediction is treated as a simulated trade. Transaction costs and slippage are deducted from active positions. The backtest is compared with buy-and-hold to estimate alpha after costs.

## Results Summary
Add horizon-specific results here after running the training scripts.

## Limitations
- Yahoo Finance data quality can vary by ticker and by date range.
- The backtest is simplified and does not model execution microstructure.
- Results should not be interpreted as investment advice.

## Future Work
- Add more alternative data sources
- Test additional model families
- Improve position sizing and risk controls
- Add walk-forward validation
