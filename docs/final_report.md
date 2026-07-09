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
For the 10-trading-day horizon run, both models used 58 engineered features and the same chronological data-building pipeline.

| Metric | LSTM | XGBoost |
| --- | ---: | ---: |
| Accuracy | 0.2980 | 0.3259 |
| Return MAE | 0.0568 | 0.0664 |
| Strategy total return | 0.00% | 146.59% |
| Buy-and-hold total return | 764.23% | 1173.96% |
| Excess return vs buy-and-hold | -764.23% | -1027.37% |
| Max drawdown | 0.00% | -55.09% |
| Sharpe ratio | 0.0000 | 2.1212 |
| Sortino ratio | 0.0000 | 2.2935 |
| Win rate | 0.00% | 59.62% |
| Active trades | 0 | 832 |

XGBoost had higher classification accuracy and generated active trading signals in the simulation. However, both strategies underperformed the buy-and-hold benchmark over this test period after costs. This is important for the academic conclusion: better classification metrics do not automatically imply positive alpha versus a strong market benchmark.

## Limitations
- Yahoo Finance data quality can vary by ticker and by date range.
- The backtest is simplified and does not model execution microstructure.
- Results should not be interpreted as investment advice.

## Future Work
- Add more alternative data sources
- Test additional model families
- Improve position sizing and risk controls
- Add walk-forward validation
