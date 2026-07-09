# Final Project Report

## Goal
This project explores an academic stock movement prediction pipeline using both an LSTM neural network and an XGBoost model. The objective is not to provide investment advice, but to study whether structured financial features can support directional predictions in a simulated setting.

## Dataset
The pipeline downloads daily OHLCV data from Yahoo Finance for selected stocks and benchmark ETFs, plus macro indicators and earnings dates.

## Feature Engineering
Technical indicators, benchmark returns, macro features, and earnings features are combined into a chronological supervised-learning dataset.

## Modeling
The LSTM model predicts BUY/HOLD/SELL classes and future returns. XGBoost is trained for both classification and regression using the same chronological split.

## Evaluation
Performance is assessed with statistical metrics and a simple event-based backtest that includes transaction and slippage costs.

## Limitations
Financial data is noisy, label definitions are simplifying assumptions, and results are only a research simulation.
