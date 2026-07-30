import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data_download import preload_training_data
from src.features import (
    add_benchmark_features,
    add_labels,
    add_macro_features,
    add_technical_indicators,
    get_feature_columns,
)
from src.pipeline import load_dataset_cache, save_dataset_cache


def make_ohlcv_frame(start="2018-01-01", end="2025-01-01", ticker="TEST") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, end=end)
    steps = np.arange(len(dates), dtype=float)
    close = 100.0 + 0.03 * steps + 2.0 * np.sin(steps / 30.0)

    return pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.010,
            "Low": close * 0.990,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000 + (steps % 50) * 1_000,
            "Ticker": ticker,
        },
        index=dates,
    )


class DataPipelineValidationTest(unittest.TestCase):
    def test_processed_dataset_has_required_coverage_and_external_features(self):
        stock_df = make_ohlcv_frame(ticker="TEST")
        benchmark_df = make_ohlcv_frame(ticker="SPY")

        macro_df = pd.DataFrame(
            {
                "macro_vix_close": 20.0 + np.sin(np.arange(len(stock_df)) / 15.0),
                "macro_vix_return_1d": pd.Series(
                    20.0 + np.sin(np.arange(len(stock_df)) / 15.0),
                    index=stock_df.index,
                ).pct_change(),
                "macro_treasury_10y_close": 4.0 + 0.1 * np.cos(np.arange(len(stock_df)) / 40.0),
                "macro_usd_index_close": 100.0 + 0.2 * np.sin(np.arange(len(stock_df)) / 20.0),
            },
            index=stock_df.index,
        ).ffill()

        processed = add_technical_indicators(stock_df)
        processed = add_benchmark_features(processed, benchmark_df)
        processed = add_macro_features(processed, macro_df)
        processed = add_labels(processed, horizon=10, buy_threshold=0.03, sell_threshold=-0.03)

        feature_columns = get_feature_columns(processed)
        usable = processed.dropna(subset=feature_columns + ["future_return"]).copy()

        coverage_years = (usable.index.max() - usable.index.min()).days / 365.25
        macro_columns = [column for column in feature_columns if column.startswith("macro_")]

        self.assertGreaterEqual(coverage_years, 3.0)
        self.assertGreaterEqual(len(macro_columns), 3)
        self.assertIn("benchmark_return_1d", feature_columns)
        self.assertFalse(usable[feature_columns].isna().any().any())
        self.assertFalse(usable["future_return"].isna().any())
        self.assertGreater((usable["future_return"] > 0).sum(), 0)
        self.assertGreater((usable["future_return"] < 0).sum(), 0)
        self.assertGreater(len(usable), 500)

    def test_inference_builds_features_for_unseen_ticker(self):
        try:
            from predict import build_latest_features
        except ModuleNotFoundError as exc:
            self.skipTest(f"prediction dependencies are not installed: {exc.name}")

        # Inference now needs a universe, not a single ticker: cross-sectional
        # ranks, breadth, dispersion and the sector composites are defined
        # relative to the other stocks on the same date, so they cannot be
        # produced for one ticker in isolation.
        config = {
            "start_date": "2018-01-01",
            "end_date": None,
            "benchmark_ticker": "SPY",
            "tickers": ["NFLX", "AAPL", "MSFT", "XOM"],
            "ticker_sectors": {
                "NFLX": "Communication Services",
                "AAPL": "Information Technology",
                "MSFT": "Information Technology",
                "XOM": "Energy",
            },
            "minimum_sector_members": 2,
            "macro_tickers": {
                "vix": "^VIX",
                "treasury_10y": "^TNX",
                "usd_index": "DX-Y.NYB",
            },
        }
        feature_columns = [
            "return_1d",
            "sma_20",
            "rsi_14",
            "benchmark_return_1d",
            "benchmark_volatility_20d",
            "macro_vix_close",
            "macro_treasury_10y_close",
            "macro_usd_index_close",
            "is_earnings_day",
            "days_to_earnings",
        ]

        def fake_download_price_data(ticker, start, end):
            return make_ohlcv_frame(ticker=ticker)

        macro_index = make_ohlcv_frame().index
        fake_macro_df = pd.DataFrame(
            {
                "macro_vix_close": 19.0,
                "macro_treasury_10y_close": 4.2,
                "macro_usd_index_close": 101.0,
            },
            index=macro_index,
        )

        with patch("predict.download_price_data", side_effect=fake_download_price_data), patch(
            "predict.download_macro_data",
            return_value=fake_macro_df,
        ), patch("predict.download_earnings_data", return_value=pd.DataFrame()):
            features = build_latest_features("NFLX", config, feature_columns, horizon=10)

        self.assertGreaterEqual(len(features), 500)
        self.assertEqual(features["Ticker"].iloc[-1], "NFLX")
        self.assertFalse(features[feature_columns].isna().any().any())

    def test_inference_rejects_a_ticker_outside_the_universe(self):
        """
        A ticker with no cross-section cannot be served, and must say so clearly.

        Cross-sectional features are defined relative to the universe, so a name
        that is not in it has no rank, no sector composite and no dispersion
        context. Failing with an explanation beats silently emitting a forecast
        built from missing context.
        """
        try:
            from predict import build_latest_features, build_universe_panel
        except ModuleNotFoundError as exc:
            self.skipTest(f"prediction dependencies are not installed: {exc.name}")

        import predict

        predict._UNIVERSE_PANEL_CACHE.clear()
        config = {
            "start_date": "2018-01-01",
            "end_date": None,
            "benchmark_ticker": "SPY",
            "tickers": ["AAPL", "MSFT"],
            "minimum_sector_members": 2,
            "macro_tickers": {},
        }

        def fake_download_price_data(ticker, start, end):
            return make_ohlcv_frame(ticker=ticker)

        with patch("predict.download_price_data", side_effect=fake_download_price_data), patch(
            "predict.download_macro_data", return_value=pd.DataFrame()
        ), patch("predict.download_earnings_data", return_value=pd.DataFrame()):
            with self.assertRaises(ValueError) as caught:
                build_latest_features("TSLA", config, ["return_1d"], horizon=10)
        self.assertIn("universe", str(caught.exception).lower())
        predict._UNIVERSE_PANEL_CACHE.clear()

    def test_dataset_cache_roundtrip_preserves_features(self):
        stock_df = make_ohlcv_frame(ticker="CACHE")
        processed = add_technical_indicators(stock_df)
        processed = add_labels(processed, horizon=21, buy_threshold=0.03, sell_threshold=-0.03)
        feature_columns = get_feature_columns(processed)
        usable = processed.dropna(subset=feature_columns + ["future_return"]).copy()

        with TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "full_dataset_h21.csv"
            save_dataset_cache(
                usable,
                feature_columns,
                cache_path,
                metadata={"prediction_horizon": 21, "buy_threshold": 0.03, "sell_threshold": -0.03},
            )
            loaded, loaded_feature_columns, loaded_metadata = load_dataset_cache(cache_path)

        self.assertEqual(feature_columns, loaded_feature_columns)
        self.assertEqual(len(usable), len(loaded))
        self.assertEqual(list(usable.columns), list(loaded.columns))
        self.assertEqual(loaded_metadata["prediction_horizon"], 21)

    def test_preload_training_data_skips_unavailable_tickers_without_failing(self):
        def fake_download_price_data(ticker, start, end):
            if ticker == "BAD":
                raise ValueError("No price data returned for ticker: BAD")
            return make_ohlcv_frame(ticker=ticker)

        with patch("src.data_download.download_price_data", side_effect=fake_download_price_data), patch(
            "src.data_download.download_macro_data",
            return_value=pd.DataFrame(),
        ), patch("src.data_download.download_earnings_data", return_value=pd.DataFrame()):
            preload_training_data(
                tickers=["GOOD", "BAD"],
                benchmark_ticker="SPY",
                start="2018-01-01",
                end=None,
                macro_tickers={},
            )


if __name__ == "__main__":
    unittest.main()
