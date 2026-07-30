"""Tests for the dashboard's rendering helpers and model-comparison logic."""

import re
import unittest

import app
from src.decision import BUY, HOLD, SELL


def make_result(model: str, expected_pct: float, signal: str, confidence: str = "Low") -> dict:
    return {
        "model": model,
        "signal": signal,
        "expected_return_pct": expected_pct,
        "lower_bound_pct": expected_pct - 12.0,
        "upper_bound_pct": expected_pct + 12.0,
        "confidence_level": 0.80,
        "confidence_label": confidence,
        "confidence_explanation": "explanation text",
        "direction_probability": 0.55,
        "market_drift_pct": 0.76,
        "model_excess_return_pct": expected_pct - 0.76,
        "forecast_sigma_pct": 10.0,
        "signal_threshold_pct": 2.92,
        "latest_data_date": "2026-07-29",
        "prediction_horizon_trading_days": 21,
    }


class HtmlRenderingTest(unittest.TestCase):
    """
    Regression guard for the reported bug where markup appeared as literal text.

    Markdown terminates an HTML block at a blank line, and a following line
    indented four or more spaces becomes an indented code block. Any template
    reaching Streamlit must therefore contain neither.
    """

    def test_compaction_removes_newlines_and_indentation(self):
        markup = """
            <div class="a">

                <div class="b">value</div>

            </div>
        """
        compact = app.compact_html(markup)
        self.assertNotIn("\n", compact)
        self.assertFalse(re.search(r" {4,}<", compact))
        self.assertTrue(compact.startswith("<div"))
        self.assertTrue(compact.endswith("</div>"))

    def test_every_card_template_is_safe_to_render(self):
        card = app.model_card_html(make_result("LSTM", 1.08, HOLD))
        header = app.range_bar_html(-12.8, 1.08, 15.0)
        for markup in (card, header):
            compact = app.compact_html(markup)
            self.assertNotIn("\n", compact)
            self.assertFalse(re.search(r" {4,}", compact))

    def test_card_contains_the_values_it_was_given(self):
        compact = app.compact_html(app.model_card_html(make_result("XGBoost", -2.50, SELL)))
        self.assertIn("XGBoost", compact)
        self.assertIn("-2.50%", compact)
        self.assertIn("SELL", compact)
        # Tag names must survive as markup, not as escaped text.
        self.assertNotIn("&lt;div", compact)


class ColourLogicTest(unittest.TestCase):
    def test_positive_is_green_negative_is_red_zero_is_neutral(self):
        self.assertEqual(app.return_color(2.1), app.POSITIVE)
        self.assertEqual(app.return_color(-2.1), app.NEGATIVE)
        self.assertEqual(app.return_color(0.0), app.NEUTRAL)

    def test_signal_badges_follow_the_same_scheme(self):
        self.assertEqual(app.SIGNAL_STYLE[BUY]["color"], app.POSITIVE)
        self.assertEqual(app.SIGNAL_STYLE[SELL]["color"], app.NEGATIVE)
        self.assertEqual(app.SIGNAL_STYLE[HOLD]["color"], app.NEUTRAL)


class RangeBarTest(unittest.TestCase):
    def test_a_degenerate_range_still_produces_a_visible_bar(self):
        """Guards the same class of bug as the sidebar RangeError: min == max."""
        for lower, expected, upper in [(0.0, 0.0, 0.0), (5.0, 5.0, 5.0), (-3.0, -3.0, -3.0)]:
            width = float(re.search(r"width:([\d.]+)%", app.range_bar_html(lower, expected, upper)).group(1))
            self.assertGreater(width, 0.0)

    def test_positions_stay_within_the_track(self):
        markup = app.range_bar_html(-40.0, 12.0, 55.0)
        for value in re.findall(r"(?:left|width):([\d.]+)%", markup):
            self.assertGreaterEqual(float(value), 0.0)
            self.assertLessEqual(float(value), 100.0)


class ComparisonChartTest(unittest.TestCase):
    """
    Regression guard for the faceted-chart crash.

    Altair refuses to facet a layered chart whose layers come from different data
    sources. Building the spec is deferred until render time, so only calling
    ``to_dict()`` actually exercises that validation.
    """

    def results(self, tickers, models):
        return {
            ticker: {
                "models": {name: make_result(name, 1.5, HOLD) for name in models},
                "errors": [],
            }
            for ticker in tickers
        }

    def test_spec_builds_for_one_ticker_and_one_model(self):
        frame = app.comparison_chart_frame(self.results(["AAPL"], ["LSTM"]))
        app.comparison_chart(frame).to_dict()

    def test_spec_builds_for_several_tickers_and_both_models(self):
        frame = app.comparison_chart_frame(
            self.results(["AAPL", "MSFT", "NVDA"], ["LSTM", "XGBoost"])
        )
        self.assertEqual(len(frame), 6)
        app.comparison_chart(frame).to_dict()

    def test_all_layers_share_one_data_source(self):
        frame = app.comparison_chart_frame(self.results(["AAPL", "MSFT"], ["LSTM", "XGBoost"]))
        spec = app.comparison_chart(frame).to_dict()
        # Data must sit at the top level, not be repeated per layer.
        self.assertIn("data", spec)
        layers = spec.get("spec", {}).get("layer", [])
        self.assertTrue(layers)
        for layer in layers:
            self.assertNotIn("data", layer)

    def test_frame_is_empty_when_no_model_produced_a_result(self):
        frame = app.comparison_chart_frame({"AAPL": {"models": {}, "errors": ["LSTM: boom"]}})
        self.assertTrue(frame.empty)


class ConsensusTest(unittest.TestCase):
    def entry(self, models):
        return {"models": models, "errors": []}

    def test_agreement_when_both_models_share_direction_and_signal(self):
        consensus = app.build_consensus(
            self.entry({"LSTM": make_result("LSTM", 3.5, BUY), "XGBoost": make_result("XGBoost", 3.9, BUY)})
        )
        self.assertEqual(consensus["state"], "strong")
        self.assertTrue(consensus["agree_direction"])
        self.assertAlmostEqual(consensus["difference_pp"], 0.4, places=6)

    def test_same_direction_but_different_signal_is_flagged_separately(self):
        consensus = app.build_consensus(
            self.entry({"LSTM": make_result("LSTM", 1.1, HOLD), "XGBoost": make_result("XGBoost", 3.9, BUY)})
        )
        self.assertEqual(consensus["state"], "directional")
        self.assertTrue(consensus["agree_direction"])
        self.assertFalse(consensus["agree_signal"])

    def test_opposite_directions_are_reported_as_conflict(self):
        consensus = app.build_consensus(
            self.entry({"LSTM": make_result("LSTM", 1.5, HOLD), "XGBoost": make_result("XGBoost", -2.2, HOLD)})
        )
        self.assertEqual(consensus["state"], "conflict")
        self.assertFalse(consensus["agree_direction"])
        self.assertEqual(consensus["color"], app.NEGATIVE)

    def test_reliable_requires_agreement_and_confidence(self):
        low = app.build_consensus(
            self.entry({"LSTM": make_result("LSTM", 5.0, BUY), "XGBoost": make_result("XGBoost", 5.4, BUY)})
        )
        high = app.build_consensus(
            self.entry(
                {
                    "LSTM": make_result("LSTM", 5.0, BUY, "High"),
                    "XGBoost": make_result("XGBoost", 5.4, BUY, "High"),
                }
            )
        )
        self.assertFalse(low["reliable"])
        self.assertTrue(high["reliable"])

    def test_single_model_and_no_model_degrade_gracefully(self):
        single = app.build_consensus(self.entry({"LSTM": make_result("LSTM", 2.0, HOLD)}))
        self.assertEqual(single["state"], "single")
        self.assertIsNone(single["difference_pp"])

        none = app.build_consensus(self.entry({}))
        self.assertEqual(none["state"], "none")
        self.assertEqual(none["label"], "No prediction")


class HorizonOptionsTest(unittest.TestCase):
    def test_all_offered_horizons_have_labels(self):
        for horizon in app.HORIZON_OPTIONS:
            self.assertIn(str(horizon), app.format_horizon(horizon))
            self.assertNotEqual(app.format_horizon(horizon), f"{horizon} trading days")

    def test_selector_supports_a_single_trained_horizon(self):
        """A select_slider with one option raises RangeError; a selectbox does not."""
        self.assertGreaterEqual(len(app.HORIZON_OPTIONS), 2)
        self.assertEqual(len(set(app.HORIZON_OPTIONS)), len(app.HORIZON_OPTIONS))
        self.assertEqual(app.HORIZON_OPTIONS, sorted(app.HORIZON_OPTIONS))


class TickerParsingTest(unittest.TestCase):
    def test_separators_and_duplicates_are_handled(self):
        self.assertEqual(app.parse_tickers("aapl, MSFT\nnvda; aapl  tsla"), ["AAPL", "MSFT", "NVDA", "TSLA"])
        self.assertEqual(app.parse_tickers("   "), [])


if __name__ == "__main__":
    unittest.main()
