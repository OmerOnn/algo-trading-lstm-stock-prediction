"""
Streamlit dashboard for the future-return regression models.

Two rules drive the layout:

* A point forecast is never shown on its own. Every expected movement appears
  with its calibrated range and an explicit confidence statement.
* Both model families are shown side by side for the same ticker and horizon,
  because agreement (or disagreement) between two independent models is itself
  evidence a user should see.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict import (  # noqa: E402
    artifact_path,
    available_models_for_horizon,
    load_config,
    load_model_and_metadata,
    load_xgboost_model_and_metadata,
    predict_ticker_with_artifacts,
    predict_xgboost_ticker_with_artifacts,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Horizons offered in the UI. A horizon is listed whether or not it has been
# trained; the sidebar states which models are available for it.
HORIZON_OPTIONS = [1, 5, 21, 126]
HORIZON_LABELS = {
    1: "1 trading day",
    5: "1 week (5 trading days)",
    21: "1 month (21 trading days)",
    126: "6 months (126 trading days)",
}

MODEL_ORDER = ["LSTM", "XGBoost"]
MODEL_BLURB = {
    "LSTM": "Sequence model over recent trading windows. Uncertainty from Monte Carlo dropout.",
    "XGBoost": "Gradient-boosted trees over engineered indicators. Uncertainty from a bootstrap ensemble.",
}

POSITIVE = "#16a34a"
NEGATIVE = "#dc2626"
NEUTRAL = "#ca8a04"

SIGNAL_STYLE = {
    "BUY": {"emoji": "▲", "color": POSITIVE, "css": "buy", "explain": "The risk-adjusted edge clears the cost hurdle on the upside."},
    "SELL": {"emoji": "▼", "color": NEGATIVE, "css": "sell", "explain": "The risk-adjusted edge clears the cost hurdle on the downside."},
    "HOLD": {"emoji": "■", "color": NEUTRAL, "css": "hold", "explain": "The expected move is too small next to its uncertainty and trading costs to act on."},
}

CONFIDENCE_STYLE = {
    "High": {"css": "conf-high", "icon": "●●●"},
    "Moderate": {"css": "conf-moderate", "icon": "●●○"},
    "Low": {"css": "conf-low", "icon": "●○○"},
}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def compact_html(markup: str) -> str:
    """
    Collapse markup to a single line before handing it to Streamlit.

    Streamlit renders markdown, and markdown ends an HTML block at a blank line;
    any following line indented four or more spaces then becomes an *indented
    code block* and is printed literally. Multi-line f-string templates hit this
    constantly. Stripping every line and removing blank ones makes the output a
    single HTML line, which always renders as markup.
    """
    return "".join(line.strip() for line in markup.strip().splitlines() if line.strip())


def render_html(markup: str) -> None:
    st.markdown(compact_html(markup), unsafe_allow_html=True)


def format_horizon(horizon: int) -> str:
    return HORIZON_LABELS.get(int(horizon), f"{horizon} trading days")


def return_color(value: float) -> str:
    if value > 0:
        return POSITIVE
    if value < 0:
        return NEGATIVE
    return NEUTRAL


def parse_tickers(raw_text: str) -> list[str]:
    cleaned = raw_text.replace("\n", ",").replace(";", ",").replace(" ", ",")
    tickers: list[str] = []
    for item in cleaned.split(","):
        ticker = item.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1320px; }
        section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #111827 52%, #020617 100%); }
        section[data-testid="stSidebar"] * { color: #e5e7eb !important; }

        .hero { border-radius: 26px; padding: 1.8rem 2rem; background: radial-gradient(circle at top left, #dbeafe 0%, #eff6ff 28%, #f8fafc 70%); border: 1px solid #dbeafe; box-shadow: 0 18px 44px rgba(15,23,42,0.08); margin-bottom: 1.2rem; }
        .hero-title { font-size: 2.4rem; font-weight: 900; letter-spacing: -0.05em; color: #0f172a; margin-bottom: 0.45rem; }
        .hero-sub { color: #475569; font-size: 1rem; line-height: 1.6; font-weight: 600; max-width: 940px; }
        .section-title { font-size: 1.35rem; font-weight: 900; letter-spacing: -0.03em; color: #0f172a; margin: 1.6rem 0 0.7rem 0; }

        .ticker-header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; border-radius: 18px 18px 0 0; padding: 0.9rem 1.2rem; background: #0f172a; }
        .ticker-name { font-size: 1.55rem; font-weight: 900; color: #ffffff; letter-spacing: -0.03em; }
        .ticker-meta { color: #94a3b8; font-weight: 700; font-size: 0.78rem; }
        .consensus-pill { border-radius: 999px; padding: 0.4rem 0.9rem; font-weight: 900; font-size: 0.8rem; color: #fff; white-space: nowrap; }

        .model-card { border-radius: 16px; padding: 1.1rem 1.2rem; border: 1px solid #e2e8f0; background: #fff; box-shadow: 0 8px 22px rgba(15,23,42,0.06); height: 100%; }
        .model-card.buy { border-top: 5px solid #16a34a; }
        .model-card.sell { border-top: 5px solid #dc2626; }
        .model-card.hold { border-top: 5px solid #ca8a04; }
        .model-name { font-size: 1.02rem; font-weight: 900; color: #0f172a; letter-spacing: -0.01em; }
        .model-blurb { color: #64748b; font-size: 0.74rem; font-weight: 600; line-height: 1.35; margin-top: 0.15rem; min-height: 2.1rem; }
        .badge { display: inline-block; border-radius: 999px; padding: 0.28rem 0.75rem; font-weight: 900; font-size: 0.78rem; color: #fff; }
        .kpi-label { font-size: 0.64rem; font-weight: 900; color: #64748b; text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 0.15rem; }
        .kpi-big { font-size: 2.1rem; font-weight: 900; letter-spacing: -0.05em; line-height: 1.05; }
        .kpi-range { font-size: 0.95rem; font-weight: 800; color: #334155; margin-top: 0.3rem; }

        .range-track { position: relative; height: 40px; margin-top: 0.45rem; }
        .range-base { position: absolute; top: 17px; left: 0; right: 0; height: 7px; border-radius: 6px; background: #e2e8f0; }
        .range-span { position: absolute; top: 17px; height: 7px; border-radius: 6px; opacity: 0.75; }
        .range-zero { position: absolute; top: 9px; width: 2px; height: 23px; background: #0f172a; opacity: 0.5; }
        .range-point { position: absolute; top: 11px; width: 4px; height: 19px; border-radius: 2px; }
        .range-caption { display: flex; justify-content: space-between; font-size: 0.68rem; font-weight: 800; color: #64748b; }

        .chip { display: inline-flex; align-items: center; gap: 0.4rem; border-radius: 999px; padding: 0.28rem 0.7rem; font-weight: 900; font-size: 0.72rem; margin-top: 0.5rem; }
        .conf-high { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
        .conf-moderate { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
        .conf-low { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }

        .facts { display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem; margin-top: 0.85rem; }
        .fact { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 11px; padding: 0.5rem 0.6rem; }
        .fact-label { font-size: 0.6rem; font-weight: 900; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }
        .fact-value { font-size: 0.88rem; font-weight: 900; color: #0f172a; }
        .note { margin-top: 0.8rem; padding-top: 0.65rem; border-top: 1px solid #e2e8f0; color: #475569; font-weight: 600; line-height: 1.45; font-size: 0.79rem; }

        .verdict { border-radius: 0 0 18px 18px; padding: 0.85rem 1.2rem; background: #f8fafc; border: 1px solid #e2e8f0; border-top: none; color: #334155; font-weight: 650; font-size: 0.85rem; line-height: 1.5; }
        .verdict strong { color: #0f172a; }
        .missing { border-radius: 16px; padding: 1.1rem 1.2rem; border: 1px dashed #cbd5e1; background: #f8fafc; color: #64748b; font-weight: 700; font-size: 0.85rem; height: 100%; display: flex; align-items: center; }

        .info-box { border-radius: 14px; padding: 0.85rem 1rem; background: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a; font-weight: 700; margin-bottom: 0.9rem; font-size: 0.87rem; }
        div[data-testid="stTextArea"] textarea { border-radius: 14px; border: 1px solid #cbd5e1; background: #f8fafc; font-weight: 700; font-size: 1rem; }
        div.stButton > button:first-child { border-radius: 13px; background: linear-gradient(90deg, #2563eb, #1d4ed8); color: #fff; border: none; font-weight: 900; height: 3rem; box-shadow: 0 10px 24px rgba(37,99,235,0.25); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached artifact loading
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def cached_lstm(horizon: int):
    return load_model_and_metadata(load_config(), horizon=int(horizon))


@st.cache_resource(show_spinner=False)
def cached_xgboost(horizon: int):
    return load_xgboost_model_and_metadata(load_config(), horizon=int(horizon))


@st.cache_data(show_spinner=False)
def load_model_quality(horizon: int, model: str) -> dict | None:
    """Read the saved out-of-sample evaluation, or None if it is not there."""
    config = load_config()
    if model == "LSTM":
        path = artifact_path(config["metrics_output_path"], int(horizon))
    else:
        path = ROOT / "reports" / f"metrics_xgboost_h{int(horizon)}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def run_predictions(tickers: list[str], horizon: int, config: dict) -> tuple[dict, list[str]]:
    """
    Predict every ticker with every available model.

    Failures are collected per ticker and per model rather than raised, so one
    bad symbol or one missing checkpoint never blanks the whole dashboard.
    """
    availability = available_models_for_horizon(config, horizon)
    loaded: dict[str, object] = {}
    errors: list[str] = []

    if availability["LSTM"]:
        try:
            loaded["LSTM"] = cached_lstm(horizon)
        except Exception as exc:
            errors.append(f"LSTM model could not be loaded: {exc}")
    if availability["XGBoost"]:
        try:
            loaded["XGBoost"] = cached_xgboost(horizon)
        except Exception as exc:
            errors.append(f"XGBoost model could not be loaded: {exc}")

    results: dict[str, dict] = {}
    if not loaded:
        return results, errors

    total = len(tickers) * len(loaded)
    progress = st.progress(0.0, text="Preparing...")
    step = 0

    for ticker in tickers:
        entry: dict = {"models": {}, "errors": []}
        for name in MODEL_ORDER:
            if name not in loaded:
                continue
            step += 1
            progress.progress(step / total, text=f"Predicting {ticker} with {name}...")
            try:
                if name == "LSTM":
                    models, metadata, scaler, feature_columns, device = loaded[name]
                    entry["models"][name] = predict_ticker_with_artifacts(
                        ticker=ticker, config=config, model=models, metadata=metadata,
                        scaler=scaler, feature_columns=feature_columns, device=device,
                        horizon=horizon,
                    )
                else:
                    ensemble, metadata, feature_columns = loaded[name]
                    entry["models"][name] = predict_xgboost_ticker_with_artifacts(
                        ticker=ticker, config=config, ensemble=ensemble, metadata=metadata,
                        feature_columns=feature_columns, horizon=horizon,
                    )
            except Exception as exc:
                entry["errors"].append(f"{name}: {exc}")
        results[ticker] = entry

    progress.empty()
    return results, errors


# ---------------------------------------------------------------------------
# Consensus between the two models
# ---------------------------------------------------------------------------


def build_consensus(entry: dict) -> dict:
    """Summarise whether the two models tell the same story."""
    models = entry["models"]
    available = [name for name in MODEL_ORDER if name in models]

    if not available:
        return {
            "state": "none",
            "color": NEUTRAL,
            "label": "No prediction",
            "text": "No model produced a prediction for this ticker.",
            "difference_pp": None,
            "reliable": False,
        }

    if len(available) < 2:
        return {
            "state": "single",
            "color": NEUTRAL,
            "label": "Single model",
            "text": (
                "Only one model is available for this horizon, so no cross-model agreement "
                "check is possible. Train the other model family to enable the comparison."
            ),
            "difference_pp": None,
            "reliable": False,
        }

    first, second = models[available[0]], models[available[1]]
    a, b = first["expected_return_pct"], second["expected_return_pct"]
    difference = abs(a - b)
    same_direction = (a >= 0) == (b >= 0)
    same_signal = first["signal"] == second["signal"]
    both_confident = all(models[n]["confidence_label"] != "Low" for n in available)

    if same_direction and same_signal:
        state, label = "strong", "Models agree"
        color = SIGNAL_STYLE[first["signal"]]["color"]
        text = (
            f"Both models point the same way and derive the same <strong>{first['signal']}</strong> "
            f"signal, differing by {difference:.2f} percentage points."
        )
    elif same_direction:
        state, label, color = "directional", "Same direction", NEUTRAL
        text = (
            f"Both models expect a move in the same direction ({difference:.2f} pp apart), but they "
            f"derive different signals ({first['signal']} vs {second['signal']}), so the size of the "
            "edge is disputed."
        )
    else:
        state, label, color = "conflict", "Models disagree", NEGATIVE
        text = (
            f"The models disagree on direction ({a:+.2f}% vs {b:+.2f}%). Treat this ticker as "
            "inconclusive rather than picking the answer you prefer."
        )

    if not both_confident:
        text += (
            " At least one model rates its own forecast as low confidence, meaning the uncertainty "
            "band is wider than the expected move."
        )

    return {
        "state": state,
        "color": color,
        "label": label,
        "text": text,
        "difference_pp": difference,
        "agree_direction": same_direction,
        "agree_signal": same_signal,
        "reliable": same_direction and both_confident,
    }


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------


def range_bar_html(lower_pct: float, expected_pct: float, upper_pct: float) -> str:
    """Zero-anchored interval bar. Guards against a degenerate (zero-width) range."""
    span = max(abs(lower_pct), abs(upper_pct), 0.5)
    scale = span * 1.15

    def position(value: float) -> float:
        return min(100.0, max(0.0, (value + scale) / (2.0 * scale) * 100.0))

    left, right = position(lower_pct), position(upper_pct)
    width = max(right - left, 0.8)
    color = return_color(expected_pct)
    return f"""
        <div class="range-track">
            <div class="range-base"></div>
            <div class="range-span" style="left:{left:.2f}%;width:{width:.2f}%;background:{color};"></div>
            <div class="range-zero" style="left:50%;"></div>
            <div class="range-point" style="left:{position(expected_pct):.2f}%;background:{color};"></div>
        </div>
        <div class="range-caption"><span>{lower_pct:+.1f}%</span><span>0%</span><span>{upper_pct:+.1f}%</span></div>
    """


def model_card_html(result: dict) -> str:
    style = SIGNAL_STYLE[result["signal"]]
    confidence = CONFIDENCE_STYLE.get(result["confidence_label"], CONFIDENCE_STYLE["Low"])
    expected = float(result["expected_return_pct"])
    lower, upper = float(result["lower_bound_pct"]), float(result["upper_bound_pct"])

    return f"""
        <div class="model-card {style['css']}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.6rem;">
                <div>
                    <div class="model-name">{result['model']}</div>
                    <div class="model-blurb">{MODEL_BLURB.get(result['model'], '')}</div>
                </div>
                <span class="badge" style="background:{style['color']};">{style['emoji']} {result['signal']}</span>
            </div>
            <div class="kpi-label" style="margin-top:0.5rem;">Expected movement</div>
            <div class="kpi-big" style="color:{return_color(expected)};">{expected:+.2f}%</div>
            <div class="kpi-range">Estimated range: {lower:+.2f}% to {upper:+.2f}%</div>
            <div class="kpi-label" style="margin-top:0.7rem;">{float(result['confidence_level']):.0%} prediction interval</div>
            {range_bar_html(lower, expected, upper)}
            <div class="chip {confidence['css']}">{confidence['icon']} &nbsp;{result['confidence_label']} confidence &nbsp;·&nbsp; {result['direction_probability']:.0%} chance the direction is right</div>
            <div class="facts">
                <div class="fact"><div class="fact-label">Market baseline</div><div class="fact-value">{result['market_drift_pct']:+.2f}%</div></div>
                <div class="fact"><div class="fact-label">Model view vs market</div><div class="fact-value" style="color:{return_color(result['model_excess_return_pct'])};">{result['model_excess_return_pct']:+.2f}%</div></div>
                <div class="fact"><div class="fact-label">Uncertainty (sigma)</div><div class="fact-value">&plusmn;{result['forecast_sigma_pct']:.2f}%</div></div>
                <div class="fact"><div class="fact-label">Cost hurdle</div><div class="fact-value">{result['signal_threshold_pct']:.2f}%</div></div>
            </div>
            <div class="note"><strong>{result['signal']}</strong> &mdash; {style['explain']} {result['confidence_explanation']}</div>
        </div>
    """


def render_ticker_block(ticker: str, entry: dict) -> None:
    models = entry["models"]
    if not models:
        st.warning(f"{ticker}: no prediction could be produced. " + " ".join(entry["errors"]))
        return

    consensus = build_consensus(entry)
    any_result = next(iter(models.values()))

    render_html(
        f"""
        <div class="ticker-header">
            <div>
                <div class="ticker-name">{ticker}</div>
                <div class="ticker-meta">Latest market date {any_result['latest_data_date']}
                    &nbsp;·&nbsp; {any_result['prediction_horizon_trading_days']} trading days ahead</div>
            </div>
            <span class="consensus-pill" style="background:{consensus['color']};">{consensus['label']}</span>
        </div>
        """
    )

    columns = st.columns(len(MODEL_ORDER), gap="small")
    for column, name in zip(columns, MODEL_ORDER):
        with column:
            if name in models:
                render_html(model_card_html(models[name]))
            else:
                reason = next((e for e in entry["errors"] if e.startswith(name)), None)
                message = (
                    f"{name}: prediction failed for this ticker."
                    if reason
                    else f"{name} has not been trained for this horizon yet."
                )
                render_html(f'<div class="missing">{message}</div>')

    difference = (
        f"Difference between models: <strong>{consensus['difference_pp']:.2f} percentage points</strong>. "
        if consensus["difference_pp"] is not None
        else ""
    )
    render_html(f'<div class="verdict">{difference}{consensus["text"]}</div>')

    for error in entry["errors"]:
        st.caption(f"⚠️ {error}")
    st.write("")


def comparison_chart_frame(results: dict) -> pd.DataFrame:
    """Flatten the nested results into one tidy row per (ticker, model)."""
    rows = []
    for ticker, entry in results.items():
        for name, result in entry["models"].items():
            rows.append(
                {
                    "Ticker": ticker,
                    "Model": name,
                    "Expected": round(float(result["expected_return_pct"]), 3),
                    "Lower": round(float(result["lower_bound_pct"]), 3),
                    "Upper": round(float(result["upper_bound_pct"]), 3),
                    "Signal": result["signal"],
                    "Confidence": result["confidence_label"],
                    # Constant column so the zero reference line can be drawn from
                    # the same data as every other layer. Faceting a layered chart
                    # requires all layers to share one data source.
                    "Zero": 0.0,
                }
            )
    return pd.DataFrame(rows)


def comparison_chart(frame: pd.DataFrame) -> alt.Chart:
    """Grouped interval chart: one facet per ticker, one row per model."""
    base = alt.Chart(frame).encode(
        y=alt.Y("Model:N", title=None, axis=alt.Axis(labelFontWeight="bold", labelFontSize=11)),
        color=alt.Color(
            "Signal:N",
            scale=alt.Scale(domain=["BUY", "HOLD", "SELL"], range=[POSITIVE, NEUTRAL, NEGATIVE]),
            legend=alt.Legend(title="Signal", orient="top"),
        ),
        tooltip=["Ticker", "Model", "Expected", "Lower", "Upper", "Signal", "Confidence"],
    )
    interval = base.mark_rule(strokeWidth=6, opacity=0.30).encode(
        x=alt.X("Lower:Q", title="Expected movement (%)", scale=alt.Scale(zero=True)),
        x2="Upper:Q",
    )
    point = base.mark_point(size=150, filled=True, opacity=1).encode(x="Expected:Q")
    zero = alt.Chart(frame).mark_rule(
        color="#0f172a", strokeDash=[4, 4], opacity=0.55
    ).encode(x="Zero:Q")

    return (
        (interval + zero + point)
        .properties(height=alt.Step(28))
        .facet(
            row=alt.Row(
                "Ticker:N",
                title=None,
                header=alt.Header(labelFontWeight="bold", labelAngle=0, labelAlign="left"),
            )
        )
        .resolve_scale(x="shared")
    )


def render_comparison_chart(results: dict) -> None:
    """Draw the comparison chart, degrading to a table if the spec cannot build."""
    frame = comparison_chart_frame(results)
    if frame.empty:
        return
    try:
        st.altair_chart(comparison_chart(frame), use_container_width=True)
    except Exception:
        # The chart is a convenience, never the only route to the numbers.
        st.dataframe(
            frame.drop(columns=["Zero"]), use_container_width=True, hide_index=True
        )


def render_model_quality(horizon: int, availability: dict) -> None:
    trained = [name for name in MODEL_ORDER if availability.get(name)]
    if not trained:
        st.info(f"No model has been trained for {format_horizon(horizon)} yet.")
        return

    tabs = st.tabs(trained)
    for tab, name in zip(tabs, trained):
        with tab:
            metrics = load_model_quality(horizon, name)
            if not metrics:
                st.info("No saved evaluation report was found for this model.")
                continue

            test = metrics.get("component_metrics", {}).get("test") or metrics.get("test_metrics", {})
            intervals = metrics.get("uncertainty", {}).get("test_interval_metrics", {})
            backtest = metrics.get("backtest_metrics", {})

            row = st.columns(4)
            row[0].metric(
                "Cross-sectional IC", f"{test.get('cross_sectional_mean_ic', 0.0):+.4f}",
                help="Mean per-date rank correlation between forecast and outcome, on the untouched test period.",
            )
            row[1].metric(
                "Directional accuracy", f"{metrics.get('test_metrics', {}).get('direction_accuracy', 0.0):.1%}",
                help="Share of test forecasts whose sign matched the realised return.",
            )
            row[2].metric(
                "Interval coverage", f"{intervals.get('coverage_picp', 0.0):.1%}",
                delta=f"{intervals.get('coverage_error', 0.0):+.1%} vs nominal",
                delta_color="off",
                help="How often the realised return fell inside the displayed range. Should sit near the nominal level.",
            )
            row[3].metric(
                "Backtest Sharpe", f"{backtest.get('sharpe_ratio', 0.0):.2f}",
                help="Net of transaction costs and slippage, on non-overlapping horizon dates.",
            )

            second = st.columns(4)
            second[0].metric("Test MAE", f"{test.get('mae', 0.0) * 100:.2f}%")
            second[1].metric("Test RMSE", f"{test.get('rmse', 0.0) * 100:.2f}%")
            second[2].metric("IC t-statistic", f"{test.get('cross_sectional_ic_t_statistic', 0.0):+.2f}")
            second[3].metric("Mean interval width", f"{intervals.get('mean_interval_width_mpiw', 0.0) * 100:.1f}%")

            baselines = metrics.get("regression_baselines_excess") or {}
            if baselines:
                with st.expander("Comparison against baselines"):
                    table = [
                        {
                            "Model": "This model",
                            "Cross-sectional IC": round(test.get("cross_sectional_mean_ic", 0.0), 4),
                            "MAE": round(test.get("mae", 0.0), 4),
                            "RMSE": round(test.get("rmse", 0.0), 4),
                        }
                    ] + [
                        {
                            "Model": key.replace("_", " "),
                            "Cross-sectional IC": round(value.get("cross_sectional_mean_ic", 0.0), 4),
                            "MAE": round(value.get("mae", 0.0), 4),
                            "RMSE": round(value.get("rmse", 0.0), 4),
                        }
                        for key, value in baselines.items()
                    ]
                    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

            folds = (metrics.get("walk_forward") or {}).get("folds")
            if folds:
                with st.expander("Purged walk-forward folds"):
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Fold": fold["label"],
                                    "Test start": fold["test"]["start"],
                                    "Test end": fold["test"]["end"],
                                    "Cross-sectional IC": round(
                                        fold["test_metrics"].get("cross_sectional_mean_ic", 0.0), 4
                                    ),
                                }
                                for fold in folds
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Algo Trading Stock Predictor", page_icon="📊", layout="wide")
    inject_css()
    config = load_config()

    with st.sidebar:
        st.markdown("### Prediction settings")
        default_horizon = int(config.get("default_prediction_horizon", 21))
        default_index = (
            HORIZON_OPTIONS.index(default_horizon) if default_horizon in HORIZON_OPTIONS else 0
        )
        # A selectbox, not a slider: a slider with a single option raises
        # "RangeError: min (0) is equal/bigger than max (0)".
        selected_horizon = st.selectbox(
            "Prediction horizon",
            options=HORIZON_OPTIONS,
            index=default_index,
            format_func=format_horizon,
        )

        availability = available_models_for_horizon(config, int(selected_horizon))
        trained = [name for name in MODEL_ORDER if availability[name]]
        missing = [name for name in MODEL_ORDER if not availability[name]]

        st.markdown("**Models available for this horizon**")
        for name in MODEL_ORDER:
            st.markdown(f"{'✅' if availability[name] else '⬜'} {name}")
        if missing:
            st.caption("Not trained yet: " + ", ".join(missing))

        st.divider()
        st.markdown("**How to read the output**")
        st.markdown(
            "- **Expected movement** is the point forecast.\n"
            "- **Estimated range** is a calibrated interval, not a best/worst case.\n"
            "- **Confidence** compares the expected move with the width of that range.\n"
            "- Agreement between the two models is itself a signal."
        )
        st.divider()
        st.caption("Academic research and simulation only. This is not financial advice.")

    render_html(
        """
        <div class="hero">
            <div class="hero-title">Algo Trading Stock Predictor</div>
            <div class="hero-sub">
                Regression forecasts of future percentage return, each with a calibrated uncertainty
                range. Every ticker is scored by two independent models so you can see where they
                agree and where they do not.
            </div>
        </div>
        """
    )

    left, right = st.columns([2.2, 1])
    with left:
        raw_tickers = st.text_area(
            "Enter stock tickers", value="AAPL, MSFT, NVDA", height=110,
            help="Separate tickers with commas, spaces, or new lines.",
        )
    with right:
        st.markdown("### Examples")
        st.code("AAPL, MSFT, NVDA, TSLA")
        st.code("GOOGL\nAMZN\nMETA")

    run_clicked = st.button("Run prediction", use_container_width=True)

    if not trained:
        st.error(
            f"No model has been trained for {format_horizon(int(selected_horizon))} yet. "
            "Choose another horizon, or train this one first."
        )
        st.code(
            f"python3 train_all_models.py --horizon {selected_horizon} --compare", language="bash"
        )
    elif missing:
        st.info(
            f"Showing **{', '.join(trained)}** for {format_horizon(int(selected_horizon))}. "
            f"{', '.join(missing)} has not been trained for this horizon, so the side-by-side "
            "comparison is unavailable."
        )

    tab_results, tab_quality, tab_about = st.tabs(
        ["Prediction results", "Model quality", "About"]
    )

    with tab_quality:
        render_html(
            f'<div class="section-title">Out-of-sample evidence · {format_horizon(int(selected_horizon))}</div>'
        )
        st.caption(
            "Measured on a purged, chronologically held-out test period never used for training, "
            "early stopping, calibration or threshold selection."
        )
        render_model_quality(int(selected_horizon), availability)

    with tab_about:
        st.markdown("### What the dashboard shows")
        st.markdown(
            """
            - **Expected movement** — the point forecast of percentage return over the selected
              horizon. It is the sum of a market baseline and the model's stock-specific view.
            - **Estimated range** — a prediction interval calibrated so that, out of sample, the
              realised return falls inside it about as often as the stated confidence level.
            - **Confidence** — compares the size of the expected move with the width of its
              interval, and reports the probability that the direction is right.
            - **Signal** — BUY, HOLD or SELL, derived from the forecast and its uncertainty using a
              hurdle never smaller than round-trip trading costs.
            - **Agreement** — whether the two independent models point the same way.
            """
        )
        st.markdown("### Why the ranges are wide")
        st.markdown(
            """
            Over one month a typical large-cap stock has a return standard deviation near 8-10%.
            No model removes that noise, so an honest interval has to reflect it. A narrow band
            would simply be a miscalibrated one, which is why interval coverage is reported as a
            headline metric next to accuracy.
            """
        )

    with tab_results:
        if not run_clicked:
            st.info("Enter one or more tickers, choose a prediction horizon, and click Run prediction.")
            return
        if not trained:
            return

        tickers = parse_tickers(raw_tickers)
        if not tickers:
            st.error("Please enter at least one stock ticker.")
            return
        if len(tickers) > 12:
            st.warning(f"Showing the first 12 of {len(tickers)} tickers to keep the page responsive.")
            tickers = tickers[:12]

        results, load_errors = run_predictions(tickers, int(selected_horizon), config)
        for error in load_errors:
            st.error(error)
        if not results or all(not entry["models"] for entry in results.values()):
            st.error(
                "No predictions were generated. Check that the ticker symbols are valid and that "
                "market data is reachable."
            )
            for ticker, entry in results.items():
                for error in entry["errors"]:
                    st.caption(f"{ticker} — {error}")
            return

        summary_rows = []
        for ticker, entry in results.items():
            consensus = build_consensus(entry)
            for name in MODEL_ORDER:
                result = entry["models"].get(name)
                if not result:
                    continue
                summary_rows.append(
                    {
                        "Ticker": ticker,
                        "Model": name,
                        "Signal": result["signal"],
                        "Expected (%)": round(result["expected_return_pct"], 2),
                        "Range Low (%)": round(result["lower_bound_pct"], 2),
                        "Range High (%)": round(result["upper_bound_pct"], 2),
                        "Confidence": result["confidence_label"],
                        "P(direction)": f"{result['direction_probability']:.0%}",
                        "Agreement": consensus["label"],
                    }
                )

        render_html('<div class="section-title">Prediction summary</div>')
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        st.download_button(
            "Download predictions as CSV",
            data=pd.DataFrame(summary_rows).to_csv(index=False).encode("utf-8"),
            file_name=f"stock_predictions_h{selected_horizon}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if len(summary_rows) > 1:
            render_html('<div class="section-title">Expected movement and uncertainty</div>')
            render_comparison_chart(results)

        render_html('<div class="section-title">Model comparison by ticker</div>')
        for ticker, entry in results.items():
            render_ticker_block(ticker, entry)

        disagreements = [t for t, e in results.items() if build_consensus(e)["state"] == "conflict"]
        if disagreements:
            st.warning(
                "The two models disagree on direction for " + ", ".join(disagreements)
                + ". Treat these as inconclusive."
            )
        low_confidence = [
            t for t, e in results.items()
            if e["models"] and all(r["confidence_label"] == "Low" for r in e["models"].values())
        ]
        if low_confidence:
            st.info(
                "Low confidence from every model for " + ", ".join(low_confidence)
                + ". For these tickers the uncertainty band is wider than the expected move, so the "
                "forecast should be read as inconclusive rather than as a directional call."
            )


if __name__ == "__main__":
    main()
