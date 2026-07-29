"""
Streamlit dashboard for the future-return regression models.

The interface is built around one rule: a point forecast is never shown on its
own. Every expected movement appears together with its calibrated range and an
explicit statement of how confident the model is, so a 0.4% forecast with a
±14% band cannot be mistaken for a reliable call.
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
    get_available_horizons,
    load_config,
    load_model_and_metadata,
    predict_ticker_with_artifacts,
)


HORIZON_LABELS = {
    1: "1 day",
    5: "1 week",
    10: "2 weeks",
    21: "1 month",
    42: "2 months",
    63: "1 quarter",
    126: "6 months",
    252: "1 year",
}

SIGNAL_STYLE = {
    "BUY": {
        "emoji": "📈",
        "css": "buy-card",
        "pill": "buy-pill",
        "explain": "The risk-adjusted edge clears the cost hurdle on the upside.",
    },
    "HOLD": {
        "emoji": "⏸️",
        "css": "hold-card",
        "pill": "hold-pill",
        "explain": "The expected move is too small relative to its uncertainty and trading costs to act on.",
    },
    "SELL": {
        "emoji": "📉",
        "css": "sell-card",
        "pill": "sell-pill",
        "explain": "The risk-adjusted edge clears the cost hurdle on the downside.",
    },
}

CONFIDENCE_STYLE = {
    "High": {"css": "confidence-high", "icon": "●●●"},
    "Moderate": {"css": "confidence-moderate", "icon": "●●○"},
    "Low": {"css": "confidence-low", "icon": "●○○"},
}


def format_horizon(horizon: int) -> str:
    label = HORIZON_LABELS.get(int(horizon))
    return f"{label} ({horizon} trading days)" if label else f"{horizon} trading days"


def parse_tickers(raw_text: str) -> list[str]:
    cleaned = raw_text.replace("\n", ",").replace(";", ",").replace(" ", ",")
    tickers: list[str] = []
    for item in cleaned.split(","):
        ticker = item.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


@st.cache_resource(show_spinner=False)
def cached_model_artifacts(config_key: str, horizon: int):
    return load_model_and_metadata(load_config(), horizon=horizon)


@st.cache_data(show_spinner=False)
def load_model_quality(horizon: int) -> dict | None:
    """Read the saved out-of-sample evaluation for the selected horizon."""
    config = load_config()
    path = artifact_path(config["metrics_output_path"], horizon)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
        section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #111827 52%, #020617 100%); }
        section[data-testid="stSidebar"] * { color: #e5e7eb !important; }

        .hero { border-radius: 28px; padding: 1.9rem 2rem; background: radial-gradient(circle at top left, #dbeafe 0%, #eff6ff 28%, #f8fafc 70%); border: 1px solid #dbeafe; box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08); margin-bottom: 1.3rem; }
        .hero-title { font-size: 2.5rem; font-weight: 900; letter-spacing: -0.05em; color: #0f172a; margin-bottom: 0.5rem; }
        .hero-subtitle { color: #475569; font-size: 1.02rem; line-height: 1.6; font-weight: 600; max-width: 950px; }
        .section-title { font-size: 1.45rem; font-weight: 900; letter-spacing: -0.03em; color: #0f172a; margin: 1.5rem 0 0.7rem 0; }

        .signal-card { border-radius: 24px; padding: 1.4rem 1.5rem; margin-bottom: 1.1rem; border: 1px solid rgba(148, 163, 184, 0.28); box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08); background: #ffffff; }
        .buy-card { border-left: 8px solid #16a34a; }
        .hold-card { border-left: 8px solid #ca8a04; }
        .sell-card { border-left: 8px solid #dc2626; }
        .ticker-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; }
        .ticker-name { font-size: 1.8rem; font-weight: 900; color: #0f172a; letter-spacing: -0.04em; line-height: 1.1; }
        .latest-date { color: #64748b; font-weight: 700; font-size: 0.82rem; }
        .signal-pill { display: inline-flex; align-items: center; gap: 0.35rem; border-radius: 999px; padding: 0.5rem 0.95rem; font-weight: 900; color: white; font-size: 0.95rem; box-shadow: 0 8px 18px rgba(15,23,42,0.14); white-space: nowrap; }
        .buy-pill { background: #16a34a; }
        .hold-pill { background: #ca8a04; }
        .sell-pill { background: #dc2626; }

        .headline-grid { display: grid; grid-template-columns: minmax(200px, 1fr) 2fr; gap: 1.6rem; align-items: center; margin-top: 1.1rem; }
        .headline-label { font-size: 0.7rem; font-weight: 900; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.15rem; }
        .headline-value { font-size: 2.6rem; font-weight: 900; letter-spacing: -0.055em; line-height: 1; }
        .headline-range { font-size: 1.02rem; font-weight: 800; color: #334155; margin-top: 0.45rem; }
        .positive { color: #16a34a; }
        .negative { color: #dc2626; }
        .neutral { color: #ca8a04; }

        .range-track { position: relative; height: 46px; margin-top: 0.3rem; }
        .range-base { position: absolute; top: 19px; left: 0; right: 0; height: 8px; border-radius: 6px; background: #e2e8f0; }
        .range-span { position: absolute; top: 19px; height: 8px; border-radius: 6px; background: linear-gradient(90deg, #bfdbfe, #60a5fa, #bfdbfe); }
        .range-zero { position: absolute; top: 10px; width: 2px; height: 26px; background: #0f172a; opacity: 0.55; }
        .range-point { position: absolute; top: 12px; width: 4px; height: 22px; border-radius: 2px; }
        .range-caption { display: flex; justify-content: space-between; font-size: 0.72rem; font-weight: 800; color: #64748b; margin-top: 0.15rem; }

        .confidence-chip { display: inline-flex; align-items: center; gap: 0.45rem; border-radius: 999px; padding: 0.32rem 0.8rem; font-weight: 900; font-size: 0.78rem; letter-spacing: 0.02em; margin-top: 0.55rem; }
        .confidence-high { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
        .confidence-moderate { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
        .confidence-low { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }

        .mini-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.7rem; margin-top: 1.15rem; }
        .mini-box { background: #f8fafc; border-radius: 14px; padding: 0.75rem 0.85rem; border: 1px solid #e2e8f0; }
        .mini-label { font-size: 0.65rem; font-weight: 900; color: #64748b; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.28rem; }
        .mini-value { font-size: 0.98rem; font-weight: 900; color: #0f172a; }
        .card-note { margin-top: 0.95rem; color: #475569; font-weight: 650; line-height: 1.5; font-size: 0.87rem; border-top: 1px solid #e2e8f0; padding-top: 0.8rem; }

        .info-box { border-radius: 16px; padding: 0.9rem 1.05rem; background: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a; font-weight: 700; margin-bottom: 1rem; font-size: 0.9rem; }
        div[data-testid="stTextArea"] textarea { border-radius: 16px; border: 1px solid #cbd5e1; background: #f8fafc; font-weight: 700; font-size: 1rem; }
        div.stButton > button:first-child { border-radius: 14px; background: linear-gradient(90deg, #2563eb, #1d4ed8); color: white; border: none; font-weight: 900; height: 3rem; box-shadow: 0 12px 26px rgba(37, 99, 235, 0.25); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def return_class(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def render_range_bar(lower_pct: float, expected_pct: float, upper_pct: float) -> str:
    """A zero-anchored bar showing the interval and where the point forecast sits."""
    scale = max(abs(lower_pct), abs(upper_pct), 1e-6) * 1.15

    def position(value: float) -> float:
        return (value + scale) / (2.0 * scale) * 100.0

    left = position(lower_pct)
    right = position(upper_pct)
    point = position(expected_pct)
    point_color = "#16a34a" if expected_pct > 0 else "#dc2626" if expected_pct < 0 else "#ca8a04"

    return f"""
        <div class="range-track">
            <div class="range-base"></div>
            <div class="range-span" style="left:{left:.2f}%;width:{max(right - left, 0.6):.2f}%;"></div>
            <div class="range-zero" style="left:50%;"></div>
            <div class="range-point" style="left:{point:.2f}%;background:{point_color};"></div>
        </div>
        <div class="range-caption">
            <span>{lower_pct:+.1f}%</span><span>0%</span><span>{upper_pct:+.1f}%</span>
        </div>
    """


def render_signal_card(result: dict) -> None:
    style = SIGNAL_STYLE[result["signal"]]
    confidence_style = CONFIDENCE_STYLE.get(result["confidence_label"], CONFIDENCE_STYLE["Low"])
    expected = float(result["expected_return_pct"])
    lower = float(result["lower_bound_pct"])
    upper = float(result["upper_bound_pct"])
    level = float(result["confidence_level"])

    st.markdown(
        f"""
        <div class="signal-card {style['css']}">
            <div class="ticker-row">
                <div>
                    <div class="ticker-name">{result['ticker']}</div>
                    <div class="latest-date">Latest market date: {result['latest_data_date']}
                        &nbsp;·&nbsp; {result['prediction_horizon_trading_days']} trading days ahead</div>
                </div>
                <div class="signal-pill {style['pill']}">{style['emoji']} {result['signal']}</div>
            </div>

            <div class="headline-grid">
                <div>
                    <div class="headline-label">Expected movement</div>
                    <div class="headline-value {return_class(expected)}">{expected:+.2f}%</div>
                    <div class="headline-range">Estimated range: {lower:+.2f}% to {upper:+.2f}%</div>
                    <div class="confidence-chip {confidence_style['css']}">
                        {confidence_style['icon']} &nbsp;{result['confidence_label']} confidence
                        &nbsp;·&nbsp; {result['direction_probability']:.0%} chance the direction is right
                    </div>
                </div>
                <div>
                    <div class="headline-label">{level:.0%} prediction interval</div>
                    {render_range_bar(lower, expected, upper)}
                </div>
            </div>

            <div class="mini-grid">
                <div class="mini-box">
                    <div class="mini-label">Market baseline</div>
                    <div class="mini-value">{result['market_drift_pct']:+.2f}%</div>
                </div>
                <div class="mini-box">
                    <div class="mini-label">Model view vs market</div>
                    <div class="mini-value {return_class(result['model_excess_return_pct'])}">
                        {result['model_excess_return_pct']:+.2f}%</div>
                </div>
                <div class="mini-box">
                    <div class="mini-label">Forecast uncertainty (σ)</div>
                    <div class="mini-value">±{result['forecast_sigma_pct']:.2f}%</div>
                </div>
                <div class="mini-box">
                    <div class="mini-label">Cost hurdle</div>
                    <div class="mini-value">{result['signal_threshold_pct']:.2f}%</div>
                </div>
            </div>

            <div class="card-note">
                <strong>{result['signal']}</strong> — {style['explain']}
                {result['confidence_explanation']}
                The expected movement splits into a {result['market_drift_pct']:+.2f}% market baseline
                and a {result['model_excess_return_pct']:+.2f}% stock-specific view from the model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_range_chart(results: list[dict]) -> None:
    """Error-bar comparison so several tickers can be ranked at a glance."""
    frame = pd.DataFrame(
        [
            {
                "Ticker": result["ticker"],
                "Expected": result["expected_return_pct"],
                "Lower": result["lower_bound_pct"],
                "Upper": result["upper_bound_pct"],
                "Confidence": result["confidence_label"],
                "Signal": result["signal"],
            }
            for result in results
        ]
    )

    base = alt.Chart(frame).encode(
        y=alt.Y("Ticker:N", sort="-x", title=None, axis=alt.Axis(labelFontWeight="bold")),
    )
    interval = base.mark_rule(strokeWidth=7, opacity=0.35, color="#60a5fa").encode(
        x=alt.X("Lower:Q", title="Expected movement (%)", scale=alt.Scale(zero=True)),
        x2="Upper:Q",
        tooltip=["Ticker", "Expected", "Lower", "Upper", "Confidence", "Signal"],
    )
    point = base.mark_point(size=140, filled=True, opacity=1).encode(
        x="Expected:Q",
        color=alt.Color(
            "Signal:N",
            scale=alt.Scale(
                domain=["BUY", "HOLD", "SELL"], range=["#16a34a", "#ca8a04", "#dc2626"]
            ),
            legend=alt.Legend(title="Signal", orient="top"),
        ),
        tooltip=["Ticker", "Expected", "Lower", "Upper", "Confidence", "Signal"],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        color="#0f172a", strokeDash=[4, 4], opacity=0.6
    ).encode(x="x:Q")

    st.altair_chart(
        (interval + zero + point).properties(height=max(140, 46 * len(frame))),
        use_container_width=True,
    )


def render_model_quality(horizon: int) -> None:
    """Show the saved out-of-sample evidence for the model behind these numbers."""
    metrics = load_model_quality(horizon)
    if not metrics:
        st.info("No saved evaluation report was found for this horizon yet.")
        return

    test = metrics.get("component_metrics", {}).get("test") or metrics.get("test_metrics", {})
    intervals = metrics.get("uncertainty", {}).get("test_interval_metrics", {})
    backtest = metrics.get("backtest_metrics", {})

    columns = st.columns(4)
    columns[0].metric(
        "Cross-sectional IC",
        f"{test.get('cross_sectional_mean_ic', 0.0):+.4f}",
        help="Mean per-date Spearman correlation between forecast and outcome on the untouched test period.",
    )
    columns[1].metric(
        "Directional accuracy",
        f"{test.get('direction_accuracy', 0.0):.1%}",
        help="Share of test forecasts whose sign matched the realised return.",
    )
    columns[2].metric(
        "Interval coverage",
        f"{intervals.get('coverage_picp', 0.0):.1%}",
        delta=f"{intervals.get('coverage_error', 0.0):+.1%} vs nominal",
        help="How often the realised return actually fell inside the displayed range, out of sample.",
    )
    columns[3].metric(
        "Backtest Sharpe",
        f"{backtest.get('sharpe_ratio', 0.0):.2f}",
        help="Net of transaction costs and slippage, on non-overlapping horizon dates.",
    )

    second = st.columns(4)
    second[0].metric("Test MAE", f"{test.get('mae', 0.0) * 100:.2f}%")
    second[1].metric("Test RMSE", f"{test.get('rmse', 0.0) * 100:.2f}%")
    second[2].metric(
        "IC t-statistic", f"{test.get('cross_sectional_ic_t_statistic', 0.0):+.2f}"
    )
    second[3].metric(
        "Mean interval width", f"{intervals.get('mean_interval_width_mpiw', 0.0) * 100:.1f}%"
    )

    with st.expander("Comparison against baselines and the walk-forward folds"):
        baselines = metrics.get("regression_baselines_excess") or metrics.get("regression_baselines", {})
        if baselines:
            rows = [
                {
                    "Model": "This model",
                    "Cross-sectional IC": round(test.get("cross_sectional_mean_ic", 0.0), 4),
                    "Direction accuracy": round(test.get("direction_accuracy", 0.0), 4),
                    "MAE": round(test.get("mae", 0.0), 4),
                    "RMSE": round(test.get("rmse", 0.0), 4),
                }
            ]
            for name, values in baselines.items():
                rows.append(
                    {
                        "Model": name.replace("_", " "),
                        "Cross-sectional IC": round(values.get("cross_sectional_mean_ic", 0.0), 4),
                        "Direction accuracy": round(values.get("direction_accuracy", 0.0), 4),
                        "MAE": round(values.get("mae", 0.0), 4),
                        "RMSE": round(values.get("rmse", 0.0), 4),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        walk_forward = metrics.get("walk_forward")
        if walk_forward and walk_forward.get("folds"):
            fold_rows = [
                {
                    "Fold": fold["label"],
                    "Test start": fold["test"]["start"],
                    "Test end": fold["test"]["end"],
                    "Cross-sectional IC": round(
                        fold["test_metrics"].get("cross_sectional_mean_ic", 0.0), 4
                    ),
                    "Direction accuracy": round(
                        fold["test_metrics"].get("direction_accuracy", 0.0), 4
                    ),
                }
                for fold in walk_forward["folds"]
            ]
            st.markdown("**Purged walk-forward folds**")
            st.dataframe(pd.DataFrame(fold_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Run `python3 train.py --walk-forward` to add per-fold stability results.")


def main() -> None:
    st.set_page_config(page_title="Algo Trading Stock Predictor", page_icon="📊", layout="wide")
    inject_css()

    config = load_config()
    available_horizons = get_available_horizons(config)
    default_horizon = int(config.get("default_prediction_horizon", available_horizons[0]))
    default_index = available_horizons.index(default_horizon) if default_horizon in available_horizons else 0

    with st.sidebar:
        st.markdown("### Prediction settings")
        selected_horizon = st.select_slider(
            "Prediction horizon",
            options=available_horizons,
            value=available_horizons[default_index],
            format_func=format_horizon,
        )
        st.metric("Input window", f"{config['window_size']} trading days")
        st.metric("Selected horizon", format_horizon(int(selected_horizon)))
        st.metric(
            "Confidence level",
            f"{float(config.get('uncertainty', {}).get('confidence_level', 0.80)):.0%}",
        )
        st.divider()
        st.markdown("**How to read the output**")
        st.markdown(
            "- **Expected movement** is the point forecast.\n"
            "- **Estimated range** is a calibrated interval, not a best/worst case.\n"
            "- **Confidence** compares the expected move with the size of that range.\n"
            "- A wide range around a small forecast means the model has no usable view."
        )
        st.divider()
        st.markdown("This dashboard is for academic research and simulation only. It is not financial advice.")

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Algo Trading Stock Predictor</div>
            <div class="hero-subtitle">
                Regression forecasts of future percentage return, with a calibrated uncertainty range
                on every prediction. Enter one or more tickers, choose how far ahead to look, and see
                both the expected movement and how much confidence the model has in it.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2.2, 1])
    with left:
        raw_tickers = st.text_area(
            "Enter stock tickers",
            value="AAPL, MSFT, NVDA",
            height=110,
            help="Separate tickers with commas, spaces, or new lines.",
        )
    with right:
        st.markdown("### Examples")
        st.code("AAPL, MSFT, NVDA, TSLA")
        st.code("GOOGL\nAMZN\nMETA")

    run_clicked = st.button("Run prediction", use_container_width=True)

    st.markdown(
        f"""
        <div class="info-box">
            The model predicts the {format_horizon(int(selected_horizon))} future return and its uncertainty.
            The BUY / HOLD / SELL signal is derived afterwards from the risk-adjusted edge: the expected
            move must beat trading costs <em>relative to</em> the width of its own uncertainty band.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_results, tab_quality, tab_about = st.tabs(
        ["Prediction results", "Model quality", "About"]
    )

    with tab_quality:
        st.markdown(
            f'<div class="section-title">Out-of-sample evidence · {format_horizon(int(selected_horizon))}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Measured on a purged, chronologically held-out test period that was never used for "
            "training, early stopping, calibration or threshold selection."
        )
        render_model_quality(int(selected_horizon))

    with tab_about:
        st.markdown("### What the dashboard shows")
        st.markdown(
            """
            - **Expected movement** — the model's point forecast of the percentage return over the
              selected horizon. It is the sum of a market baseline and the model's stock-specific view.
            - **Estimated range** — a prediction interval calibrated on validation data so that, out of
              sample, the realised return falls inside it about as often as the stated confidence level.
            - **Confidence** — compares the size of the expected move with the width of its interval,
              and reports the probability that the direction is right.
            - **Signal** — BUY, HOLD or SELL, derived from the forecast and its uncertainty using a
              hurdle that is never smaller than round-trip trading costs.
            """
        )
        st.markdown("### Why the range is wide")
        st.markdown(
            """
            Over a one-month horizon a typical large-cap stock has a return standard deviation of
            roughly 8-10%. No model removes that noise; an honest interval has to reflect it. A narrow
            band would simply be a miscalibrated one, which is why interval coverage is reported as a
            headline metric alongside accuracy.
            """
        )

    with tab_results:
        if not run_clicked:
            st.info("Enter one or more tickers, choose a prediction horizon, and click Run prediction.")
            return

        tickers = parse_tickers(raw_tickers)
        if not tickers:
            st.error("Please enter at least one stock ticker.")
            return

        try:
            models, metadata, scaler, feature_columns, device = cached_model_artifacts(
                "default", int(selected_horizon)
            )
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.code(f"python3 train.py --horizon {selected_horizon}")
            return
        except Exception as exc:
            st.error(f"Failed to load model artifacts: {exc}")
            return

        results: list[dict] = []
        errors: list[str] = []
        progress = st.progress(0.0, text="Running predictions...")

        for index, ticker in enumerate(tickers, start=1):
            try:
                results.append(
                    predict_ticker_with_artifacts(
                        ticker=ticker,
                        config=config,
                        model=models,
                        metadata=metadata,
                        scaler=scaler,
                        feature_columns=feature_columns,
                        device=device,
                        horizon=int(selected_horizon),
                    )
                )
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")
            progress.progress(index / len(tickers), text=f"Predicted {index}/{len(tickers)}")
        progress.empty()

        for error in errors:
            st.warning(error)
        if not results:
            st.error("No predictions were generated.")
            return

        summary_df = pd.DataFrame(
            [
                {
                    "Ticker": result["ticker"],
                    "Signal": result["signal"],
                    "Expected Movement (%)": round(result["expected_return_pct"], 2),
                    "Range Low (%)": round(result["lower_bound_pct"], 2),
                    "Range High (%)": round(result["upper_bound_pct"], 2),
                    "Confidence": result["confidence_label"],
                    "P(direction)": f"{result['direction_probability']:.0%}",
                    "σ (%)": round(result["forecast_sigma_pct"], 2),
                    "Latest Data Date": result["latest_data_date"],
                }
                for result in results
            ]
        )

        st.markdown('<div class="section-title">Prediction summary</div>', unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download predictions as CSV",
            data=pd.DataFrame(results).to_csv(index=False).encode("utf-8"),
            file_name=f"stock_predictions_h{selected_horizon}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if len(results) > 1:
            st.markdown(
                '<div class="section-title">Expected movement and uncertainty</div>',
                unsafe_allow_html=True,
            )
            render_range_chart(results)

        st.markdown('<div class="section-title">Detailed results</div>', unsafe_allow_html=True)
        for result in results:
            render_signal_card(result)

        low_confidence = [r["ticker"] for r in results if r["confidence_label"] == "Low"]
        if low_confidence:
            st.warning(
                "Low confidence for "
                + ", ".join(low_confidence)
                + ". For these tickers the uncertainty band is wider than the expected move, so the "
                "forecast should be treated as inconclusive rather than as a directional call."
            )


if __name__ == "__main__":
    main()
