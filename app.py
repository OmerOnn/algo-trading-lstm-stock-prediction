from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict import (  # noqa: E402
    get_available_horizons,
    load_config,
    load_model_and_metadata,
    predict_ticker_with_artifacts,
)


SIGNAL_STYLE = {
    "BUY": {
        "emoji": "📈",
        "css": "buy-card",
        "pill": "buy-pill",
        "tone": "Positive signal",
        "explain": "The model detected a stronger upside scenario for the selected horizon.",
    },
    "HOLD": {
        "emoji": "⏸️",
        "css": "hold-card",
        "pill": "hold-pill",
        "tone": "Neutral signal",
        "explain": "The model did not detect a strong enough directional signal.",
    },
    "SELL": {
        "emoji": "📉",
        "css": "sell-card",
        "pill": "sell-pill",
        "tone": "Negative signal",
        "explain": "The model detected a stronger downside scenario for the selected horizon.",
    },
}


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
    config = load_config()
    return load_model_and_metadata(config, horizon=horizon)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 52%, #020617 100%);
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb !important;
        }

        .hero {
            border-radius: 30px;
            padding: 2rem;
            background: radial-gradient(circle at top left, #dbeafe 0%, #eff6ff 28%, #f8fafc 70%);
            border: 1px solid #dbeafe;
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.4rem;
        }

        .hero-title {
            font-size: 2.7rem;
            font-weight: 950;
            letter-spacing: -0.055em;
            color: #0f172a;
            margin-bottom: 0.55rem;
        }

        .hero-subtitle {
            color: #475569;
            font-size: 1.05rem;
            line-height: 1.6;
            font-weight: 650;
            max-width: 950px;
        }

        .section-title {
            font-size: 1.55rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            color: #0f172a;
            margin: 1.3rem 0 0.8rem 0;
        }

        .signal-card {
            border-radius: 26px;
            padding: 1.35rem 1.45rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.09);
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,250,252,0.94));
        }

        .buy-card { border-left: 9px solid #16a34a; }
        .hold-card { border-left: 9px solid #ca8a04; }
        .sell-card { border-left: 9px solid #dc2626; }

        .ticker-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }

        .ticker-name {
            font-size: 2rem;
            font-weight: 950;
            color: #0f172a;
            letter-spacing: -0.04em;
        }

        .signal-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.55rem 1rem;
            font-weight: 950;
            color: white;
            box-shadow: 0 10px 22px rgba(15,23,42,0.16);
        }

        .buy-pill { background: #16a34a; }
        .hold-pill { background: #ca8a04; }
        .sell-pill { background: #dc2626; }

        .latest-date {
            color: #64748b;
            font-weight: 800;
            margin-top: -0.2rem;
            margin-bottom: 1rem;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .mini-box {
            background: #f1f5f9;
            border-radius: 18px;
            padding: 0.95rem 1rem;
            border: 1px solid #e2e8f0;
        }

        .mini-label {
            font-size: 0.72rem;
            font-weight: 900;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }

        .mini-value {
            font-size: 1rem;
            font-weight: 950;
            color: #0f172a;
        }

        .positive { color: #16a34a; }
        .negative { color: #dc2626; }
        .neutral { color: #ca8a04; }

        .info-box {
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
            font-weight: 750;
            margin-bottom: 1rem;
        }

        .warning-box {
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #9a3412;
            font-weight: 750;
            margin-bottom: 1rem;
        }

        div[data-testid="stTextArea"] textarea {
            border-radius: 18px;
            border: 1px solid #cbd5e1;
            background: #f8fafc;
            font-weight: 700;
            font-size: 1rem;
        }

        div.stButton > button:first-child {
            border-radius: 16px;
            background: linear-gradient(90deg, #ef4444, #f97316);
            color: white;
            border: none;
            font-weight: 950;
            height: 3.05rem;
            box-shadow: 0 14px 30px rgba(239, 68, 68, 0.22);
        }
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


def probability_chart(result: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Signal": ["BUY", "HOLD", "SELL"],
            "Probability": [result["prob_buy"], result["prob_hold"], result["prob_sell"]],
        }
    )


def render_signal_card(result: dict) -> None:
    signal = result["signal"]
    style = SIGNAL_STYLE[signal]
    expected = float(result["expected_return_pct"])
    return_css = return_class(expected)

    st.markdown(
        f"""
        <div class="signal-card {style['css']}">
            <div class="ticker-row">
                <div>
                    <div class="ticker-name">{result['ticker']}</div>
                    <div class="latest-date">Latest market date: {result['latest_data_date']}</div>
                </div>
                <div class="signal-pill {style['pill']}">{style['emoji']} {signal}</div>
            </div>
            <div class="mini-grid">
                <div class="mini-box">
                    <div class="mini-label">Expected Movement</div>
                    <div class="mini-value {return_css}">{expected:.2f}%</div>
                </div>
                <div class="mini-box">
                    <div class="mini-label">Confidence</div>
                    <div class="mini-value">{result['confidence'] * 100:.2f}%</div>
                </div>
                <div class="mini-box">
                    <div class="mini-label">Prediction Horizon</div>
                    <div class="mini-value">{result['prediction_horizon_trading_days']} trading days</div>
                </div>
            </div>
            <p style="margin-top:1rem;color:#475569;font-weight:700;line-height:1.55;">{style['explain']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Algo Trading Stock Predictor",
        page_icon="📊",
        layout="wide",
    )
    inject_css()

    config = load_config()
    available_horizons = get_available_horizons(config)
    default_horizon = int(config.get("default_prediction_horizon", available_horizons[0]))
    default_index = available_horizons.index(default_horizon) if default_horizon in available_horizons else 0

    with st.sidebar:
        st.markdown("### Prediction Settings")
        st.markdown("Choose the prediction horizon and enter one or more stock tickers.")
        selected_horizon = st.select_slider(
            "Prediction horizon",
            options=available_horizons,
            value=available_horizons[default_index],
            format_func=lambda x: f"{x} trading days",
        )
        st.metric("Input Window", f"{config['window_size']} days")
        st.metric("Selected Horizon", f"{selected_horizon} days")
        st.divider()
        st.markdown("**Important**")
        st.markdown("This dashboard is for academic research and simulation only. It is not financial advice.")

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Algo Trading Stock Predictor</div>
            <div class="hero-subtitle">
                Enter one or more stock tickers, choose how many trading days ahead to predict,
                and view the model's simulated BUY / HOLD / SELL signal with confidence and expected movement.
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
            help="You can enter tickers separated by commas, spaces, or new lines.",
        )
    with right:
        st.markdown("### Examples")
        st.code("AAPL, MSFT, NVDA, TSLA")
        st.code("GOOGL\nAMZN\nMETA")

    run_clicked = st.button("Run Prediction", use_container_width=True)

    st.markdown(
        f"""
        <div class="info-box">
            The model will predict {selected_horizon} trading days ahead using the latest available market data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_results, tab_about = st.tabs(["Prediction Results", "About"])

    with tab_about:
        st.markdown("### What the dashboard shows")
        st.markdown(
            """
            - **Signal**: the predicted action class: BUY, HOLD, or SELL.
            - **Expected Movement**: the model's estimated percentage movement for the selected horizon.
            - **Confidence**: the highest probability assigned by the classifier.
            - **Class Probabilities**: the model's probability distribution across BUY, HOLD, and SELL.

            The displayed expected movement is aligned with the selected signal so the UI stays consistent. For example,
            a SELL signal is displayed as a negative expected movement.
            """
        )

    with tab_results:
        if not run_clicked:
            st.info("Enter one or more tickers, choose a prediction horizon, and click Run Prediction.")
            return

        tickers = parse_tickers(raw_tickers)
        if not tickers:
            st.error("Please enter at least one stock ticker.")
            return

        try:
            model, metadata, scaler, feature_columns, device = cached_model_artifacts("default", int(selected_horizon))
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.code(f"python3 train.py --horizon {selected_horizon}")
            return
        except Exception as exc:
            st.error(f"Failed to load model artifacts: {exc}")
            return

        results: list[dict] = []
        errors: list[str] = []

        with st.spinner("Running predictions..."):
            for ticker in tickers:
                try:
                    result = predict_ticker_with_artifacts(
                        ticker=ticker,
                        config=config,
                        model=model,
                        metadata=metadata,
                        scaler=scaler,
                        feature_columns=feature_columns,
                        device=device,
                        horizon=int(selected_horizon),
                    )
                    results.append(result)
                except Exception as exc:
                    errors.append(f"{ticker}: {exc}")

        for error in errors:
            st.warning(error)

        if not results:
            st.error("No predictions were generated.")
            return

        summary_rows = []
        for result in results:
            summary_rows.append(
                {
                    "Ticker": result["ticker"],
                    "Signal": result["signal"],
                    "Expected Movement (%)": round(result["expected_return_pct"], 2),
                    "Confidence": f"{result['confidence'] * 100:.2f}%",
                    "BUY Probability": f"{result['prob_buy'] * 100:.2f}%",
                    "HOLD Probability": f"{result['prob_hold'] * 100:.2f}%",
                    "SELL Probability": f"{result['prob_sell'] * 100:.2f}%",
                    "Latest Data Date": result["latest_data_date"],
                    "Horizon": result["prediction_horizon_trading_days"],
                }
            )

        st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download predictions as CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"stock_predictions_h{selected_horizon}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown('<div class="section-title">Detailed Results</div>', unsafe_allow_html=True)
        for result in results:
            render_signal_card(result)
            st.bar_chart(probability_chart(result), x="Signal", y="Probability", use_container_width=True)


if __name__ == "__main__":
    main()
