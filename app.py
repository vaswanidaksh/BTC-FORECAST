import streamlit as st
import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from arch import arch_model
import plotly.graph_objects as go
from datetime import datetime, timezone
import json
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Forecast Dashboard",
    page_icon="₿",
    layout="wide"
)

HISTORY_FILE = "prediction_history.jsonl"

# ── Helper: Fetch BTC data ────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_btc_data(limit=500):
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    df = pd.DataFrame(r.json(), columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df.set_index("open_time", inplace=True)
    return df["close"]

# ── Helper: Rolling entropy ───────────────────────────────────────────────────
def rolling_entropy(x, window=60, bins=20):
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return x.rolling(window).apply(ent, raw=True)

# ── Helper: Run GBM model ─────────────────────────────────────────────────────
def run_model(prices):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    am = arch_model(log_ret * 100, vol='FIGARCH', p=1, o=0, q=1, dist='studentst')
    res = am.fit(disp='off')
    sigma_fig = res.conditional_volatility / 100
    resid = (log_ret * 100 - res.params['mu']) / res.conditional_volatility
    nu = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])
    H_series = rolling_entropy(resid)
    M_series = log_ret.abs().rolling(60).mean()
    bar_sigma2 = (sigma_fig**2).mean()
    redundancy = 1 + 0.1 * np.log1p(
        prices.rolling(5).var() / prices.rolling(20).var()
    )
    info_filter = (H_series > H_series.mean()).astype(float)
    H_max = H_series.max() if H_series.max() > 0 else 1.0
    M_max = M_series.max() if M_series.max() > 0 else 1.0
    α0, δ0 = 0.5, 0.3
    if α0 * H_max + δ0 * M_max >= 1:
        fac = 0.95 / (α0 * H_max + δ0 * M_max)
        α0 *= fac
        δ0 *= fac
    base_params = {
        'alpha': α0, 'delta': δ0,
        'gamma': 0.2, 'kappa': 0.1, 'eta': 1e-3
    }

    def update_params(p, sigma2, bar_sigma2, t):
        err = sigma2 - bar_sigma2
        lr = p['eta'] / (1 + t**0.55)
        p['gamma'] = np.clip(p['gamma'] + lr * err, 0.01, 0.5)
        return p

    def simulate_once(S0, mu, sigma_fig, H, M, params, bar_sigma2):
        S = S0
        sigma2 = sigma_fig.iloc[-1] ** 2
        H_val = min(H.iloc[-1] / H_max, 1.0)
        M_val = min(M.iloc[-1] / M_max, 1.0)
        crisis = (H_val > 0.8) or (M_val > 0.8)
        delta_t = params['delta'] if crisis else 0.0
        sigma2 = (
            sigma_fig.iloc[-1]**2 * (1 + params['alpha'] * H_val + delta_t * M_val)
            + params['gamma'] * (bar_sigma2 - sigma2)
        )
        sigma2 *= max(1e-12, redundancy.iloc[-1])
        sigma2 *= 1 + 0.5 * info_filter.iloc[-1]
        sigma2 = max(1e-6, min(sigma2, 0.5))
        Z = np.random.standard_t(nu) * np.sqrt((nu - 2) / nu)
        return S * np.exp((mu - 0.5 * sigma2) + np.sqrt(sigma2) * Z)

    S0 = prices.iloc[-1]
    mu = log_ret.mean()
    simulated = np.array([
        simulate_once(S0, mu, sigma_fig, H_series, M_series,
                      base_params.copy(), bar_sigma2)
        for _ in range(10_000)
    ])
    low95, high95 = np.percentile(simulated, [5.0, 95.0])
    return low95, high95, S0

# ── Part C: Save prediction to history ───────────────────────────────────────
def save_prediction(current_price, low95, high95, target_hour):
    """Save prediction. target_hour is the candle we're predicting."""
    record = {
        "predicted_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "target_hour": target_hour,
        "current_price": round(current_price, 2),
        "low_95": round(low95, 2),
        "high_95": round(high95, 2),
        "actual_price": None,  # filled in later
        "hit": None            # filled in later
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

# ── Part C: Load and update history with actuals ──────────────────────────────
def load_and_update_history(prices):
    """
    Load all saved predictions.
    For any prediction whose target_hour has now passed,
    fill in the actual price from Binance data.
    """
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        return []

    # Build a lookup: hour timestamp → actual close price
    price_lookup = {
        str(ts): price
        for ts, price in prices.items()
    }

    updated = False
    for rec in records:
        if rec["actual_price"] is None:
            # Check if target hour has closed and we have the actual price
            actual = price_lookup.get(rec["target_hour"])
            if actual:
                rec["actual_price"] = round(actual, 2)
                rec["hit"] = int(rec["low_95"] <= actual <= rec["high_95"])
                updated = True

    # Rewrite file if any actuals were filled in
    if updated:
        with open(HISTORY_FILE, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    return records

# ── Load backtest results ─────────────────────────────────────────────────────
@st.cache_data
def load_backtest_results():
    try:
        records = []
        with open("backtest_results.jsonl") as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        return (
            df["coverage_95"].mean(),
            df["width_95"].mean(),
            df["winkler"].mean()
        )
    except FileNotFoundError:
        return None, None, None

# ═════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
st.title("₿ BTC/USDT — Next Hour Forecast")
st.caption(f"Last refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# ── Backtest metrics ──────────────────────────────────────────────────────────
coverage, avg_width, winkler = load_backtest_results()
st.subheader("Backtest Performance (719 predictions, 30-day window)")
col1, col2, col3 = st.columns(3)
col1.metric("Coverage (95% CI)", f"{coverage:.2%}" if coverage else "N/A",
            help="Target: ~95%")
col2.metric("Avg Range Width", f"${avg_width:,.0f}" if avg_width else "N/A",
            help="Narrower = better")
col3.metric("Winkler Score", f"{winkler:,.1f}" if winkler else "N/A",
            help="Lower = better")

st.divider()

# ── Fetch data + run model ────────────────────────────────────────────────────
with st.spinner("Fetching latest BTC data and running model..."):
    try:
        prices = get_btc_data(limit=500)
        low95, high95, current_price = run_model(prices)

        # The candle we're predicting = next hour
        last_candle_time = prices.index[-1]
        next_candle_time = last_candle_time + pd.Timedelta(hours=1)
        target_hour_str = str(next_candle_time)

        # ── Part C: Save this prediction ─────────────────────────────────────
        # Only save once per target hour (avoid duplicates on page refresh)
        history = load_and_update_history(prices)
        already_saved = any(r["target_hour"] == target_hour_str for r in history)
        if not already_saved:
            save_prediction(current_price, low95, high95, target_hour_str)
            history = load_and_update_history(prices)  # reload with new entry

        # ── Live prediction display ───────────────────────────────────────────
        st.subheader("Live Prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric("Current BTC Price", f"${current_price:,.2f}")
        c2.metric("Predicted Low (95% CI)", f"${low95:,.2f}")
        c3.metric("Predicted High (95% CI)", f"${high95:,.2f}")

        st.info(
            f"Next hour forecast: BTC has a 95% chance of being between "
            f"${low95:,.2f} and ${high95:,.2f}  |  "
            f"Range width: ${high95 - low95:,.2f}"
        )

        # ── Chart ─────────────────────────────────────────────────────────────
        st.subheader("Price Chart — Last 50 Hours + Next Hour Forecast")
        last50 = prices.tail(50)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=last50.index, y=last50.values,
            mode='lines', name='BTC Price',
            line=dict(color='#F7931A', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[last_candle_time, next_candle_time,
               next_candle_time, last_candle_time],
            y=[high95, high95, low95, low95],
            fill='toself',
            fillcolor='rgba(100, 149, 237, 0.3)',
            line=dict(color='rgba(0,0,0,0)'),
            name='95% Forecast Range'
        ))
        fig.add_trace(go.Scatter(
            x=[last_candle_time], y=[current_price],
            mode='markers',
            marker=dict(color='white', size=8),
            name='Current Price'
        ))
        fig.update_layout(
            template='plotly_dark',
            xaxis_title='Time (UTC)',
            yaxis_title='Price (USDT)',
            legend=dict(orientation='h', y=1.1),
            height=450,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Part C: Prediction History Table ─────────────────────────────────
        st.divider()
        st.subheader("📜 Prediction History")
        st.caption("Every visit saves a prediction. Actuals fill in automatically once the hour closes.")

        if history:
            # Most recent first
            df_hist = pd.DataFrame(history[::-1])

            # Format columns nicely
            df_hist["Current Price"] = df_hist["current_price"].apply(
                lambda x: f"${x:,.2f}")
            df_hist["Predicted Range"] = df_hist.apply(
                lambda r: f"${r['low_95']:,.2f} – ${r['high_95']:,.2f}", axis=1)
            df_hist["Width"] = df_hist.apply(
                lambda r: f"${r['high_95'] - r['low_95']:,.2f}", axis=1)
            df_hist["Actual Price"] = df_hist["actual_price"].apply(
                lambda x: f"${x:,.2f}" if x else "⏳ Pending")
            df_hist["Result"] = df_hist["hit"].apply(
                lambda x: "✅ Hit" if x == 1 else ("❌ Miss" if x == 0 else "⏳"))

            display_cols = {
                "predicted_at": "Predicted At",
                "target_hour": "Target Hour",
                "Current Price": "Current Price",
                "Predicted Range": "Predicted Range",
                "Width": "Width",
                "Actual Price": "Actual Price",
                "Result": "Result"
            }
            df_display = df_hist[list(display_cols.keys())].rename(
                columns=display_cols)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Live coverage from history
            completed = [r for r in history if r["hit"] is not None]
            if completed:
                live_coverage = sum(r["hit"] for r in completed) / len(completed)
                st.metric(
                    "Live Coverage (from history)",
                    f"{live_coverage:.2%}",
                    help="How often our live predictions have been correct so far"
                )
        else:
            st.info("No prediction history yet. Visit again after an hour to see actuals fill in!")

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()