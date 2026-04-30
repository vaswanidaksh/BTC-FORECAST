import streamlit as st
import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from arch import arch_model
import plotly.graph_objects as go
from datetime import datetime
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Forecast Dashboard",
    page_icon="₿",
    layout="wide"
)

# ── Helper: Fetch BTC data ────────────────────────────────────────────────────
@st.cache_data(ttl=300)  # cache for 5 minutes
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

# ── Helper: Run GBM model and return 95% range ───────────────────────────────
def run_model(prices):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    
    # Fit FIGARCH model
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
    n_sims = 10_000
    
    simulated = np.array([
        simulate_once(S0, mu, sigma_fig, H_series, M_series,
                      base_params.copy(), bar_sigma2)
        for _ in range(n_sims)
    ])
    
    low95, high95 = np.percentile(simulated, [5.0, 95.0])
    return low95, high95, S0

# ── Load backtest results ─────────────────────────────────────────────────────
@st.cache_data
def load_backtest_results():
    records = []
    try:
        with open("backtest_results.jsonl") as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        coverage = df["coverage_95"].mean()
        avg_width = df["width_95"].mean()
        winkler = df["winkler"].mean()
        return coverage, avg_width, winkler
    except FileNotFoundError:
        return None, None, None

# ── Main Dashboard ────────────────────────────────────────────────────────────
st.title("₿ BTC/USDT — Next Hour Forecast")
st.caption(f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# Load backtest metrics
coverage, avg_width, winkler = load_backtest_results()

# Show backtest metrics at top
st.subheader("Backtest Performance (719 predictions, 30-day window)")
col1, col2, col3 = st.columns(3)
col1.metric(
    "Coverage (95% CI)",
    f"{coverage:.2%}" if coverage else "N/A",
    help="Target: ~95%. How often real price fell inside our predicted range."
)
col2.metric(
    "Avg Range Width",
    f"${avg_width:,.0f}" if avg_width else "N/A",
    help="Average width of our predicted range. Narrower = better."
)
col3.metric(
    "Winkler Score",
    f"{winkler:,.1f}" if winkler else "N/A",
    help="Combined accuracy + tightness score. Lower = better."
)

st.divider()

# Fetch live data and run model
with st.spinner("Fetching latest BTC data and running model..."):
    try:
        prices = get_btc_data(limit=500)
        low95, high95, current_price = run_model(prices)

        # Current price + prediction
        st.subheader("Live Prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric("Current BTC Price", f"${current_price:,.2f}")
        c2.metric("Predicted Low (95% CI)", f"${low95:,.2f}")
        c3.metric("Predicted High (95% CI)", f"${high95:,.2f}")

        st.info(
            f"**Next hour forecast:** BTC has a 95% chance of being between "
            f"**${low95:,.2f}** and **${high95:,.2f}**  |  "
            f"Range width: **${high95 - low95:,.2f}**"
        )

        # Chart: last 50 bars + shaded prediction
        st.subheader("Price Chart — Last 50 Hours + Next Hour Forecast")
        last50 = prices.tail(50)
        
        # Create next hour timestamp
        last_time = last50.index[-1]
        next_time = last_time + pd.Timedelta(hours=1)

        fig = go.Figure()

        # Price line
        fig.add_trace(go.Scatter(
            x=last50.index, y=last50.values,
            mode='lines', name='BTC Price',
            line=dict(color='#F7931A', width=2)
        ))

        # Shaded prediction band
        fig.add_trace(go.Scatter(
            x=[last_time, next_time, next_time, last_time],
            y=[high95, high95, low95, low95],
            fill='toself',
            fillcolor='rgba(100, 149, 237, 0.3)',
            line=dict(color='rgba(0,0,0,0)'),
            name='95% Forecast Range'
        ))

        # Current price dot
        fig.add_trace(go.Scatter(
            x=[last_time], y=[current_price],
            mode='markers',
            marker=dict(color='white', size=8, symbol='circle'),
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

    except Exception as e:
        st.error(f"Error running model: {e}")
        st.stop()