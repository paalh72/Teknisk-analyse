import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# Fetch stock data
def fetch_data(ticker, period):
    try:
        # Adjust intervals based on yfinance limitations
        intervals = {
            "1d": "5m",
            "1wk": "15m",
            "1mo": "1h",
            "3mo": "1d",
            "6mo": "1d",
            "1y": "1d",
            "5y": "1wk"
        }
        interval = intervals.get(period, "1d")
        # Validate ticker format
        if not ticker or not isinstance(ticker, str) or len(ticker) < 1:
            raise ValueError("Invalid ticker format")
        
        # Clean ticker (remove spaces, ensure uppercase)
        ticker = ticker.strip().upper()
        
        # Try fetching data
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, prepost=False, threads=False)
        
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker} with period {period} and interval {interval}")
        
        if 'Close' not in data.columns or 'Volume' not in data.columns:
            raise ValueError("Missing 'Close' or 'Volume' column in yfinance data")
        
        data = data.dropna(how='all')
        data.index = pd.to_datetime(data.index, utc=True).tz_convert(None)
        close = np.ravel(data['Close'].values)
        volume = np.ravel(data['Volume'].values)
        dates = np.array(data.index, dtype='datetime64[ms]')
        
        st.write(f"Fetch data: ticker={ticker}, period={period}, interval={interval}, close shape={close.shape}, volume shape={volume.shape}, dates shape={dates.shape}, close dtype={close.dtype}")
        
        return close, volume, dates
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        # Fallback to static data for testing
        st.warning("Using static data as fallback due to yfinance failure")
        periods = {"1d": 288, "1wk": 480, "1mo": 672, "3mo": 63, "6mo": 126, "1y": 252, "5y": 1260}
        n = periods.get(period, 252)
        dates = pd.date_range(end='2025-07-05', periods=n, freq='B')
        close = np.random.uniform(100, 200, n)
        volume = np.random.randint(100000, 1000000, n)
        dates_np = np.array(dates, dtype='datetime64[ms]')
        st.write(f"Fallback static data: close shape={close.shape}, volume shape={volume.shape}, dates shape={dates_np.shape}")
        return close, volume, dates_np

# Calculate Moving Average
def calculate_ma(close, window):
    try:
        if not isinstance(close, np.ndarray) or close.ndim != 1:
            raise ValueError(f"Close must be a 1D NumPy array, got {type(close)} with shape {close.shape}")
        kernel = np.ones(window) / window
        ma = np.convolve(close, kernel, mode='valid')
        ma = np.pad(ma, (window-1, 0), mode='constant', constant_values=np.nan)
        st.write(f"MA output: ma shape={ma.shape}, ma dtype={ma.dtype}")
        return ma
    except Exception as e:
        st.error(f"Error in MA calculation: {str(e)}")
        return np.full(len(close), np.nan)

# App UI
st.title("📈 Teknisk analyse – Aksjekurser")
st.write("Skriv inn ticker (f.eks. DNB.OL for Oslo Børs, AAPL for NYSE)")
ticker = st.text_input("Ticker", "DNB.OL")
period = st.selectbox("Periode", ["1d", "1wk", "1mo", "3mo", "6mo", "1y", "5y"], index=4)
show_ma = st.checkbox("Vis Moving Average", value=True)
ma_window = st.slider("MA-vindu (dager)", min_value=5, max_value=50, value=20, disabled=not show_ma)
show_volume = st.checkbox("Vis Volum", value=True)

if ticker:
    close, volume, dates = fetch_data(ticker, period)
    if close is None or volume is None or dates is None:
        st.warning(f"Kunne ikke hente data for ticker: {ticker}")
    else:
        try:
            # Calculate Moving Average if enabled
            ma = calculate_ma(close, ma_window) if show_ma else np.full(len(close), np.nan)

            # Create Plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=close, name="Close", line=dict(color="#1f77b4")))

            # Add Moving Average
            if show_ma and not np.all(np.isnan(ma)):
                fig.add_trace(go.Scatter(x=dates, y=ma, name=f"MA-{ma_window}", line=dict(color="#ff7f0e")))

            # Add Volume as bar chart
            if show_volume:
                fig.add_trace(go.Bar(x=dates, y=volume, name="Volume", yaxis="y2", opacity=0.3, marker=dict(color="#636efa")))

            # Update layout with dual y-axes if volume is shown
            fig.update_layout(
                title=f"{ticker} - Aksjekurs og indikatorer",
                xaxis_title="Dato",
                yaxis_title="Pris",
                showlegend=True,
                yaxis=dict(title="Pris"),
                yaxis2=dict(title="Volum", overlaying="y", side="right", showgrid=False) if show_volume else None,
                barmode="overlay"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary statistics
            st.subheader("Siste verdier")
            st.metric("Pris", f"{close[-1]:.2f}" if not np.isnan(close[-1]) else "N/A")
            if show_ma:
                st.metric(f"MA-{ma_window}", f"{ma[-1]:.2f}" if not np.isnan(ma[-1]) else "N/A")
            if show_volume:
                st.metric("Volum", f"{int(volume[-1])}" if not np.isnan(volume[-1]) else "N/A")

        except Exception as e:
            st.error(f"En feil oppstod: {str(e)}")
