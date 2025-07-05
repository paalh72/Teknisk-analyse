import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# Simple Moving Average
def calculate_ma(close_series, window=20):
    try:
        if not isinstance(close_series, pd.Series):
            raise ValueError(f"Input must be a pandas Series, got {type(close_series)}")
        if not pd.api.types.is_numeric_dtype(close_series):
            raise ValueError(f"Close series must be numeric, got {close_series.dtype}")
        if close_series.isna().all():
            raise ValueError("Close series contains only NaN values")
        
        st.write(f"MA input: type={type(close_series)}, dtype={close_series.dtype}, shape={close_series.shape}")
        
        # Calculate MA
        ma = close_series.rolling(window=window, min_periods=1).mean()
        close_np = np.ravel(close_series.to_numpy())
        ma_np = np.ravel(ma.to_numpy())
        
        st.write(f"MA output: close_np shape={close_np.shape}, ma_np shape={ma_np.shape}")
        
        return close_np, ma_np, close_series.index
    except Exception as e:
        st.error(f"Error in MA calculation: {str(e)}")
        return np.zeros(len(close_series)), np.zeros(len(close_series)), close_series.index

# App UI
st.title("📈 Teknisk analyse – Oslo Børs (Moving Average)")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)
ma_window = st.slider("MA-vindu (dager)", min_value=5, max_value=50, value=20)

if ticker:
    try:
        def fetch_data(ticker, period):
            data = yf.download(ticker, period=period, interval="1d", auto_adjust=False, prepost=False, threads=False)
            if data.empty:
                raise ValueError("No data returned from yfinance")
            if 'Close' not in data.columns:
                raise ValueError("Close column missing in yfinance data")
            data = data.dropna(how='all')
            data.index = pd.to_datetime(data.index, utc=True).tz_convert(None)
            close = pd.Series(data['Close'].values, index=data.index, dtype='float64')
            if close.isna().all():
                raise ValueError("Close data contains only NaN values")
            st.write(f"Fetch data: type={type(close)}, dtype={close.dtype}, shape={close.shape}")
            return close

        close_series = fetch_data(ticker, period)
        if close_series.empty:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            # Calculate Moving Average
            close_np, ma_np, dates = calculate_ma(close_series, window=ma_window)

            # Convert dates to NumPy for plotting
            dates_np = np.array(dates, dtype='datetime64[ms]')
            st.write(f"Plot data: dates shape={dates_np.shape}, close shape={close_np.shape}, ma shape={ma_np.shape}")

            # Plot Close price with MA
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates_np, y=close_np, name="Close", line=dict(color="#1f77b4")))
            fig.add_trace(go.Scatter(x=dates_np, y=ma_np, name=f"MA-{ma_window}", line=dict(color="#ff7f0e")))
            fig.update_layout(
                title=f"{ticker} - Pris og {ma_window}-dagers Moving Average",
                xaxis_title="Dato",
                yaxis_title="Pris",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary statistic
            st.subheader("Siste verdier")
            st.metric("Pris", f"{close_np[-1]:.2f}" if not np.isnan(close_np[-1]) else "N/A")
            st.metric(f"MA-{ma_window}", f"{ma_np[-1]:.2f}" if not np.isnan(ma_np[-1]) else "N/A")

    except Exception as e:
        st.error(f"En feil oppstod: {str(e)}")
