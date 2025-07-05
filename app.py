import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from alpha_vantage.timeseries import TimeSeries

ALPHA_VANTAGE_API_KEY = 'YOUR_API_KEY'  # Replace with your Alpha Vantage API key

st.title("📈 Minimal Teknisk analyse – Oslo Børs")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        @st.cache_data
        def fetch_data(ticker, period, _version=11):
            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
            outputsize = 'compact' if period == '3mo' else 'full'
            data, meta = ts.get_daily(symbol=ticker, outputsize=outputsize)
            data = data.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
            data.index = pd.to_datetime(data.index, utc=True).tz_convert(None)
            data = data.sort_index()
            data['Close'] = pd.to_numeric(data['Close'], errors='coerce').astype('float64')
            if data['Close'].isna().all():
                raise ValueError("Column 'Close' contains only non-numeric or missing values")
            return data

        data = fetch_data(ticker, period)
        if data.empty:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            # Convert to NumPy to avoid pandas serialization
            close = data['Close'].to_numpy()
            index = data.index.to_pydatetime()  # Convert index to Python datetime
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=index, y=close, name="Close"))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"En feil oppstod: {str(e)}")
