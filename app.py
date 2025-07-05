import streamlit as st
import plotly.graph_objs as go
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime

ALPHA_VANTAGE_API_KEY = 'YOUR_API_KEY'  # Replace with your Alpha Vantage API key

st.title("📈 Minimal Teknisk analyse – Oslo Børs")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        def fetch_data(ticker, period):
            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='json')
            outputsize = 'compact' if period == '3mo' else 'full'
            data, meta = ts.get_daily(symbol=ticker, outputsize=outputsize)
            dates = []
            closes = []
            for date, values in data.items():
                dates.append(datetime.strptime(date, '%Y-%m-%d'))
                closes.append(float(values['4. close']))
            dates = np.array(dates, dtype='datetime64[D]')
            closes = np.array(closes, dtype=np.float64)
            if np.isnan(closes).all():
                raise ValueError("Close data contains only NaN values")
            return dates, closes

        dates, closes = fetch_data(ticker, period)
        if len(dates) == 0:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=closes, name="Close"))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"En feil oppstod: {str(e)}")
