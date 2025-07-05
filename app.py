import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# DeMark 9-13
def demark(close_series):
    try:
        if not isinstance(close_series, pd.Series):
            raise ValueError(f"Input must be a pandas Series, got {type(close_series)}")
        if not pd.api.types.is_numeric_dtype(close_series):
            raise ValueError(f"Close series must be numeric, got {close_series.dtype}")
        if close_series.isna().all():
            raise ValueError("Close series contains only NaN values")
        
        st.write(f"DeMark input: type={type(close_series)}, dtype={close_series.dtype}, len={len(close_series)}")
        
        setup = np.zeros(len(close_series), dtype='int64')
        countdown = np.zeros(len(close_series), dtype='int64')
        close = close_series.to_numpy()
        c4 = np.roll(close, 4)
        c2 = np.roll(close, 2)
        
        for i in range(4, len(close)):
            if np.isnan(close[i]) or np.isnan(c4[i]):
                setup[i] = 0
            elif close[i] > c4[i]:
                setup[i] = setup[i-1] + 1 if i > 0 else 1
            else:
                setup[i] = 0
        
        cd = 0
        started = False
        for i in range(2, len(close)):
            if setup[i] == 9:
                started = True
                cd = 0
            if started:
                if np.isnan(close[i]) or np.isnan(c2[i]):
                    cd = 0
                elif close[i] > c2[i]:
                    cd += 1
                countdown[i] = cd
                if cd == 13:
                    started = False
        
        return pd.DataFrame({
            'Close': close_series,
            'Setup': setup,
            'Countdown': countdown
        }, index=close_series.index)
    except Exception as e:
        st.error(f"Error in DeMark calculation: {str(e)}")
        return pd.DataFrame({
            'Close': close_series,
            'Setup': np.zeros(len(close_series), dtype='int64'),
            'Countdown': np.zeros(len(close_series), dtype='int64')
        }, index=close_series.index)

# App UI
st.title("📈 Teknisk analyse – Oslo Børs (DeMark)")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

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
            close = pd.Series(data['Close'], dtype='float64')
            if close.isna().all():
                raise ValueError("Close data contains only NaN values")
            st.write(f"Fetch data: type={type(data)}, columns={data.columns}, Close dtype={close.dtype}")
            return close

        close_series = fetch_data(ticker, period)
        if close_series.empty:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            # Calculate DeMark
            data = demark(close_series)

            # Convert to NumPy for plotting
            dates = np.array(data.index, dtype='datetime64[ms]')
            close = data['Close'].to_numpy()
            setup = data['Setup'].to_numpy()
            countdown = data['Countdown'].to_numpy()
            st.write(f"Plot data: dates type={type(dates)}, close type={type(close)}, setup type={type(setup)}")

            # Plot Close price with DeMark signals
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=close, name="Close", line=dict(color="#1f77b4")))
            setup_9_mask = setup == 9
            countdown_13_mask = countdown == 13
            if np.any(setup_9_mask):
                fig.add_trace(go.Scatter(
                    x=dates[setup_9_mask], 
                    y=close[setup_9_mask], 
                    text=["9"] * np.sum(setup_9_mask), 
                    mode="markers+text", 
                    marker=dict(color="green", size=10),
                    name="Setup 9"
                ))
            if np.any(countdown_13_mask):
                fig.add_trace(go.Scatter(
                    x=dates[countdown_13_mask], 
                    y=close[countdown_13_mask], 
                    text=["13"] * np.sum(countdown_13_mask), 
                    mode="markers+text", 
                    marker=dict(color="red", size=10),
                    name="Countdown 13"
                ))
            fig.update_layout(
                title=f"{ticker} - Pris og DeMark signaler",
                xaxis_title="Dato",
                yaxis_title="Pris",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary statistic
            st.subheader("Siste verdi")
            st.metric("Pris", f"{close[-1]:.2f}" if not np.isnan(close[-1]) else "N/A")

    except Exception as e:
        st.error(f"En feil oppstod: {str(e)}")
