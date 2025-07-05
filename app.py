import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# DeMark 9-13
def demark(data):
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Input must be a pandas DataFrame, got {type(data)}")
        if 'Close' not in data.columns:
            raise ValueError("DataFrame missing 'Close' column")
        close = pd.Series(data['Close'], index=data.index, dtype='float64')
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError(f"Close column must be numeric, got {close.dtype}")
        if close.isna().all():
            raise ValueError("Close column contains only NaN values")
        
        setup = pd.Series(0, index=data.index, dtype='int64')
        countdown = pd.Series(0, index=data.index, dtype='int64')
        c4 = close.shift(4)
        c2 = close.shift(2)
        
        for i in range(4, len(close)):
            if pd.isna(close.iloc[i]) or pd.isna(c4.iloc[i]):
                setup.iloc[i] = 0
            elif close.iloc[i] > c4.iloc[i]:
                setup.iloc[i] = setup.iloc[i-1] + 1 if i > 0 else 1
            else:
                setup.iloc[i] = 0
        
        cd = 0
        started = False
        for i in range(2, len(close)):
            if setup.iloc[i] == 9:
                started = True
                cd = 0
            if started:
                if pd.isna(close.iloc[i]) or pd.isna(c2.iloc[i]):
                    cd = 0
                elif close.iloc[i] > c2.iloc[i]:
                    cd += 1
                countdown.iloc[i] = cd
                if cd == 13:
                    started = False
        
        return pd.DataFrame({
            'Close': close,
            'Setup': setup,
            'Countdown': countdown
        }, index=data.index)
    except Exception as e:
        st.error(f"Error in DeMark calculation: {str(e)}")
        return pd.DataFrame({
            'Close': data['Close'],
            'Setup': np.zeros(len(data), dtype='int64'),
            'Countdown': np.zeros(len(data), dtype='int64')
        }, index=data.index)

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
            data['Close'] = pd.to_numeric(data['Close'], errors='coerce').astype('float64')
            if data['Close'].isna().all():
                raise ValueError("Close data contains only NaN values")
            return data[['Close']]

        data = fetch_data(ticker, period)
        if data.empty:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            # Debug data type
            st.write(f"Data type: {type(data)}, Columns: {data.columns}, Close dtype: {data['Close'].dtype}")
            
            # Calculate DeMark
            data = demark(data)

            # Convert to NumPy for plotting
            dates = np.array(data.index, dtype='datetime64[ms]')
            close = data['Close'].to_numpy()
            setup = data['Setup'].to_numpy()
            countdown = data['Countdown'].to_numpy()

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
