import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# DeMark 9-13
def demark(data):
    try:
        if not isinstance(data['Close'], pd.Series):
            raise ValueError("Input 'Close' column must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(data['Close']):
            raise ValueError("Column 'Close' must be numeric")
        result_data = data.copy()
        result_data["C4"] = data["Close"].shift(4)
        result_data["C2"] = data["Close"].shift(2)
        result_data["Setup"] = pd.Series(np.zeros(len(data), dtype='int64'), index=data.index)
        result_data["Countdown"] = pd.Series(np.zeros(len(data), dtype='int64'), index=data.index)
        count = 0
        for i in range(len(data)):
            if i < 4:
                continue
            if pd.isna(data["Close"].iloc[i]) or pd.isna(result_data["C4"].iloc[i]):
                count = 0
            elif data["Close"].iloc[i] > result_data["C4"].iloc[i]:
                count += 1
            else:
                count = 0
            result_data["Setup"].iloc[i] = count
        cd = 0
        started = False
        for i in range(len(data)):
            if result_data["Setup"].iloc[i] == 9:
                started = True
                cd = 0
            if started and i >= 2:
                if pd.isna(data["Close"].iloc[i]) or pd.isna(result_data["C2"].iloc[i]):
                    cd = 0
                elif data["Close"].iloc[i] > result_data["C2"].iloc[i]:
                    cd += 1
                result_data["Countdown"].iloc[i] = cd
                if cd == 13:
                    started = False
        return result_data[["Close", "Setup", "Countdown"]]
    except Exception as e:
        st.error(f"Error in DeMark calculation: {str(e)}")
        return pd.DataFrame({"Close": data["Close"], "Setup": np.zeros(len(data), dtype='int64'), 
                            "Countdown": np.zeros(len(data), dtype='int64')}, index=data.index)

# App UI
st.title("📈 Teknisk analyse – Oslo Børs (DeMark)")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        def fetch_data(ticker, period):
            data = yf.download(ticker, period=period, interval="1d", auto_adjust=False, prepost=False, threads=False)
            if data.empty or 'Close' not in data.columns:
                return pd.DataFrame()
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
            # Calculate DeMark
            data = demark(data)

            # Plot Close price with DeMark signals
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="#1f77b4")))
            setup_9_indices = data[data["Setup"] == 9].index
            countdown_13_indices = data[data["Countdown"] == 13].index
            if len(setup_9_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=setup_9_indices, 
                    y=data.loc[setup_9_indices, "Close"], 
                    text=["9"] * len(setup_9_indices), 
                    mode="markers+text", 
                    marker=dict(color="green", size=10),
                    name="Setup 9"
                ))
            if len(countdown_13_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=countdown_13_indices, 
                    y=data.loc[countdown_13_indices, "Close"], 
                    text=["13"] * len(countdown_13_indices), 
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
            st.metric("Pris", f"{data['Close'].iloc[-1]:.2f}" if not pd.isna(data['Close'].iloc[-1]) else "N/A")

    except Exception as e:
        st.error(f"En feil oppstod: {str(e)}")
