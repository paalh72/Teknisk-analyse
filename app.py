import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# ------------------------ Teknisk indikatorer ------------------------

def rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def ma(data, period=20):
    return data['Close'].rolling(window=period).mean()

def mfi(data, period=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    neg_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    pos_mf = pos_flow.rolling(window=period).sum()
    neg_mf = neg_flow.rolling(window=period).sum()
    mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
    return mfi

def demark(data):
    data = data.copy()
    data["C4"] = data["Close"].shift(4)
    data["Setup"] = 0
    count = 0
    for i in range(4, len(data)):
        close_val = data["Close"].iloc[i]
        c4_val = data["C4"].iloc[i]
        if pd.notna(c4_val) and pd.notna(close_val) and float(close_val) > float(c4_val):
            count += 1
        else:
            count = 0
        data.at[data.index[i], "Setup"] = count
    data["Countdown"] = 0  # Placeholder
    return data

# ------------------------ Streamlit App ------------------------

st.set_page_config(page_title="Teknisk analyse – Oslo Børs", layout="wide")
st.title("📈 Teknisk analyse – Oslo Børs")

# Input fra bruker
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=True)

        if data.empty:
            st.warning("❗ Fant ikke data for valgt ticker.")
        else:
            # Beregninger
            data = demark(data)
            data["RSI"] = rsi(data)
            data["MA20"] = ma(data)
            data["MFI"] = mfi(data)

            # Plot: Kurs med glidende snitt
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
            fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20", line=dict(dash='dot')))
            fig.update_layout(title=f"{ticker} - Kurs og MA", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Plot: RSI og MFI
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color='orange')))
            fig2.add_trace(go.Scatter(x=data.index, y=data["MFI"], name="MFI", line=dict(color='green')))
            fig2.update_layout(title="RSI og MFI", height=400)
            st.plotly_chart(fig2, use_container_width=True)

            # Vise tabell nederst
            st.subheader("📊 Rådata (siste 10 dager)")
            st.dataframe(data.tail(10))

    except Exception as e:
        st.error(f"⚠️ En feil oppstod: {e}")
