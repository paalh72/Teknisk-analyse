import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

# RSI
def rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta>0, 0).rolling(period).mean()
    loss = -delta.where(delta<0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

# MA
def ma(data, period=20):
    return data['Close'].rolling(period).mean()

# MFI
def mfi(data, period=14):
    tp = (data['High']+data['Low']+data['Close'])/3
    mf = tp * data['Volume']
    pos = mf.where(tp>tp.shift(), 0).rolling(period).sum()
    neg = mf.where(tp<tp.shift(), 0).rolling(period).sum()
    return 100 - (100/(1 + pos/neg))

# MACD
def macd(data, short=12, long=26, signal=9):
    short_ema = data['Close'].ewm(span=short).mean()
    long_ema = data['Close'].ewm(span=long).mean()
    macd_line = short_ema - long_ema
    sig_line = macd_line.ewm(span=signal).mean()
    return macd_line, sig_line

# DeMark 9-13
def demark(data):
    data["C4"] = data["Close"].shift(4)
    data["C2"] = data["Close"].shift(2)
    data["Setup"] = 0
    count = 0
    for i in range(len(data)):
        if i<4: continue
        if data["Close"].iloc[i] > data["C4"].iloc[i]: count+=1
        else: count=0
        data.at[data.index[i], "Setup"] = count
    data["Countdown"] = 0
    cd = 0; started=False
    for i in range(len(data)):
        if data["Setup"].iloc[i]==9:
            started=True; cd=0
        if started and i>=2:
            if data["Close"].iloc[i] > data["C2"].iloc[i]: cd+=1
            data.at[data.index[i], "Countdown"] = cd
            if cd==13: started=False
    return data

# App UI
st.title("📈 Teknisk analyse – Oslo Børs")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    data = yf.download(ticker, period=period, interval="1d")
    if data.empty:
        st.warning("Fant ikke data.")
    else:
        data = demark(data)
        data["RSI"] = rsi(data)
        data["MA20"] = ma(data)
        data["MFI"] = mfi(data)
        data["MACD"], data["SIGNAL"] = macd(data)
        data["VolMA"] = data["Volume"].rolling(20).mean()

        # Pris + demark
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20", line=dict(dash="dot")))
        for i in data.index:
            if data.at[i, "Setup"]==9:
                fig.add_trace(go.Scatter(x=[i], y=[data.at[i,"Close"]], text=["9"], mode="markers+text", marker=dict(color="green",size=10)))
            if data.at[i, "Countdown"]==13:
                fig.add_trace(go.Scatter(x=[i], y=[data.at[i,"Close"]], text=["13"], mode="markers+text", marker=dict(color="red",size=10)))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("RSI")
        st.line_chart(data["RSI"])

        st.subheader("MACD")
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"))
        macd_fig.add_trace(go.Scatter(x=data.index, y=data["SIGNAL"], name="Signal", line=dict(dash="dot")))
        st.plotly_chart(macd_fig, use_container_width=True)

        st.subheader("MFI")
        st.line_chart(data["MFI"])

        st.subheader("Volum og snitt")
        vfig = go.Figure()
        vfig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"))
        vfig.add_trace(go.Scatter(x=data.index, y=data["VolMA"], name="VolMA"))
        st.plotly_chart(vfig, use_container_width=True)
