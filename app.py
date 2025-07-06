import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

st.title("📈 DeMark 9-13 Analyse – Oslo Børs")

# Input
ticker = st.text_input("Skriv inn tickeren (f.eks. DNB.OL for DNB)", "DNB.OL")
period = st.selectbox("Velg tidsperiode", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

# Hent data
data = yf.download(ticker, period=period, interval="1d")

if data.empty:
    st.warning("Fant ikke data for tickeren.")
    st.stop()

data["Close Shift 4"] = data["Close"].shift(4)
data["Close Shift 2"] = data["Close"].shift(2)

# Setup 9
data["Setup Count"] = 0
count = 0
for i in range(len(data)):
    if i < 4:
        data.at[data.index[i], "Setup Count"] = 0
        continue
    if data["Close"].iloc[i] > data["Close Shift 4"].iloc[i]:
        count += 1
    else:
        count = 0
    data.at[data.index[i], "Setup Count"] = count

# Countdown 13
data["Countdown Count"] = 0
countdown = 0
countdown_started = False
for i in range(len(data)):
    if data["Setup Count"].iloc[i] == 9:
        countdown_started = True
        countdown = 0
    if countdown_started and i >= 2:
        if data["Close"].iloc[i] > data["Close Shift 2"].iloc[i]:
            countdown += 1
        data.at[data.index[i], "Countdown Count"] = countdown
        if countdown == 13:
            countdown_started = False

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Kurs", mode="lines"))

# Marker 9 og 13
for i in range(len(data)):
    if data["Setup Count"].iloc[i] == 9:
        fig.add_trace(go.Scatter(x=[data.index[i]], y=[data["Close"].iloc[i]],
                                 mode='markers+text', text=["9"], textposition="top center",
                                 marker=dict(color='green', size=10), name="Setup 9"))
    if data["Countdown Count"].iloc[i] == 13:
        fig.add_trace(go.Scatter(x=[data.index[i]], y=[data["Close"].iloc[i]],
                                 mode='markers+text', text=["13"], textposition="bottom center",
                                 marker=dict(color='red', size=10), name="Countdown 13"))

fig.update_layout(title=f"Kurs og DeMark 9-13 for {ticker}", xaxis_title="Dato", yaxis_title="Pris")
st.plotly_chart(fig, use_container_width=True)
