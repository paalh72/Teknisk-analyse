import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go

# RSI
def rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# MA
def ma(data, period=20):
    return data['Close'].rolling(period).mean()

# MFI
def mfi(data, period=14):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    mf = tp * data['Volume']
    pos = mf.where(tp > tp.shift(), 0).rolling(period).sum()
    neg = mf.where(tp < tp.shift(), 0).rolling(period).sum()
    return 100 - (100 / (1 + pos / neg))

# MACD
def macd(data, short=12, long=26, signal=9):
    short_ema = data['Close'].ewm(span=short, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long, adjust=False).mean()
    macd_line = short_ema - long_ema
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line

# DeMark 9-13
def demark(data):
    # Ensure required columns exist
    required_columns = ['Close', 'High', 'Low', 'Volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"DataFrame missing required columns: {', '.join(set(required_columns) - set(data.columns))}")
    
    # Initialize columns
    data = data.copy()  # Avoid modifying the original DataFrame
    data["C4"] = data["Close"].shift(4)
    data["C2"] = data["Close"].shift(2)
    data["Setup"] = 0
    data["Countdown"] = 0
    
    count = 0
    for i in range(len(data)):
        if i < 4:  # Skip rows where C4 is NaN
            continue
        try:
            close_val = data["Close"].iloc[i]
            c4_val = data["C4"].iloc[i]
            if pd.isna(close_val) or pd.isna(c4_val):
                count = 0
            elif not (isinstance(close_val, (int, float)) and isinstance(c4_val, (int, float))):
                count = 0
            elif close_val > c4_val:
                count += 1
            else:
                count = 0
            data.at[data.index[i], "Setup"] = count
        except Exception as e:
            st.warning(f"Error processing Setup at index {i}: {str(e)}")
            count = 0
    
    cd = 0
    started = False
    for i in range(len(data)):
        if data["Setup"].iloc[i] == 9:
            started = True
            cd = 0
        if started and i >= 2:
            try:
                close_val = data["Close"].iloc[i]
                c2_val = data["C2"].iloc[i]
                if pd.isna(close_val) or pd.isna(c2_val):
                    cd = 0
                elif not (isinstance(close_val, (int, float)) and isinstance(c2_val, (int, float))):
                    cd = 0
                elif close_val > c2_val:
                    cd += 1
                data.at[data.index[i], "Countdown"] = cd
                if cd == 13:
                    started = False
            except Exception as e:
                st.warning(f"Error processing Countdown at index {i}: {str(e)}")
                cd = 0
    return data

# App UI
st.title("📈 Teknisk analyse – Oslo Børs")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
        if data.empty:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            # Ensure numeric columns
            numeric_columns = ['Close', 'High', 'Low', 'Volume']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if data[col].isna().all():
                    raise ValueError(f"Column '{col}' contains only non-numeric or missing values after conversion")
            
            # Debug: Inspect data
            # st.write("Data columns:", data.columns)
            # st.write("First few rows:", data.head())
            
            # Check if Close is numeric
            if not pd.api.types.is_numeric_dtype(data['Close']):
                raise ValueError("Column 'Close' must be numeric after conversion")
            
            # Calculate technical indicators
            data = demark(data)
            data["RSI"] = rsi(data)
            data["MA20"] = ma(data)
            data["MFI"] = mfi(data)
            data["MACD"], data["SIGNAL"] = macd(data)
            data["VolMA"] = data["Volume"].rolling(20).mean()

            # Pris + DeMark
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
            fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20", line=dict(dash="dot")))
            for i in data.index:
                if data.at[i, "Setup"] == 9:
                    fig.add_trace(go.Scatter(x=[i], y=[data.at[i, "Close"]], text=["9"], mode="markers+text", marker=dict(color="green", size=10)))
                if data.at[i, "Countdown"] == 13:
                    fig.add_trace(go.Scatter(x=[i], y=[data.at[i, "Close"]], text=["13"], mode="markers+text", marker=dict(color="red", size=10)))
            st.plotly_chart(fig, use_container_width=True)

            # RSI
            st.subheader("RSI")
            st.line_chart(data["RSI"])

            # MACD
            st.subheader("MACD")
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"))
            macd_fig.add_trace(go.Scatter(x=data.index, y=data["SIGNAL"], name="Signal", line=dict(dash="dot")))
            st.plotly_chart(macd_fig, use_container_width=True)

            # MFI
            st.subheader("MFI")
            st.line_chart(data["MFI"])

            # Volume
            st.subheader("Volum og snitt")
            vfig = go.Figure()
            vfig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"))
            vfig.add_trace(go.Scatter(x=data.index, y=data["VolMA"], name="VolMA"))
            st.plotly_chart(vfig, use_container_width=True)
    except Exception as e:
        st.error(f"En feil oppstod: {str(e)}")
