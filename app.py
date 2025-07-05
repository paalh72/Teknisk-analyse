import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from alpha_vantage.timeseries import TimeSeries

# Replace with your Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = 'YOUR_API_KEY'  # Get a free key at https://www.alphavantage.co

# RSI
def rsi(data, period=14):
    try:
        if not isinstance(data['Close'], pd.Series):
            raise ValueError("Input 'Close' column must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(data['Close']):
            raise ValueError("Column 'Close' must be numeric")
        if data['Close'].isna().all():
            raise ValueError("Column 'Close' contains only NaN values")
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.where(loss != 0, np.finfo(float).eps)
        result = 100 - (100 / (1 + rs))
        result = pd.to_numeric(result, errors='coerce').astype('float64')
        result = result.replace([np.inf, -np.inf], np.nan)
        if result.isna().all():
            raise ValueError("RSI contains only NaN values")
        return result
    except Exception as e:
        st.error(f"Error in RSI calculation: {str(e)}")
        return None

# MA
def ma(data, period=20):
    try:
        if not isinstance(data['Close'], pd.Series):
            raise ValueError("Input 'Close' column must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(data['Close']):
            raise ValueError("Column 'Close' must be numeric")
        if data['Close'].isna().all():
            raise ValueError("Column 'Close' contains only NaN values")
        result = data['Close'].rolling(window=period, min_periods=1).mean()
        result = pd.to_numeric(result, errors='coerce').astype('float64')
        result = result.replace([np.inf, -np.inf], np.nan)
        if result.isna().all():
            raise ValueError("MA contains only NaN values")
        return result
    except Exception as e:
        st.error(f"Error in MA calculation: {str(e)}")
        return None

# MFI
def mfi(data, period=14):
    try:
        if not all(isinstance(data[col], pd.Series) for col in ['High', 'Low', 'Close', 'Volume']):
            raise ValueError("Input columns 'High', 'Low', 'Close', 'Volume' must be pandas Series")
        if not all(pd.api.types.is_numeric_dtype(data[col]) for col in ['High', 'Low', 'Close', 'Volume']):
            raise ValueError("Columns 'High', 'Low', 'Close', 'Volume' must be numeric")
        if data[['High', 'Low', 'Close', 'Volume']].isna().all().all():
            raise ValueError("Columns 'High', 'Low', 'Close', 'Volume' contain only NaN values")
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        mf = tp * data['Volume']
        pos = mf.where(tp > tp.shift(), 0).rolling(window=period, min_periods=1).sum()
        neg = mf.where(tp < tp.shift(), 0).rolling(window=period, min_periods=1).sum()
        result = 100 - (100 / (1 + pos / neg.where(neg != 0, np.finfo(float).eps)))
        result = pd.to_numeric(result, errors='coerce').astype('float64')
        result = result.replace([np.inf, -np.inf], np.nan)
        if result.isna().all():
            raise ValueError("MFI contains only NaN values")
        return result
    except Exception as e:
        st.error(f"Error in MFI calculation: {str(e)}")
        return None

# MACD
def macd(data, short=12, long=26, signal=9):
    try:
        if not isinstance(data['Close'], pd.Series):
            raise ValueError("Input 'Close' column must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(data['Close']):
            raise ValueError("Column 'Close' must be numeric")
        if data['Close'].isna().all():
            raise ValueError("Column 'Close' contains only NaN values")
        short_ema = data['Close'].ewm(span=short, adjust=False).mean()
        long_ema = data['Close'].ewm(span=long, adjust=False).mean()
        macd_line = short_ema - long_ema
        sig_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_line = pd.to_numeric(macd_line, errors='coerce').astype('float64')
        sig_line = pd.to_numeric(sig_line, errors='coerce').astype('float64')
        macd_line = macd_line.replace([np.inf, -np.inf], np.nan)
        sig_line = sig_line.replace([np.inf, -np.inf], np.nan)
        if macd_line.isna().all() or sig_line.isna().all():
            raise ValueError("MACD or Signal contains only NaN values")
        return macd_line, sig_line
    except Exception as e:
        st.error(f"Error in MACD calculation: {str(e)}")
        return None, None

# DeMark 9-13
def demark(data):
    try:
        required_columns = ['Close', 'High', 'Low', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"DataFrame missing required columns: {', '.join(set(required_columns) - set(data.columns))}")
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column '{col}' must be numeric")
            if data[col].isna().all():
                raise ValueError(f"Column '{col}' contains only NaN values")
        data = data.copy()
        data["C4"] = data["Close"].shift(4)
        data["C2"] = data["Close"].shift(2)
        data["Setup"] = 0
        data["Countdown"] = 0
        count = 0
        for i in range(len(data)):
            if i < 4:
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
    except Exception as e:
        st.error(f"Error in DeMark calculation: {str(e)}")
        return None

# App UI
st.title("📈 Teknisk analyse – Oslo Børs")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        # Cache data fetching
        @st.cache_data
        def fetch_data(ticker, period, _version=9):
            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
            outputsize = 'compact' if period == '3mo' else 'full'
            data, meta = ts.get_daily(symbol=ticker, outputsize=outputsize)
            data = data.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            return data

        data = fetch_data(ticker, period)
        if data.empty:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            # Validate input data
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if data[col].isna().all():
                    st.error(f"Column '{col}' contains only non-numeric or missing values")
                    raise ValueError(f"Column '{col}' contains only non-numeric or missing values")

            # Minimal DataFrame for plotting
            plot_data = pd.DataFrame(index=data.index)
            plot_data['Close'] = data['Close']
            plot_data['Volume'] = data['Volume']

            # Calculate indicators one at a time
            demark_data = demark(data)
            if demark_data is not None:
                plot_data['Setup'] = demark_data['Setup']
                plot_data['Countdown'] = demark_data['Countdown']
            else:
                plot_data['Setup'] = 0
                plot_data['Countdown'] = 0

            rsi_result = rsi(data)
            if rsi_result is not None:
                plot_data['RSI'] = rsi_result

            ma_result = ma(data)
            if ma_result is not None:
                plot_data['MA20'] = ma_result

            mfi_result = mfi(data)
            if mfi_result is not None:
                plot_data['MFI'] = mfi_result

            macd_result, signal_result = macd(data)
            if macd_result is not None and signal_result is not None:
                plot_data['MACD'] = macd_result
                plot_data['SIGNAL'] = signal_result

            plot_data['VolMA'] = data["Volume"].rolling(window=20, min_periods=1).mean()

            # Pris + DeMark
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data["Close"], name="Close"))
            if 'MA20' in plot_data:
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data["MA20"], name="MA20", line=dict(dash="dot")))
            for i in plot_data.index:
                if plot_data.at[i, "Setup"] == 9:
                    fig.add_trace(go.Scatter(x=[i], y=[plot_data.at[i, "Close"]], text=["9"], mode="markers+text", marker=dict(color="green", size=10)))
                if plot_data.at[i, "Countdown"] == 13:
                    fig.add_trace(go.Scatter(x=[i], y=[plot_data.at[i, "Close"]], text=["13"], mode="markers+text", marker=dict(color="red", size=10)))
            st.plotly_chart(fig, use_container_width=True)

            # RSI
            if 'RSI' in plot_data:
                st.subheader("RSI")
                rsi_data = plot_data["RSI"].astype('float64').replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
                if not rsi_data.isna().all():
                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(go.Scatter(x=plot_data.index, y=rsi_data, name="RSI"))
                    st.plotly_chart(rsi_fig, use_container_width=True)
                else:
                    st.warning("RSI data contains only NaN values")

            # MACD
            if 'MACD' in plot_data and 'SIGNAL' in plot_data:
                st.subheader("MACD")
                macd_fig = go.Figure()
                macd_data = plot_data["MACD"].astype('float64').replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
                signal_data = plot_data["SIGNAL"].astype('float64').replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
                if not (macd_data.isna().all() or signal_data.isna().all()):
                    macd_fig.add_trace(go.Scatter(x=plot_data.index, y=macd_data, name="MACD"))
                    macd_fig.add_trace(go.Scatter(x=plot_data.index, y=signal_data, name="Signal", line=dict(dash="dot")))
                    st.plotly_chart(macd_fig, use_container_width=True)
                else:
                    st.warning("MACD or Signal data contains only NaN values")

            # MFI
            if 'MFI' in plot_data:
                st.subheader("MFI")
                mfi_data = plot_data["MFI"].astype('float64').replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
                if not mfi_data.isna().all():
                    mfi_fig = go.Figure()
                    mfi_fig.add_trace(go.Scatter(x=plot_data.index, y=mfi_data, name="MFI"))
                    st.plotly_chart(mfi_fig, use_container_width=True)
                else:
                    st.warning("MFI data contains only NaN values")

            # Volume
            st.subheader("Volum og snitt")
            vfig = go.Figure()
            vfig.add_trace(go.Bar(x=plot_data.index, y=plot_data["Volume"], name="Volume"))
            vfig.add_trace(go.Scatter(x=plot_data.index, y=plot_data["VolMA"], name="VolMA"))
            st.plotly_chart(vfig, use_container_width=True)
    except Exception as e:
        st.error(f"En feil oppstod: {str(e)}")
