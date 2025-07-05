import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# RSI
def rsi(data, period=14):
    try:
        if not isinstance(data['Close'], pd.Series):
            raise ValueError("Input 'Close' column must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(data['Close']):
            raise ValueError("Column 'Close' must be numeric for RSI calculation")
        delta = data['Close'].diff()
        if not isinstance(delta, pd.Series):
            raise ValueError("Delta from diff() is not a pandas Series")
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
        # Avoid division by zero
        rs = gain / loss.where(loss != 0, np.finfo(float).eps)
        result = 100 - (100 / (1 + rs))
        # Ensure numeric and handle inf/NaN
        result = pd.to_numeric(result, errors='coerce').astype('float64')
        result = result.replace([np.inf, -np.inf], np.nan)
        if not isinstance(result, pd.Series):
            raise ValueError("RSI result is not a pandas Series")
        # Debug: Check for non-numeric values
        if result.dtype != 'float64':
            raise ValueError(f"RSI contains non-float64 values: {result.dtype}")
        return result
    except Exception as e:
        st.error(f"Error in RSI calculation: {str(e)}")
        return pd.Series(np.nan, index=data.index, dtype='float64')

# MA
def ma(data, period=20):
    try:
        if not isinstance(data['Close'], pd.Series):
            raise ValueError("Input 'Close' column must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(data['Close']):
            raise ValueError("Column 'Close' must be numeric for MA calculation")
        result = data['Close'].rolling(window=period, min_periods=1).mean()
        result = pd.to_numeric(result, errors='coerce').astype('float64')
        result = result.replace([np.inf, -np.inf], np.nan)
        if not isinstance(result, pd.Series):
            raise ValueError("MA result is not a pandas Series")
        if result.dtype != 'float64':
            raise ValueError(f"MA contains non-float64 values: {result.dtype}")
        return result
    except Exception as e:
        st.error(f"Error in MA calculation: {str(e)}")
        return pd.Series(np.nan, index=data.index, dtype='float64')

# MFI
def mfi(data, period=14):
    try:
        if not all(isinstance(data[col], pd.Series) for col in ['High', 'Low', 'Close', 'Volume']):
            raise ValueError("Input columns 'High', 'Low', 'Close', 'Volume' must be pandas Series")
        if not all(pd.api.types.is_numeric_dtype(data[col]) for col in ['High', 'Low', 'Close', 'Volume']):
            raise ValueError("Columns 'High', 'Low', 'Close', 'Volume' must be numeric for MFI calculation")
        tp = (data['High'] + data['Low'] + data['Close']) / 3
        if not isinstance(tp, pd.Series):
            raise ValueError("Typical price (tp) is not a pandas Series")
        mf = tp * data['Volume']
        pos = mf.where(tp > tp.shift(), 0).rolling(window=period, min_periods=1).sum()
        neg = mf.where(tp < tp.shift(), 0).rolling(window=period, min_periods=1).sum()
        # Avoid division by zero
        result = 100 - (100 / (1 + pos / neg.where(neg != 0, np.finfo(float).eps)))
        result = pd.to_numeric(result, errors='coerce').astype('float64')
        result = result.replace([np.inf, -np.inf], np.nan)
        if not isinstance(result, pd.Series):
            raise ValueError("MFI result is not a pandas Series")
        if result.dtype != 'float64':
            raise ValueError(f"MFI contains non-float64 values: {result.dtype}")
        return result
    except Exception as e:
        st.error(f"Error in MFI calculation: {str(e)}")
        return pd.Series(np.nan, index=data.index, dtype='float64')

# MACD
def macd(data, short=12, long=26, signal=9):
    try:
        if not isinstance(data['Close'], pd.Series):
            raise ValueError("Input 'Close' column must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(data['Close']):
            raise ValueError("Column 'Close' must be numeric for MACD calculation")
        short_ema = data['Close'].ewm(span=short, adjust=False).mean()
        long_ema = data['Close'].ewm(span=long, adjust=False).mean()
        macd_line = short_ema - long_ema
        sig_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_line = pd.to_numeric(macd_line, errors='coerce').astype('float64')
        sig_line = pd.to_numeric(sig_line, errors='coerce').astype('float64')
        macd_line = macd_line.replace([np.inf, -np.inf], np.nan)
        sig_line = sig_line.replace([np.inf, -np.inf], np.nan)
        if not (isinstance(macd_line, pd.Series) and isinstance(sig_line, pd.Series)):
            raise ValueError("MACD or Signal line is not a pandas Series")
        if macd_line.dtype != 'float64' or sig_line.dtype != 'float64':
            raise ValueError(f"MACD or Signal contains non-float64 values: MACD {macd_line.dtype}, Signal {sig_line.dtype}")
        return macd_line, sig_line
    except Exception as e:
        st.error(f"Error in MACD calculation: {str(e)}")
        return pd.Series(np.nan, index=data.index, dtype='float64'), pd.Series(np.nan, index=data.index, dtype='float64')

# DeMark 9-13
def demark(data):
    try:
        # Ensure required columns exist
        required_columns = ['Close', 'High', 'Low', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"DataFrame missing required columns: {', '.join(set(required_columns) - set(data.columns))}")
        
        # Ensure columns are numeric
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column '{col}' must be numeric")
        
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
    except Exception as e:
        st.error(f"Error in DeMark calculation: {str(e)}")
        return data

# App UI
st.title("📈 Teknisk analyse – Oslo Børs")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        # Cache data fetching to avoid repeated API calls
        @st.cache_data
        def fetch_data(ticker, period, _version=2):
            return yf.download(ticker, period=period, interval="1d", auto_adjust=False)

        data = fetch_data(ticker, period)
        if data.empty:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            # Debug: Inspect raw data
            st.write("Data columns:", data.columns)
            st.write("Data types:", data.dtypes)
            st.write("First few rows:", data.head())

            # Ensure numeric columns
            numeric_columns = ['Close', 'High', 'Low', 'Volume']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                if data[col].isna().all():
                    raise ValueError(f"Column '{col}' contains only non-numeric or missing values after conversion")
            
            # Check if Close is numeric
            if not pd.api.types.is_numeric_dtype(data['Close']):
                raise ValueError("Column 'Close' must be numeric after conversion")
            
            # Calculate technical indicators
            data = demark(data)
            data["RSI"] = rsi(data)
            data["MA20"] = ma(data)
            data["MFI"] = mfi(data)
            data["MACD"], data["SIGNAL"] = macd(data)
            data["VolMA"] = data["Volume"].rolling(window=20, min_periods=1).mean()

            # Debug: Inspect calculated columns
            st.write("Calculated columns:", data[['RSI', 'MA20', 'MFI', 'MACD', 'SIGNAL']].dtypes)
            st.write("First few rows of calculated columns:", data[['RSI', 'MA20', 'MFI', 'MACD', 'SIGNAL']].head())
            # Check for non-numeric values
            for col in ['RSI', 'MA20', 'MFI', 'MACD', 'SIGNAL']:
                if data[col].dtype != 'float64':
                    st.warning(f"Column {col} has non-float64 dtype: {data[col].dtype}")
                if data[col].isna().all():
                    st.warning(f"Column {col} contains only NaN values")
                else:
                    non_numeric = data[col][~data[col].apply(lambda x: isinstance(x, (int, float)) or pd.isna(x))]
                    if not non_numeric.empty:
                        st.warning(f"Column {col} contains non-numeric values: {non_numeric.head().to_list()}")

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
            rsi_data = data["RSI"].astype('float64').replace([np.inf, -np.inf], np.nan)
            if rsi_data.isna().all():
                st.warning("RSI data contains only NaN values")
            else:
                st.line_chart(rsi_data)

            # MACD
            st.subheader("MACD")
            macd_fig = go.Figure()
            macd_data = data["MACD"].astype('float64').replace([np.inf, -np.inf], np.nan)
            signal_data = data["SIGNAL"].astype('float64').replace([np.inf, -np.inf], np.nan)
            if macd_data.isna().all() or signal_data.isna().all():
                st.warning("MACD or Signal data contains only NaN values")
            else:
                macd_fig.add_trace(go.Scatter(x=data.index, y=macd_data, name="MACD"))
                macd_fig.add_trace(go.Scatter(x=data.index, y=signal_data, name="Signal", line=dict(dash="dot")))
                st.plotly_chart(macd_fig, use_container_width=True)

            # MFI
            st.subheader("MFI")
            mfi_data = data["MFI"].astype('float64').replace([np.inf, -np.inf], np.nan)
            if mfi_data.isna().all():
                st.warning("MFI data contains only NaN values")
            else:
                st.line_chart(mfi_data)

            # Volume
            st.subheader("Volum og snitt")
            vfig = go.Figure()
            vfig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"))
            vfig.add_trace(go.Scatter(x=data.index, y=data["VolMA"], name="VolMA"))
            st.plotly_chart(vfig, use_container_width=True)
    except Exception as e:
        st.error(f"En feil oppstod: {str(e)}")
