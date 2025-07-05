import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from alpha_vantage.timeseries import TimeSeries

ALPHA_VANTAGE_API_KEY = 'YOUR_API_KEY'  # Replace with your Alpha Vantage API key

# RSI
def rsi(close, period=14):
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Input 'Close' must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError("Column 'Close' must be numeric")
        if close.isna().all():
            raise ValueError("Column 'Close' contains only NaN values")
        delta = close.diff()
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
def ma(close, period=20):
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Input 'Close' must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError("Column 'Close' must be numeric")
        if close.isna().all():
            raise ValueError("Column 'Close' contains only NaN values")
        result = close.rolling(window=period, min_periods=1).mean()
        result = pd.to_numeric(result, errors='coerce').astype('float64')
        result = result.replace([np.inf, -np.inf], np.nan)
        if result.isna().all():
            raise ValueError("MA contains only NaN values")
        return result
    except Exception as e:
        st.error(f"Error in MA calculation: {str(e)}")
        return None

# MFI
def mfi(high, low, close, volume, period=14):
    try:
        if not all(isinstance(col, pd.Series) for col in [high, low, close, volume]):
            raise ValueError("Input columns must be pandas Series")
        if not all(pd.api.types.is_numeric_dtype(col) for col in [high, low, close, volume]):
            raise ValueError("Input columns must be numeric")
        if any(col.isna().all() for col in [high, low, close, volume]):
            raise ValueError("One or more input columns contain only NaN values")
        tp = (high + low + close) / 3
        mf = tp * volume
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
def macd(close, short=12, long=26, signal=9):
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Input 'Close' must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError("Column 'Close' must be numeric")
        if close.isna().all():
            raise ValueError("Column 'Close' contains only NaN values")
        short_ema = close.ewm(span=short, adjust=False).mean()
        long_ema = close.ewm(span=long, adjust=False).mean()
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
def demark(close, index):
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Input 'Close' must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError("Column 'Close' must be numeric")
        if close.isna().all():
            raise ValueError("Column 'Close' contains only NaN values")
        setup = pd.Series(0, index=index, dtype='int32')
        countdown = pd.Series(0, index=index, dtype='int32')
        c4 = close.shift(4)
        c2 = close.shift(2)
        count = 0
        for i in range(len(close)):
            if i < 4:
                continue
            close_val = close.iloc[i]
            c4_val = c4.iloc[i]
            if pd.isna(close_val) or pd.isna(c4_val):
                count = 0
            elif not (isinstance(close_val, (int, float)) and isinstance(c4_val, (int, float))):
                count = 0
            elif close_val > c4_val:
                count += 1
            else:
                count = 0
            setup.iloc[i] = count
        cd = 0
        started = False
        for i in range(len(close)):
            if setup.iloc[i] == 9:
                started = True
                cd = 0
            if started and i >= 2:
                close_val = close.iloc[i]
                c2_val = c2.iloc[i]
                if pd.isna(close_val) or pd.isna(c2_val):
                    cd = 0
                elif not (isinstance(close_val, (int, float)) and isinstance(c2_val, (int, float))):
                    cd = 0
                elif close_val > c2_val:
                    cd += 1
                countdown.iloc[i] = cd
                if cd == 13:
                    started = False
        return setup, countdown
    except Exception as e:
        st.error(f"Error in DeMark calculation: {str(e)}")
        return None, None

# App UI
st.title("📈 Teknisk analyse – Oslo Børs")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        @st.cache_data
        def fetch_data(ticker, period, _version=10):
            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
            outputsize = 'compact' if period == '3mo' else 'full'
            data, meta = ts.get_daily(symbol=ticker, outputsize=outputsize)
            data = data.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
            data.index = pd.to_datetime(data.index, utc=True).tz_convert(None)  # Clean datetime index
            data = data.sort_index()
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')
                if data[col].isna().all():
                    raise ValueError(f"Column '{col}' contains only non-numeric or missing values")
            return data

        data = fetch_data(ticker, period)
        if data.empty:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            index = data.index
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']

            # Calculate indicators
            setup, countdown = demark(close, index)
            rsi_data = rsi(close)
            ma_data = ma(close)
            mfi_data = mfi(high, low, close, volume)
            macd_data, signal_data = macd(close)
            vol_ma = volume.rolling(window=20, min_periods=1).mean()

            # Pris + DeMark
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=index, y=close, name="Close"))
            if ma_data is not None:
                fig.add_trace(go.Scatter(x=index, y=ma_data, name="MA20", line=dict(dash="dot")))
            if setup is not None and countdown is not None:
                for i in range(len(index)):
                    if setup.iloc[i] == 9:
                        fig.add_trace(go.Scatter(x=[index[i]], y=[close.iloc[i]], text=["9"], mode="markers+text", marker=dict(color="green", size=10)))
                    if countdown.iloc[i] == 13:
                        fig.add_trace(go.Scatter(x=[index[i]], y=[close.iloc[i]], text=["13"], mode="markers+text", marker=dict(color="red", size=10)))
            st.plotly_chart(fig, use_container_width=True)

            # RSI
            if rsi_data is not None:
                st.subheader("RSI")
                rsi_data = rsi_data.astype('float64').replace([np.inf, -np.inf], np.nan)
                if not rsi_data.isna().all():
                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(go.Scatter(x=index, y=rsi_data, name="RSI"))
                    st.plotly_chart(rsi_fig, use_container_width=True)
                else:
                    st.warning("RSI data contains only NaN values")

            # MACD
            if macd_data is not None and signal_data is not None:
                st.subheader("MACD")
                macd_data = macd_data.astype('float64').replace([np.inf, -np.inf], np.nan)
                signal_data = signal_data.astype('float64').replace([np.inf, -np.inf], np.nan)
                if not (macd_data.isna().all() or signal_data.isna().all()):
                    macd_fig = go.Figure()
                    macd_fig.add_trace(go.Scatter(x=index, y=macd_data, name="MACD"))
                    macd_fig.add_trace(go.Scatter(x=index, y=signal_data, name="Signal", line=dict(dash="dot")))
                    st.plotly_chart(macd_fig, use_container_width=True)
                else:
                    st.warning("MACD or Signal data contains only NaN values")

            # MFI
            if mfi_data is not None:
                st.subheader("MFI")
                mfi_data = mfi_data.astype('float64').replace([np.inf, -np.inf], np.nan)
                if not mfi_data.isna().all():
                    mfi_fig = go.Figure()
                    mfi_fig.add_trace(go.Scatter(x=index, y=mfi_data, name="MFI"))
                    st.plotly_chart(mfi_fig, use_container_width=True)
                else:
                    st.warning("MFI data contains only NaN values")

            # Volume
            st.subheader("Volum og snitt")
            vfig = go.Figure()
            vfig.add_trace(go.Bar(x=index, y=volume, name="Volume"))
            vfig.add_trace(go.Scatter(x=index, y=vol_ma, name="VolMA"))
            st.plotly_chart(vfig, use_container_width=True)
    except Exception as e:
        st.error(f"En feil oppstod: {str(e)}")
