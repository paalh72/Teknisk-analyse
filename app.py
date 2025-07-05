import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# RSI
def rsi(close, period=14):
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Input 'Close' must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError("Column 'Close' must be numeric for RSI calculation")
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.finfo(float).eps)
        result = 100 - (100 / (1 + rs))
        return result.astype('float64')
    except Exception as e:
        st.error(f"Error in RSI calculation: {str(e)}")
        return pd.Series(np.nan, index=close.index, dtype='float64')

# MA
def ma(close, period=20):
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Input 'Close' must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError("Column 'Close' must be numeric for MA calculation")
        result = close.rolling(window=period, min_periods=1).mean()
        return result.astype('float64')
    except Exception as e:
        st.error(f"Error in MA calculation: {str(e)}")
        return pd.Series(np.nan, index=close.index, dtype='float64')

# MFI
def mfi(high, low, close, volume, period=14):
    try:
        if not all(isinstance(col, pd.Series) for col in [high, low, close, volume]):
            raise ValueError("Input columns must be pandas Series")
        if not all(pd.api.types.is_numeric_dtype(col) for col in [high, low, close, volume]):
            raise ValueError("Columns must be numeric for MFI calculation")
        tp = (high + low + close) / 3
        mf = tp * volume
        pos = mf.where(tp > tp.shift(), 0).rolling(window=period, min_periods=1).sum()
        neg = mf.where(tp < tp.shift(), 0).rolling(window=period, min_periods=1).sum()
        result = 100 - (100 / (1 + pos / neg.replace(0, np.finfo(float).eps)))
        return result.astype('float64')
    except Exception as e:
        st.error(f"Error in MFI calculation: {str(e)}")
        return pd.Series(np.nan, index=close.index, dtype='float64')

# MACD
def macd(close, short=12, long=26, signal=9):
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Input 'Close' must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError("Column 'Close' must be numeric for MACD calculation")
        short_ema = close.ewm(span=short, adjust=False).mean()
        long_ema = close.ewm(span=long, adjust=False).mean()
        macd_line = short_ema - long_ema
        sig_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line.astype('float64'), sig_line.astype('float64')
    except Exception as e:
        st.error(f"Error in MACD calculation: {str(e)}")
        return pd.Series(np.nan, index=close.index, dtype='float64'), pd.Series(np.nan, index=close.index, dtype='float64')

# DeMark 9-13
def demark(close, index):
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Input 'Close' must be a pandas Series")
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError("Column 'Close' must be numeric")
        setup = pd.Series(0, index=index, dtype='int64')
        countdown = pd.Series(0, index=index, dtype='int64')
        c4 = close.shift(4)
        c2 = close.shift(2)
        count = 0
        for i in range(len(close)):
            if i < 4:
                continue
            if pd.isna(close.iloc[i]) or pd.isna(c4.iloc[i]):
                count = 0
            elif close.iloc[i] > c4.iloc[i]:
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
                if pd.isna(close.iloc[i]) or pd.isna(c2.iloc[i]):
                    cd = 0
                elif close.iloc[i] > c2.iloc[i]:
                    cd += 1
                countdown.iloc[i] = cd
                if cd == 13:
                    started = False
        return setup, countdown
    except Exception as e:
        st.error(f"Error in DeMark calculation: {str(e)}")
        return pd.Series(np.nan, index=index, dtype='int64'), pd.Series(np.nan, index=index, dtype='int64')

# App UI
st.title("📈 Teknisk analyse – Oslo Børs")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        def fetch_data(ticker, period):
            data = yf.download(ticker, period=period, interval="1d", auto_adjust=False, prepost=False, threads=False)
            if data.empty:
                return pd.DataFrame()
            data = data.dropna(how='all')
            data.index = pd.to_datetime(data.index, utc=True).tz_convert(None)
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')
            return data

        data = fetch_data(ticker, period)
        if data.empty:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            # Validate essential columns
            essential_columns = ['Close', 'High', 'Low', 'Volume']
            for col in essential_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing essential column: {col}")
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(f"Column '{col}' is not numeric")
                if data[col].isna().all():
                    raise ValueError(f"Column '{col}' contains only missing values")

            # Calculate indicators
            index = data.index.to_pydatetime()
            close = data['Close'].to_numpy()
            high = data['High'].to_numpy()
            low = data['Low'].to_numpy()
            volume = data['Volume'].to_numpy()
            setup, countdown = demark(pd.Series(data['Close'], index=data.index), data.index)
            rsi_data = rsi(pd.Series(data['Close'], index=data.index))
            ma_data = ma(pd.Series(data['Close'], index=data.index))
            mfi_data = mfi(pd.Series(data['High'], index=data.index), 
                          pd.Series(data['Low'], index=data.index), 
                          pd.Series(data['Close'], index=data.index), 
                          pd.Series(data['Volume'], index=data.index))
            macd_data, signal_data = macd(pd.Series(data['Close'], index=data.index))
            vol_ma = pd.Series(data['Volume'], index=data.index).rolling(window=20, min_periods=1).mean().to_numpy()

            # Pris + DeMark
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(index, close, label="Close", color="#1f77b4")
            if ma_data is not None:
                ax.plot(index, ma_data.to_numpy(), label="MA20", linestyle="--", color="#ff7f0e")
            if setup is not None and countdown is not None:
                setup_9 = setup[setup == 9].index.to_pydatetime()
                countdown_13 = countdown[countdown == 13].index.to_pydatetime()
                if len(setup_9) > 0:
                    ax.scatter(setup_9, close[setup.index.isin(setup_9)], 
                              color="green", s=50, label="Setup 9")
                    for x, y in zip(setup_9, close[setup.index.isin(setup_9)]):
                        ax.text(x, y, "9", fontsize=8, verticalalignment="bottom")
                if len(countdown_13) > 0:
                    ax.scatter(countdown_13, close[countdown.index.isin(countdown_13)], 
                              color="red", s=50, label="Countdown 13")
                    for x, y in zip(countdown_13, close[countdown.index.isin(countdown_13)]):
                        ax.text(x, y, "13", fontsize=8, verticalalignment="bottom")
            ax.set_title(f"{ticker} - Pris og DeMark signaler")
            ax.set_xlabel("Dato")
            ax.set_ylabel("Pris")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # RSI
            if rsi_data is not None:
                st.subheader("RSI")
                rsi_data = rsi_data.to_numpy()
                if not np.isnan(rsi_data).all():
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(index, rsi_data, label="RSI", color="#1f77b4")
                    ax.axhline(y=70, linestyle="--", color="red", label="Overkjøpt (70)")
                    ax.axhline(y=30, linestyle="--", color="green", label="Oversolgt (30)")
                    ax.set_title("RSI")
                    ax.set_xlabel("Dato")
                    ax.set_ylabel("RSI")
                    ax.set_ylim(0, 100)
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("RSI data contains only NaN values")

            # MACD
            if macd_data is not None and signal_data is not None:
                st.subheader("MACD")
                macd_data = macd_data.to_numpy()
                signal_data = signal_data.to_numpy()
                if not (np.isnan(macd_data).all() or np.isnan(signal_data).all()):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(index, macd_data, label="MACD", color="#1f77b4")
                    ax.plot(index, signal_data, label="Signal", linestyle="--", color="#ff7f0e")
                    ax.axhline(y=0, linestyle="--", color="gray")
                    ax.set_title("MACD")
                    ax.set_xlabel("Dato")
                    ax.set_ylabel("MACD")
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("MACD or Signal data contains only NaN values")

            # MFI
            if mfi_data is not None:
                st.subheader("MFI (Money Flow Index)")
                mfi_data = mfi_data.to_numpy()
                if not np.isnan(mfi_data).all():
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(index, mfi_data, label="MFI", color="#1f77b4")
                    ax.axhline(y=80, linestyle="--", color="red", label="Overkjøpt (80)")
                    ax.axhline(y=20, linestyle="--", color="green", label="Oversolgt (20)")
                    ax.set_title("MFI")
                    ax.set_xlabel("Dato")
                    ax.set_ylabel("MFI")
                    ax.set_ylim(0, 100)
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("MFI data contains only NaN values")

            # Volume
            st.subheader("Volum og snitt")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(index, volume, label="Volume", color="#1f77b4", alpha=0.5)
            ax.plot(index, vol_ma, label="VolMA", color="orange")
            ax.set_title("Volum")
            ax.set_xlabel("Dato")
            ax.set_ylabel("Volum")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Summary statistics
            st.subheader("Siste verdier")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pris", f"{close[-1]:.2f}" if not np.isnan(close[-1]) else "N/A")
            with col2:
                st.metric("RSI", f"{rsi_data[-1]:.1f}" if rsi_data is not None and not np.isnan(rsi_data[-1]) else "N/A")
            with col3:
                st.metric("MFI", f"{mfi_data[-1]:.1f}" if mfi_data is not None and not np.isnan(mfi_data[-1]) else "N/A")
            with col4:
                st.metric("MACD", f"{macd_data[-1]:.4f}" if macd_data is not None and not np.isnan(macd_data[-1]) else "N/A")

    except Exception as e:
        st.error(f"En feil oppstod: {str(e)}")
