import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# DeMark 9-13 (NumPy)
def demark(close):
    try:
        if np.isnan(close).all():
            raise ValueError("Close data contains only NaN values")
        setup = np.zeros(len(close), dtype=np.int64)
        countdown = np.zeros(len(close), dtype=np.int64)
        c4 = np.roll(close, 4)
        c2 = np.roll(close, 2)
        count = 0
        for i in range(4, len(close)):
            if np.isnan(close[i]) or np.isnan(c4[i]):
                count = 0
            elif close[i] > c4[i]:
                count += 1
            else:
                count = 0
            setup[i] = count
        cd = 0
        started = False
        for i in range(2, len(close)):
            if setup[i] == 9:
                started = True
                cd = 0
            if started:
                if np.isnan(close[i]) or np.isnan(c2[i]):
                    cd = 0
                elif close[i] > c2[i]:
                    cd += 1
                countdown[i] = cd
                if cd == 13:
                    started = False
        return setup, countdown
    except Exception as e:
        st.error(f"Error in DeMark calculation: {str(e)}")
        return np.zeros(len(close), dtype=np.int64), np.zeros(len(close), dtype=np.int64)

# App UI
st.title("📈 Teknisk analyse – Oslo Børs (DeMark)")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        def fetch_data(ticker, period):
            data = yf.download(ticker, period=period, interval="1d", auto_adjust=False, prepost=False, threads=False)
            if data.empty or 'Close' not in data.columns:
                return None, None
            data = data.dropna(how='all')
            dates = np.array([d.to_pydatetime() for d in pd.to_datetime(data.index, utc=True).tz_convert(None)])
            close = data['Close'].to_numpy(dtype=np.float64)
            if np.isnan(close).all():
                raise ValueError("Close data contains only NaN values")
            return dates, close

        dates, close = fetch_data(ticker, period)
        if dates is None:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            # Calculate DeMark
            setup, countdown = demark(close)

            # Plot Close price with DeMark signals
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(dates, close, label="Close", color="#1f77b4")
            setup_9 = dates[setup == 9]
            countdown_13 = dates[countdown == 13]
            if len(setup_9) > 0:
                ax.scatter(setup_9, close[setup == 9], color="green", s=50, label="Setup 9")
                for x, y in zip(setup_9, close[setup == 9]):
                    ax.text(x, y, "9", fontsize=8, verticalalignment="bottom")
            if len(countdown_13) > 0:
                ax.scatter(countdown_13, close[countdown == 13], color="red", s=50, label="Countdown 13")
                for x, y in zip(countdown_13, close[countdown == 13]):
                    ax.text(x, y, "13", fontsize=8, verticalalignment="bottom")
            ax.set_title(f"{ticker} - Pris og DeMark signaler")
            ax.set_xlabel("Dato")
            ax.set_ylabel("Pris")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Summary statistic
            st.subheader("Siste verdi")
            st.metric("Pris", f"{close[-1]:.2f}" if not np.isnan(close[-1]) else "N/A")

    except Exception as e:
        st.error(f"En feil oppstod: {str(e)}")
