import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import yfinance as yf # Importerer yfinance for å hente data fra Yahoo Finance

# RSI (Relative Strength Index)
def rsi(close, period=14):
    """
    Kalkulerer Relative Strength Index (RSI).

    Parametere:
    close (pd.Series): En Pandas Series med lukkekurser.
    period (int): Perioden for RSI-beregningen (standard er 14).

    Returnerer:
    pd.Series: En Pandas Series med RSI-verdier, eller None hvis en feil oppstår.
    """
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Input 'close' må være en pandas Series.")
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError("Kolonnen 'close' må være numerisk.")
        if close.isna().all():
            raise ValueError("Kolonnen 'close' inneholder kun NaN-verdier.")

        delta = close.diff()
        # Beregn gjennomsnittlig gevinst og tap
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()

        # Unngå divisjon med null for RS
        rs = gain / loss.where(loss != 0, np.finfo(float).eps)
        result = 100 - (100 / (1 + rs))

        # Konverter til numerisk og håndter uendelige verdier
        result = pd.to_numeric(result, errors='coerce').astype('float64')
        result = result.replace([np.inf, -np.inf], np.nan)

        if result.isna().all():
            raise ValueError("RSI inneholder kun NaN-verdier etter beregning.")
        return result
    except Exception as e:
        st.error(f"Feil i RSI-beregning: {str(e)}")
        return None

# MA (Moving Average)
def ma(close, period=20):
    """
    Kalkulerer enkelt glidende gjennomsnitt (SMA).

    Parametere:
    close (pd.Series): En Pandas Series med lukkekurser.
    period (int): Perioden for MA-beregningen (standard er 20).

    Returnerer:
    pd.Series: En Pandas Series med MA-verdier, eller None hvis en feil oppstår.
    """
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Input 'close' må være en pandas Series.")
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError("Kolonnen 'close' må være numerisk.")
        if close.isna().all():
            raise ValueError("Kolonnen 'close' inneholder kun NaN-verdier.")

        result = close.rolling(window=period, min_periods=1).mean()

        result = pd.to_numeric(result, errors='coerce').astype('float64')
        result = result.replace([np.inf, -np.inf], np.nan)

        if result.isna().all():
            raise ValueError("MA inneholder kun NaN-verdier etter beregning.")
        return result
    except Exception as e:
        st.error(f"Feil i MA-beregning: {str(e)}")
        return None

# MFI (Money Flow Index)
def mfi(high, low, close, volume, period=14):
    """
    Kalkulerer Money Flow Index (MFI).

    Parametere:
    high (pd.Series): En Pandas Series med høyeste kurser.
    low (pd.Series): En Pandas Series med laveste kurser.
    close (pd.Series): En Pandas Series med lukkekurser.
    volume (pd.Series): En Pandas Series med volum.
    period (int): Perioden for MFI-beregningen (standard er 14).

    Returnerer:
    pd.Series: En Pandas Series med MFI-verdier, eller None hvis en feil oppstår.
    """
    try:
        if not all(isinstance(col, pd.Series) for col in [high, low, close, volume]):
            raise ValueError("Alle input-kolonner må være pandas Series.")
        if not all(pd.api.types.is_numeric_dtype(col) for col in [high, low, close, volume]):
            raise ValueError("Alle input-kolonner må være numeriske.")
        if any(col.isna().all() for col in [high, low, close, volume]):
            raise ValueError("En eller flere input-kolonner inneholder kun NaN-verdier.")

        tp = (high + low + close) / 3  # Typical Price
        mf = tp * volume  # Money Flow

        # Positiv og negativ pengeflyt
        pos = mf.where(tp > tp.shift(), 0).rolling(window=period, min_periods=1).sum()
        neg = mf.where(tp < tp.shift(), 0).rolling(window=period, min_periods=1).sum()

        # Unngå divisjon med null
        result = 100 - (100 / (1 + pos / neg.where(neg != 0, np.finfo(float).eps)))

        result = pd.to_numeric(result, errors='coerce').astype('float64')
        result = result.replace([np.inf, -np.inf], np.nan)

        if result.isna().all():
            raise ValueError("MFI inneholder kun NaN-verdier etter beregning.")
        return result
    except Exception as e:
        st.error(f"Feil i MFI-beregning: {str(e)}")
        return None

# MACD (Moving Average Convergence Divergence)
def macd(close, short=12, long=26, signal=9):
    """
    Kalkulerer Moving Average Convergence Divergence (MACD).

    Parametere:
    close (pd.Series): En Pandas Series med lukkekurser.
    short (int): Perioden for den korte EMA (standard er 12).
    long (int): Perioden for den lange EMA (standard er 26).
    signal (int): Perioden for signallinjen (standard er 9).

    Returnerer:
    Tuple[pd.Series, pd.Series]: MACD-linjen og signallinjen, eller (None, None) hvis en feil oppstår.
    """
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Input 'close' må være en pandas Series.")
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError("Kolonnen 'close' må være numerisk.")
        if close.isna().all():
            raise ValueError("Kolonnen 'close' inneholder kun NaN-verdier.")

        short_ema = close.ewm(span=short, adjust=False).mean()
        long_ema = close.ewm(span=long, adjust=False).mean()
        macd_line = short_ema - long_ema
        sig_line = macd_line.ewm(span=signal, adjust=False).mean()

        macd_line = pd.to_numeric(macd_line, errors='coerce').astype('float64')
        sig_line = pd.to_numeric(sig_line, errors='coerce').astype('float64')
        macd_line = macd_line.replace([np.inf, -np.inf], np.nan)
        sig_line = sig_line.replace([np.inf, -np.inf], np.nan)

        if macd_line.isna().all() or sig_line.isna().all():
            raise ValueError("MACD-linjen eller Signallinjen inneholder kun NaN-verdier etter beregning.")
        return macd_line, sig_line
    except Exception as e:
        st.error(f"Feil i MACD-beregning: {str(e)}")
        return None, None

# DeMark 9-13 Sekvensiell
def demark(close, index):
    """
    Kalkulerer DeMark Setup (9) og Countdown (13) sekvenser.

    Parametere:
    close (pd.Series): En Pandas Series med lukkekurser.
    index (pd.Index): Indeksen for tidsserien.

    Returnerer:
    Tuple[pd.Series, pd.Series]: Setup- og Countdown-seriene, eller (None, None) hvis en feil oppstår.
    """
    try:
        if not isinstance(close, pd.Series):
            raise ValueError("Input 'close' må være en pandas Series.")
        if not pd.api.types.is_numeric_dtype(close):
            raise ValueError("Kolonnen 'close' må være numerisk.")
        if close.isna().all():
            raise ValueError("Kolonnen 'close' inneholder kun NaN-verdier.")

        setup = pd.Series(0, index=index, dtype='int32')
        countdown = pd.Series(0, index=index, dtype='int32')

        # Shifted close prices for DeMark Setup
        c4 = close.shift(4)

        count = 0
        for i in range(len(close)):
            if i < 4: # Trenger minst 4 tidligere datapunkter for setup
                continue
            close_val = close.iloc[i]
            c4_val = c4.iloc[i]

            # Sjekk for NaN og ikke-numeriske verdier
            if pd.isna(close_val) or pd.isna(c4_val):
                count = 0
            elif not (isinstance(close_val, (int, float)) and isinstance(c4_val, (int, float))):
                count = 0
            elif close_val > c4_val:
                count += 1
            else:
                count = 0
            setup.iloc[i] = count

        # Shifted close prices for DeMark Countdown
        c2 = close.shift(2)

        cd = 0
        started = False # Flag for å indikere om countdown har startet (etter setup 9)
        for i in range(len(close)):
            if setup.iloc[i] == 9:
                started = True
                cd = 0 # Reset countdown når en setup 9 er nådd
            
            if started and i >= 2: # Trenger minst 2 tidligere datapunkter for countdown
                close_val = close.iloc[i]
                c2_val = c2.iloc[i]

                # Sjekk for NaN og ikke-numeriske verdier
                if pd.isna(close_val) or pd.isna(c2_val):
                    cd = 0
                elif not (isinstance(close_val, (int, float)) and isinstance(c2_val, (int, float))):
                    cd = 0
                elif close_val > c2_val:
                    cd += 1
                countdown.iloc[i] = cd
                
                if cd == 13:
                    started = False # Reset started flag etter countdown 13

        return setup, countdown
    except Exception as e:
        st.error(f"Feil i DeMark-beregning: {str(e)}")
        return None, None

# Streamlit App UI
st.title("📈 Teknisk analyse – Oslo Børs")

# Input for ticker og periode
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        @st.cache_data(ttl=3600) # Cache data for 1 time (3600 sekunder)
        def fetch_data(ticker_symbol, data_period, _version=14): # Øk _version hvis logikken i denne funksjonen endres
            """
            Henter daglig aksjedata fra Yahoo Finance.

            Parametere:
            ticker_symbol (str): Ticker-symbolet.
            data_period (str): Perioden for data ('3mo', '6mo', '1y', '2y').
            _version (int): Versjonsnummer for cache-kontroll.

            Returnerer:
            pd.DataFrame: DataFrame med aksjedata.
            """
            # Hent data fra Yahoo Finance
            data = yf.download(ticker_symbol, period=data_period)
            
            if data.empty:
                return pd.DataFrame() # Return empty DataFrame if no data

            # Sørg for at indeksen er DatetimeIndex og tidssone-naiv
            data.index = pd.to_datetime(data.index)
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            data = data.sort_index()

            # Velg kun de relevante kolonnene og sørg for numerisk type
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Sjekk om alle nødvendige kolonner eksisterer
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Følgende nødvendige kolonner ble ikke funnet i dataen fra Yahoo Finance: {', '.join(missing_cols)}. Sjekk ticker-symbolet.")

            # Lag en kopi av de valgte kolonnene for å unngå SettingWithCopyWarning
            data = data[required_cols].copy() 

            for col in required_cols:
                # Konverter til numerisk, og tving feil til NaN
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Etter å ha tvunget til numerisk, sørg for at dtype er float64.
                # Hvis en kolonne kun inneholdt NaN, kan pd.to_numeric la den være som 'object'.
                # Konverter eksplisitt til float64.
                data[col] = data[col].astype('float64')

                if data[col].isna().all() and len(data) > 0: # Sjekk om alle verdier er NaN for ikke-tomme data
                    raise ValueError(f"Kolonnen '{col}' inneholder kun ikke-numeriske eller manglende verdier etter konvertering.")
            
            return data

        # Hent data
        data = fetch_data(ticker, period)

        if data.empty:
            st.warning(f"Fant ikke data for ticker: {ticker}. Sjekk ticker-symbolet. Det kan hende at Yahoo Finance ikke har data for dette symbolet eller perioden.")
        else:
            # Forbered data for indikatorberegninger
            index = data.index
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']

            # Beregn indikatorer
            setup, countdown = demark(close, index)
            rsi_data = rsi(close)
            ma_data = ma(close)
            mfi_data = mfi(high, low, close, volume)
            macd_data, signal_data = macd(close)
            vol_ma = volume.rolling(window=20, min_periods=1).mean()

            # --- Plotting av grafer ---

            # Pris + MA + DeMark
            st.subheader("Prisutvikling med MA20 og DeMark-signaler")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=index, y=close, name="Lukkekurs", mode='lines', line=dict(color='blue')))
            if ma_data is not None:
                fig.add_trace(go.Scatter(x=index, y=ma_data, name="MA20", mode='lines', line=dict(dash="dot", color='orange')))
            
            # Legg til DeMark 9- og 13-signaler
            if setup is not None and countdown is not None:
                setup_9_indices = setup[setup == 9].index
                countdown_13_indices = countdown[countdown == 13].index

                for i in setup_9_indices:
                    fig.add_trace(go.Scatter(x=[i], y=[close.loc[i]], text=["9"], mode="markers+text",
                                             marker=dict(color="green", size=10, symbol='triangle-up'),
                                             textposition="top center", name="DeMark 9 Buy"))
                for i in countdown_13_indices:
                    fig.add_trace(go.Scatter(x=[i], y=[close.loc[i]], text=["13"], mode="markers+text",
                                             marker=dict(color="red", size=10, symbol='triangle-down'),
                                             textposition="bottom center", name="DeMark 13 Sell"))
            
            fig.update_layout(hovermode="x unified", legend_title_text="Indikatorer")
            st.plotly_chart(fig, use_container_width=True)

            # RSI
            if rsi_data is not None:
                st.subheader("RSI (Relative Strength Index)")
                rsi_data = rsi_data.astype('float64').replace([np.inf, -np.inf], np.nan)
                if not rsi_data.isna().all():
                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(go.Scatter(x=index, y=rsi_data, name="RSI", line=dict(color='purple')))
                    rsi_fig.add_hline(y=70, annotation_text="Overkjøpt (70)", annotation_position="top right", line_dash="dot", line_color="red")
                    rsi_fig.add_hline(y=30, annotation_text="Oversolgt (30)", annotation_position="bottom right", line_dash="dot", line_color="green")
                    rsi_fig.update_layout(hovermode="x unified")
                    st.plotly_chart(rsi_fig, use_container_width=True)
                else:
                    st.warning("RSI-data inneholder kun NaN-verdier og kan ikke plottes.")

            # MACD
            if macd_data is not None and signal_data is not None:
                st.subheader("MACD (Moving Average Convergence Divergence)")
                macd_data = macd_data.astype('float64').replace([np.inf, -np.inf], np.nan)
                signal_data = signal_data.astype('float64').replace([np.inf, -np.inf], np.nan)
                if not (macd_data.isna().all() or signal_data.isna().all()):
                    macd_fig = go.Figure()
                    macd_fig.add_trace(go.Scatter(x=index, y=macd_data, name="MACD-linje", line=dict(color='blue')))
                    macd_fig.add_trace(go.Scatter(x=index, y=signal_data, name="Signallinje", line=dict(dash="dot", color='red')))
                    macd_fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    macd_fig.update_layout(hovermode="x unified", legend_title_text="MACD")
                    st.plotly_chart(macd_fig, use_container_width=True)
                else:
                    st.warning("MACD- eller Signallinje-data inneholder kun NaN-verdier og kan ikke plottes.")

            # MFI
            if mfi_data is not None:
                st.subheader("MFI (Money Flow Index)")
                mfi_data = mfi_data.astype('float64').replace([np.inf, -np.inf], np.nan)
                if not mfi_data.isna().all():
                    mfi_fig = go.Figure()
                    mfi_fig.add_trace(go.Scatter(x=index, y=mfi_data, name="MFI", line=dict(color='teal')))
                    mfi_fig.add_hline(y=80, annotation_text="Overkjøpt (80)", annotation_position="top right", line_dash="dot", line_color="red")
                    mfi_fig.add_hline(y=20, annotation_text="Oversolgt (20)", annotation_position="bottom right", line_dash="dot", line_color="green")
                    mfi_fig.update_layout(hovermode="x unified")
                    st.plotly_chart(mfi_fig, use_container_width=True)
                else:
                    st.warning("MFI-data inneholder kun NaN-verdier og kan ikke plottes.")

            # Volume
            st.subheader("Volum og Volum Glidende Gjennomsnitt")
            vfig = go.Figure()
            vfig.add_trace(go.Bar(x=index, y=volume, name="Volum", marker_color='gray', opacity=0.7))
            vfig.add_trace(go.Scatter(x=index, y=vol_ma, name="Volum MA20", line=dict(color='darkblue', dash='dot')))
            vfig.update_layout(hovermode="x unified", legend_title_text="Volum")
            st.plotly_chart(vfig, use_container_width=True)

    except Exception as e:
        st.error(f"En uventet feil oppstod: {str(e)}")
        st.info("Vennligst sjekk at ticker-symbolet er korrekt. Yahoo Finance bruker ofte '.OL' for Oslo Børs, f.eks. 'DNB.OL'.")
