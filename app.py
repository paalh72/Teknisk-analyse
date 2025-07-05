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
        rs = gain / loss
        result = 100 - (100 / (1 + rs))
        if not isinstance(result, pd.Series):
            raise ValueError("RSI result is not a pandas Series")
        return result.astype('float64')
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
        if not isinstance(result, pd.Series):
            raise ValueError("MA result is not a pandas Series")
        return result.astype('float64')
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
        # Handle division by zero
        neg = neg.replace(0, np.nan)
        result = 100 - (100 / (1 + pos / neg))
        result = result.fillna(50)  # Fill NaN with neutral value
        if not isinstance(result, pd.Series):
            raise ValueError("MFI result is not a pandas Series")
        return result.astype('float64')
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
        if not (isinstance(macd_line, pd.Series) and isinstance(sig_line, pd.Series)):
            raise ValueError("MACD or Signal line is not a pandas Series")
        return macd_line.astype('float64'), sig_line.astype('float64')
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
        
        # Initialize columns with proper dtypes
        data = data.copy()
        # CRITICAL FIX: Ensure shifted columns are float64, not object
        data["C4"] = data["Close"].shift(4).astype('float64')
        data["C2"] = data["Close"].shift(2).astype('float64')
        data["Setup"] = pd.Series(0, index=data.index, dtype='int64')
        data["Countdown"] = pd.Series(0, index=data.index, dtype='int64')
        
        count = 0
        for i in range(len(data)):
            if i < 4:  # Skip rows where C4 is NaN
                continue
            try:
                close_val = data["Close"].iloc[i]
                c4_val = data["C4"].iloc[i]
                if pd.isna(close_val) or pd.isna(c4_val):
                    count = 0
                elif close_val > c4_val:
                    count += 1
                else:
                    count = 0
                data.iloc[i, data.columns.get_loc("Setup")] = count
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
                    elif close_val > c2_val:
                        cd += 1
                    data.iloc[i, data.columns.get_loc("Countdown")] = cd
                    if cd == 13:
                        started = False
                except Exception as e:
                    st.warning(f"Error processing Countdown at index {i}: {str(e)}")
                    cd = 0
        
        # Ensure all columns maintain proper dtypes
        data["Setup"] = data["Setup"].astype('int64')
        data["Countdown"] = data["Countdown"].astype('int64')
        data["C4"] = data["C4"].astype('float64')
        data["C2"] = data["C2"].astype('float64')
        
        return data
    except Exception as e:
        st.error(f"Error in DeMark calculation: {str(e)}")
        return data

def clean_dataframe_for_arrow(df):
    """
    Aggressive DataFrame cleaning specifically for Arrow/Streamlit compatibility
    """
    df = df.copy()
    
    # Step 1: Replace all infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Step 2: Handle each column individually
    for col in df.columns:
        original_dtype = df[col].dtype
        
        # Handle datetime columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue  # Keep datetime as is
        
        # Handle numeric columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Convert to standard float64 or int64
            if original_dtype in ['float16', 'float32'] or 'Float' in str(original_dtype):
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            elif original_dtype in ['int8', 'int16', 'int32'] or 'Int' in str(original_dtype):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        
        # Handle object columns
        else:
            # Try to convert to numeric first
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            
            # If more than 80% of values are valid numbers, treat as numeric
            if numeric_series.notna().sum() > len(df) * 0.8:
                df[col] = numeric_series.astype('float64')
            else:
                # Convert to standard string
                df[col] = df[col].astype(str)
    
    # Step 3: Specific column type enforcement
    float_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 
                  'C4', 'C2', 'RSI', 'MA20', 'MFI', 'MACD', 'SIGNAL', 'VolMA']
    int_cols = ['Setup', 'Countdown']
    
    # Enforce float columns
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    
    # Enforce integer columns
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
    
    # Step 4: Final safety check - no problematic dtypes should remain
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if dtype_str == 'object':
            df[col] = df[col].astype(str)
        elif 'Float' in dtype_str or 'Int' in dtype_str:
            # These are pandas nullable dtypes that can cause Arrow issues
            if 'Float' in dtype_str:
                df[col] = df[col].astype('float64')
            else:
                df[col] = df[col].astype('int64')
    
    # Step 5: Round floating point numbers to avoid precision issues
    float_columns = df.select_dtypes(include=['float64']).columns
    df[float_columns] = df[float_columns].round(8)
    
    return df

# App UI
st.title("📈 Teknisk analyse – Oslo Børs")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        # Cache data fetching to avoid repeated API calls
        @st.cache_data
        def fetch_data(ticker, period):
            try:
                data = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
                
                # Basic validation
                if data.empty:
                    return pd.DataFrame()
                
                # Reset index to make Date a column
                data = data.reset_index()
                
                # Ensure all columns are properly typed from the start
                for col in data.columns:
                    if col == 'Date':
                        continue  # Keep Date as datetime
                    elif data[col].dtype == 'object':
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')
                    elif 'float' in str(data[col].dtype):
                        data[col] = data[col].astype('float64')
                    elif 'int' in str(data[col].dtype):
                        data[col] = data[col].astype('int64')
                
                # Set Date back as index
                if 'Date' in data.columns:
                    data = data.set_index('Date')
                
                return data
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                return pd.DataFrame()

        raw_data = fetch_data(ticker, period)
        if raw_data.empty:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            # Clean initial data
            data = clean_dataframe_for_arrow(raw_data)
            
            # Validate essential columns
            essential_columns = ['Close', 'High', 'Low', 'Volume']
            for col in essential_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing essential column: {col}")
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(f"Column '{col}' is not numeric")
                if data[col].isna().all():
                    raise ValueError(f"Column '{col}' contains only missing values")
            
            # Calculate technical indicators
            data = demark(data)
            data["RSI"] = rsi(data)
            data["MA20"] = ma(data)
            data["MFI"] = mfi(data)
            data["MACD"], data["SIGNAL"] = macd(data)
            data["VolMA"] = data["Volume"].rolling(window=20, min_periods=1).mean().astype('float64')

            # Final cleanup for Arrow compatibility
            data = clean_dataframe_for_arrow(data)
            
            # Pris + DeMark
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", showlegend=True))
            fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA20", line=dict(dash="dot"), showlegend=True))
            
            # Add DeMark signals
            setup_9_indices = data[data["Setup"] == 9].index
            countdown_13_indices = data[data["Countdown"] == 13].index
            
            if len(setup_9_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=setup_9_indices, 
                    y=data.loc[setup_9_indices, "Close"], 
                    text=["9"] * len(setup_9_indices), 
                    mode="markers+text", 
                    marker=dict(color="green", size=10),
                    name="Setup 9",
                    showlegend=True
                ))
            
            if len(countdown_13_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=countdown_13_indices, 
                    y=data.loc[countdown_13_indices, "Close"], 
                    text=["13"] * len(countdown_13_indices), 
                    mode="markers+text", 
                    marker=dict(color="red", size=10),
                    name="Countdown 13",
                    showlegend=True
                ))
            
            fig.update_layout(title=f"{ticker} - Pris og DeMark signaler")
            st.plotly_chart(fig, use_container_width=True)

            # RSI
            st.subheader("RSI")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI"))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overkjøpt (70)")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversolgt (30)")
            rsi_fig.update_layout(title="RSI", yaxis_range=[0, 100])
            st.plotly_chart(rsi_fig, use_container_width=True)

            # MACD
            st.subheader("MACD")
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"))
            macd_fig.add_trace(go.Scatter(x=data.index, y=data["SIGNAL"], name="Signal", line=dict(dash="dot")))
            macd_fig.add_hline(y=0, line_dash="dash", line_color="gray")
            macd_fig.update_layout(title="MACD")
            st.plotly_chart(macd_fig, use_container_width=True)

            # MFI
            st.subheader("MFI (Money Flow Index)")
            mfi_fig = go.Figure()
            mfi_fig.add_trace(go.Scatter(x=data.index, y=data["MFI"], name="MFI"))
            mfi_fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overkjøpt (80)")
            mfi_fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversolgt (20)")
            mfi_fig.update_layout(title="MFI", yaxis_range=[0, 100])
            st.plotly_chart(mfi_fig, use_container_width=True)

            # Volume
            st.subheader("Volum og snitt")
            vfig = go.Figure()
            vfig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"))
            vfig.add_trace(go.Scatter(x=data.index, y=data["VolMA"], name="VolMA", line=dict(color="orange")))
            vfig.update_layout(title="Volum")
            st.plotly_chart(vfig, use_container_width=True)

            # Display summary statistics
            st.subheader("Siste verdier")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Pris", f"{data['Close'].iloc[-1]:.2f}")
            with col2:
                st.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
            with col3:
                st.metric("MFI", f"{data['MFI'].iloc[-1]:.1f}")
            with col4:
                st.metric("MACD", f"{data['MACD'].iloc[-1]:.4f}")
            
            # Show dataframe with improved error handling
            if st.checkbox("Vis rådata"):
                try:
                    # Create a safe display version
                    display_data = clean_dataframe_for_arrow(data)
                    
                    # Additional safety for display
                    display_data = display_data.fillna(0)  # Replace any remaining NaN
                    
                    # Show basic info first
                    st.write(f"**Dataframe info:** {display_data.shape[0]} rader, {display_data.shape[1]} kolonner")
                    
                    # Try to display - use st.table as fallback if st.dataframe fails
                    try:
                        st.dataframe(display_data, use_container_width=True)
                    except Exception as df_error:
                        st.warning(f"st.dataframe failed ({str(df_error)}), using st.table instead")
                        # Show only last 20 rows if using st.table to avoid performance issues
                        st.table(display_data.tail(20))
                        
                except Exception as e:
                    st.error(f"Kunne ikke vise dataframe: {str(e)}")
                    st.info("Viser grunnleggende statistikk i stedet:")
                    st.write(f"**Shape:** {data.shape}")
                    st.write(f"**Columns:** {list(data.columns)}")
                    st.write("**Data types:**")
                    for col, dtype in data.dtypes.items():
                        st.write(f"  - {col}: {dtype}")

    except Exception as e:
        st.error(f"En feil oppstod: {str(e)}")
        st.error("Prøv å oppfriske siden eller velg en annen ticker.")
        
        # Debug information
        if st.checkbox("Vis debug info"):
            st.write("**Debug informasjon:**")
            st.write(f"Ticker: {ticker}")
            st.write(f"Period: {period}")
            st.write(f"Error type: {type(e).__name__}")
            st.write(f"Error message: {str(e)}")
