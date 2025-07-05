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
        
        # Create a copy to avoid modifying original
        result_data = data.copy()
        
        # Initialize with explicit numpy arrays, then convert to Series
        result_data["C4"] = pd.Series(data["Close"].shift(4).values, index=data.index, dtype='float64')
        result_data["C2"] = pd.Series(data["Close"].shift(2).values, index=data.index, dtype='float64')
        result_data["Setup"] = pd.Series(np.zeros(len(data), dtype='int64'), index=data.index, dtype='int64')
        result_data["Countdown"] = pd.Series(np.zeros(len(data), dtype='int64'), index=data.index, dtype='int64')
        
        # Convert to numpy arrays for faster processing
        close_vals = result_data["Close"].values
        c4_vals = result_data["C4"].values
        c2_vals = result_data["C2"].values
        setup_vals = np.zeros(len(data), dtype='int64')
        countdown_vals = np.zeros(len(data), dtype='int64')
        
        # Setup calculation
        count = 0
        for i in range(len(data)):
            if i < 4:  # Skip rows where C4 is NaN
                continue
            if not (np.isnan(close_vals[i]) or np.isnan(c4_vals[i])):
                if close_vals[i] > c4_vals[i]:
                    count += 1
                else:
                    count = 0
                setup_vals[i] = count
        
        # Countdown calculation
        cd = 0
        started = False
        for i in range(len(data)):
            if setup_vals[i] == 9:
                started = True
                cd = 0
            if started and i >= 2:
                if not (np.isnan(close_vals[i]) or np.isnan(c2_vals[i])):
                    if close_vals[i] > c2_vals[i]:
                        cd += 1
                    countdown_vals[i] = cd
                    if cd == 13:
                        started = False
        
        # Assign back to DataFrame
        result_data["Setup"] = pd.Series(setup_vals, index=data.index, dtype='int64')
        result_data["Countdown"] = pd.Series(countdown_vals, index=data.index, dtype='int64')
        
        return result_data
    except Exception as e:
        st.error(f"Error in DeMark calculation: {str(e)}")
        return data

def bulletproof_arrow_clean(df):
    """
    Ultra-aggressive DataFrame cleaning for Arrow compatibility
    """
    try:
        # Create completely new DataFrame with simple range index
        cleaned_df = pd.DataFrame(index=range(len(df)))
        
        # Handle datetime index separately
        if hasattr(df.index, 'to_pydatetime'):
            try:
                cleaned_df['Date'] = pd.to_datetime(df.index).strftime('%Y-%m-%d')
            except:
                cleaned_df['Date'] = [f"Row_{i}" for i in range(len(df))]
        
        # Process each column with extreme care
        for col in df.columns:
            try:
                series = df[col]
                
                # Skip empty columns
                if len(series) == 0:
                    continue
                
                # Get raw values
                raw_values = series.values
                
                # Handle different data types
                if pd.api.types.is_datetime64_any_dtype(series):
                    # Convert datetime to string
                    try:
                        cleaned_df[col] = pd.to_datetime(raw_values).strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        cleaned_df[col] = [str(x) for x in raw_values]
                        
                elif pd.api.types.is_numeric_dtype(series):
                    # Handle numeric data
                    try:
                        # Convert to float64, handling all edge cases
                        numeric_values = pd.to_numeric(raw_values, errors='coerce')
                        
                        # Replace infinite values
                        numeric_values = numeric_values.replace([np.inf, -np.inf], np.nan)
                        
                        # Fill NaN values
                        numeric_values = numeric_values.fillna(0.0)
                        
                        # Ensure it's proper float64
                        cleaned_df[col] = np.array(numeric_values, dtype=np.float64)
                        
                    except:
                        # Fallback: convert to string
                        cleaned_df[col] = [str(x) for x in raw_values]
                        
                else:
                    # Everything else becomes string
                    try:
                        cleaned_df[col] = [str(x) if x is not None else '' for x in raw_values]
                    except:
                        cleaned_df[col] = [''] * len(raw_values)
                        
            except Exception as col_error:
                # Skip problematic columns
                st.warning(f"Skipping column {col}: {str(col_error)}")
                continue
        
        # Final validation and type enforcement
        for col in list(cleaned_df.columns):
            try:
                # Check if column is problematic for Arrow
                test_series = cleaned_df[col]
                
                # Try to convert to Arrow-compatible type
                if test_series.dtype == 'object':
                    # Try numeric conversion first
                    try:
                        numeric_test = pd.to_numeric(test_series, errors='coerce')
                        if numeric_test.notna().sum() > len(test_series) * 0.5:  # If >50% are numbers
                            cleaned_df[col] = numeric_test.fillna(0.0).astype(np.float64)
                        else:
                            # Keep as string but ensure all values are strings
                            cleaned_df[col] = test_series.astype(str)
                    except:
                        cleaned_df[col] = test_series.astype(str)
                        
                elif test_series.dtype not in ['float64', 'int64', 'bool', 'datetime64[ns]']:
                    # Convert to float64 if possible, otherwise string
                    try:
                        cleaned_df[col] = pd.to_numeric(test_series, errors='coerce').fillna(0.0).astype(np.float64)
                    except:
                        cleaned_df[col] = test_series.astype(str)
                        
            except Exception as final_error:
                # Remove problematic columns entirely
                st.warning(f"Removing column {col}: {str(final_error)}")
                cleaned_df = cleaned_df.drop(columns=[col])
        
        # Ensure we have some data
        if cleaned_df.empty:
            st.error("No data could be processed for display")
            return pd.DataFrame({'Error': ['No displayable data']})
        
        return cleaned_df
        
    except Exception as e:
        st.error(f"Critical error in data cleaning: {str(e)}")
        return pd.DataFrame({'Error': [str(e)]})

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
                # Use more conservative yfinance settings
                data = yf.download(
                    ticker, 
                    period=period, 
                    interval="1d", 
                    auto_adjust=False,
                    prepost=False,
                    threads=False  # Disable threading for stability
                )
                
                # Basic validation
                if data.empty:
                    return pd.DataFrame()
                
                # Clean data immediately after download
                data = data.dropna(how='all')  # Remove completely empty rows
                
                # Ensure proper column types from the start
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                for col in numeric_columns:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')
                
                # Remove any rows with all NaN values in essential columns
                essential_cols = ['Close', 'High', 'Low', 'Volume']
                available_essential = [col for col in essential_cols if col in data.columns]
                if available_essential:
                    data = data.dropna(subset=available_essential, how='all')
                
                return data
                
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                return pd.DataFrame()

        raw_data = fetch_data(ticker, period)
        if raw_data.empty:
            st.warning(f"Fant ikke data for ticker: {ticker}")
        else:
            # Work with a copy
            data = raw_data.copy()
            
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
                st.write(f"**Dataframe info:** {data.shape[0]} rader, {data.shape[1]} kolonner")
                
                try:
                    # Clean the data for display
                    display_data = bulletproof_arrow_clean(data)
                    
                    # Try different display methods
                    if not display_data.empty:
                        st.write("**Komplett dataset:**")
                        
                        # Method 1: Try st.dataframe with container width
                        try:
                            st.dataframe(display_data, use_container_width=True, height=400)
                        except Exception as e1:
                            st.warning(f"st.dataframe failed: {str(e1)[:100]}...")
                            
                            # Method 2: Try showing limited rows
                            try:
                                st.write("**Siste 20 rader:**")
                                limited_data = display_data.tail(20)
                                st.dataframe(limited_data, use_container_width=True)
                            except Exception as e2:
                                st.warning(f"Limited dataframe failed: {str(e2)[:100]}...")
                                
                                # Method 3: Show as table
                                try:
                                    st.write("**Tabell format (siste 10 rader):**")
                                    table_data = display_data.tail(10)
                                    st.table(table_data)
                                except Exception as e3:
                                    st.warning(f"Table display failed: {str(e3)[:100]}...")
                                    
                                    # Method 4: Manual display
                                    st.write("**Grunnleggende informasjon:**")
                                    st.write(f"- Antall rader: {len(display_data)}")
                                    st.write(f"- Antall kolonner: {len(display_data.columns)}")
                                    st.write(f"- Kolonner: {list(display_data.columns)}")
                                    
                                    # Show some sample data
                                    st.write("**Eksempel data (siste rader):**")
                                    for i, (idx, row) in enumerate(display_data.tail(5).iterrows()):
                                        if i < 3:  # Show only first 3 rows
                                            st.write(f"Rad {idx}: {dict(row)}")
                    else:
                        st.error("Ingen data kunne vises")
                        
                except Exception as main_error:
                    st.error(f"Kunne ikke vise data: {str(main_error)}")
                    
                    # Fallback: Show basic info
                    st.write("**Debug informasjon:**")
                    st.write(f"- DataFrame shape: {data.shape}")
                    st.write(f"- Columns: {list(data.columns)}")
                    st.write(f"- Data types: {dict(data.dtypes)}")
                    st.write(f"- Memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

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
            
            # Show Python/Streamlit version info
            import sys
            st.write(f"Python version: {sys.version}")
            st.write(f"Streamlit version: {st.__version__}")
            
            # Show available memory
            try:
                import psutil
                memory = psutil.virtual_memory()
                st.write(f"Available memory: {memory.available / 1024 / 1024 / 1024:.2f} GB")
            except:
                st.write("Could not get memory info")
