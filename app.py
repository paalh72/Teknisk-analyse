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
        # Start completely fresh
        cleaned_data = {}
        
        # Handle index - convert to string if datetime
        if isinstance(df.index, pd.DatetimeIndex):
            cleaned_data['Date'] = [str(d)[:10] for d in df.index]
        
        # Process each column with maximum aggression
        for col in df.columns:
            try:
                # Get raw values as numpy array
                raw_values = df[col].values
                
                # Convert everything to native Python types first
                python_values = []
                for val in raw_values:
                    try:
                        # Handle various types
                        if pd.isna(val) or val is None:
                            python_values.append(None)
                        elif hasattr(val, 'item'):  # numpy scalar
                            python_values.append(val.item())
                        else:
                            python_values.append(val)
                    except:
                        python_values.append(None)
                
                # Try to determine the best type
                non_null_values = [v for v in python_values if v is not None]
                
                if not non_null_values:
                    # All null - make it string
                    cleaned_data[col] = [''] * len(python_values)
                    continue
                
                # Check if all values can be converted to float
                try:
                    float_values = []
                    for val in python_values:
                        if val is None or pd.isna(val):
                            float_values.append(0.0)
                        else:
                            float_val = float(val)
                            # Replace inf/-inf with 0
                            if np.isinf(float_val):
                                float_values.append(0.0)
                            else:
                                float_values.append(round(float_val, 8))
                    
                    # Verify all values are finite
                    if all(isinstance(v, (int, float)) and np.isfinite(v) for v in float_values):
                        cleaned_data[col] = float_values
                        continue
                except:
                    pass
                
                # Check if all values can be converted to int
                try:
                    int_values = []
                    all_int = True
                    for val in python_values:
                        if val is None or pd.isna(val):
                            int_values.append(0)
                        else:
                            int_val = int(float(val))
                            int_values.append(int_val)
                    
                    if all_int:
                        cleaned_data[col] = int_values
                        continue
                except:
                    pass
                
                # Fall back to string
                str_values = []
                for val in python_values:
                    if val is None or pd.isna(val):
                        str_values.append('')
                    else:
                        str_values.append(str(val))
                
                cleaned_data[col] = str_values
                
            except Exception as col_error:
                # Ultimate fallback - make it all empty strings
                cleaned_data[col] = [''] * len(df)
                continue
        
        # Create new DataFrame from clean data
        if not cleaned_data:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(cleaned_data)
        
        # Force explicit dtypes
        for col in result_df.columns:
            try:
                # Check what we have
                sample_val = result_df[col].iloc[0] if len(result_df) > 0 else None
                
                if isinstance(sample_val, str):
                    result_df[col] = result_df[col].astype('object')
                elif isinstance(sample_val, int):
                    result_df[col] = result_df[col].astype('int64')
                elif isinstance(sample_val, float):
                    result_df[col] = result_df[col].astype('float64')
                else:
                    result_df[col] = result_df[col].astype('object')
            except:
                result_df[col] = result_df[col].astype('object')
        
        return result_df
        
    except Exception as e:
        st.error(f"DataFrame cleaning failed: {str(e)}")
        return pd.DataFrame()

def safe_display_dataframe(df, title="Data"):
    """
    Multiple fallback strategies for displaying DataFrames
    """
    st.write(f"**{title}** ({df.shape[0]} rows, {df.shape[1]} columns)")
    
    # Try to clean the dataframe first
    try:
        display_data = bulletproof_arrow_clean(df)
        if display_data.empty:
            st.warning("Could not clean DataFrame for display")
            return False
        
        # Strategy 1: Try st.dataframe with cleaned data
        try:
            st.dataframe(display_data, use_container_width=True)
            return True
        except Exception as e1:
            st.warning(f"st.dataframe failed: {str(e1)}")
        
        # Strategy 2: Try with limited rows
        try:
            limited_data = display_data.head(20)
            st.write("**First 20 rows:**")
            st.dataframe(limited_data, use_container_width=True)
            return True
        except Exception as e2:
            st.warning(f"Limited st.dataframe failed: {str(e2)}")
        
        # Strategy 3: Try st.table
        try:
            table_data = display_data.head(10)
            st.write("**First 10 rows (table):**")
            st.table(table_data)
            return True
        except Exception as e3:
            st.warning(f"st.table failed: {str(e3)}")
        
        # Strategy 4: Manual display
        try:
            st.write("**Manual display (last 5 rows):**")
            last_data = display_data.tail(5)
            
            # Show as simple text
            for idx, row in last_data.iterrows():
                row_text = f"Row {idx}: "
                for col in last_data.columns:
                    try:
                        val = row[col]
                        if isinstance(val, (int, float)):
                            row_text += f"{col}={val:.4f}, "
                        else:
                            row_text += f"{col}={str(val)[:20]}, "
                    except:
                        row_text += f"{col}=ERROR, "
                st.text(row_text[:-2])  # Remove last comma
            return True
        except Exception as e4:
            st.warning(f"Manual display failed: {str(e4)}")
        
    except Exception as clean_error:
        st.error(f"DataFrame cleaning failed: {str(clean_error)}")
    
    # Final fallback - show basic info
    try:
        st.write("**Basic DataFrame Information:**")
        st.write(f"Shape: {df.shape}")
        st.write(f"Columns: {list(df.columns)}")
        
        # Show last values for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("**Last values (numeric columns):**")
            for col in numeric_cols[:10]:  # Show first 10 numeric columns
                try:
                    last_val = df[col].iloc[-1]
                    if pd.notna(last_val):
                        st.write(f"- {col}: {last_val:.4f}")
                    else:
                        st.write(f"- {col}: NaN")
                except:
                    st.write(f"- {col}: Error reading value")
        
        return True
    except Exception as final_error:
        st.error(f"All display methods failed: {str(final_error)}")
        return False

# App UI
st.title("📈 Teknisk analyse – Oslo Børs")
ticker = st.text_input("Ticker (f.eks. DNB.OL)", "DNB.OL")
period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

if ticker:
    try:
        # Cache data fetching to avoid repeated API calls
        @st.cache_data(ttl=300)  # Cache for 5 minutes
        def fetch_data(ticker, period):
            try:
                data = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
                
                # Basic validation
                if data.empty:
                    return pd.DataFrame()
                
                # Clean data immediately after download
                data = data.dropna(how='all')
                
                # Ensure proper column types
                for col in data.columns:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')
                
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
                safe_display_dataframe(data, "Komplett dataset")

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
