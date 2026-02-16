"""
Gamma Scalping Strategy - Interactive Streamlit Dashboard

This application provides an interactive interface for backtesting and analyzing
the gamma scalping options trading strategy for Nifty options.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta

# Import strategy modules
from gamma_scalping_system import (
    GammaScalpingStrategy, 
    generate_sample_data,
    generate_performance_report,
    BlackScholes
)
from config import CONSERVATIVE, AGGRESSIVE, HIGH_FREQUENCY

# Import API data fetcher
try:
    from api_data_fetcher import APIDataFetcher
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    st.warning("⚠️ API data fetcher not available. Install nsepy: pip install nsepy")

# Page configuration
st.set_page_config(
    page_title="Gamma Scalping Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    /* Metric cards styling - works in both light and dark mode */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 500;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    /* Remove the metric background that was causing issues */
    div[data-testid="metric-container"] {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    h1 {
        padding-bottom: 20px;
    }
    .reportview-container .main .block-container {
        max-width: 95%;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📈 Gamma Scalping Strategy Dashboard")
st.markdown("### Interactive Backtesting & Analysis for Nifty Options")

# Sidebar - Strategy Configuration
st.sidebar.header("⚙️ Strategy Configuration")

# Strategy Presets
st.sidebar.markdown("### Quick Presets")
col1, col2, col3 = st.sidebar.columns(3)

if col1.button("Conservative", use_container_width=True):
    st.session_state.preset = 'conservative'
if col2.button("Aggressive", use_container_width=True):
    st.session_state.preset = 'aggressive'
if col3.button("High Freq", use_container_width=True):
    st.session_state.preset = 'high_frequency'

# Initialize preset values
if 'preset' not in st.session_state:
    st.session_state.preset = None

# Set default values based on preset
if st.session_state.preset == 'conservative':
    default_delta = CONSERVATIVE['DELTA_THRESHOLD']
    default_profit = CONSERVATIVE['PROFIT_TARGET']
    default_loss = CONSERVATIVE['MAX_LOSS']
    default_iv_entry = CONSERVATIVE['IV_ENTRY_PERCENTILE']
elif st.session_state.preset == 'aggressive':
    default_delta = AGGRESSIVE['DELTA_THRESHOLD']
    default_profit = AGGRESSIVE['PROFIT_TARGET']
    default_loss = AGGRESSIVE['MAX_LOSS']
    default_iv_entry = AGGRESSIVE['IV_ENTRY_PERCENTILE']
elif st.session_state.preset == 'high_frequency':
    default_delta = HIGH_FREQUENCY['DELTA_THRESHOLD']
    default_profit = HIGH_FREQUENCY['PROFIT_TARGET']
    default_loss = HIGH_FREQUENCY['MAX_LOSS']
    default_iv_entry = 30
else:
    default_delta = 0.15
    default_profit = 0.50
    default_loss = -0.30
    default_iv_entry = 30

# Strategy Parameters
st.sidebar.markdown("### Strategy Parameters")

delta_threshold = st.sidebar.slider(
    "Delta Threshold",
    min_value=0.05,
    max_value=0.30,
    value=default_delta,
    step=0.01,
    help="Rehedge when |portfolio delta| exceeds this value"
)

iv_entry_percentile = st.sidebar.slider(
    "IV Entry Percentile",
    min_value=10,
    max_value=50,
    value=int(default_iv_entry),
    step=5,
    help="Enter position when IV is below this percentile"
)

iv_exit_percentile = st.sidebar.slider(
    "IV Exit Percentile",
    min_value=50,
    max_value=90,
    value=70,
    step=5,
    help="Exit position when IV exceeds this percentile"
)

profit_target = st.sidebar.slider(
    "Profit Target (%)",
    min_value=10,
    max_value=100,
    value=int(default_profit * 100),
    step=5,
    help="Exit at this profit % of premium paid"
) / 100

max_loss = st.sidebar.slider(
    "Max Loss (%)",
    min_value=-50,
    max_value=-10,
    value=int(default_loss * 100),
    step=5,
    help="Stop loss at this % of premium paid"
) / 100

time_to_expiry = st.sidebar.slider(
    "Days to Expiry",
    min_value=2,
    max_value=21,
    value=7,
    step=1,
    help="Option time to expiry in days"
)

risk_free_rate = st.sidebar.slider(
    "Risk Free Rate (%)",
    min_value=2.0,
    max_value=10.0,
    value=6.0,
    step=0.5,
    help="Annual risk-free rate for Greeks calculation"
) / 100

# Data Configuration
st.sidebar.markdown("### Data Configuration")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    options=['1min', '3min', '5min'],
    index=2,
    help="Candlestick timeframe"
)

backtest_days = st.sidebar.slider(
    "Backtest Days",
    min_value=10,
    max_value=90,
    value=30,
    step=5,
    help="Number of days to backtest"
)

# Data source selection
data_source_options = ["Generate Sample Data", "Upload CSV"]
if API_AVAILABLE:
    data_source_options.append("Fetch from NSE API")

data_source = st.sidebar.radio(
    "Data Source",
    options=data_source_options,
    help="Choose data source for backtesting"
)

uploaded_file = None
api_start_date = None
api_end_date = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Nifty Price Data",
        type=['csv'],
        help="CSV should have: (date+time OR timestamp) and (close OR spot) columns. High/low optional."
    )
elif data_source == "Fetch from API (Yahoo Finance)":
    st.sidebar.markdown("#### API Data Range")
    
    # Date range selector
    col1, col2 = st.sidebar.columns(2)
    with col1:
        api_start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=30),
            max_value=datetime.now().date(),
            help="Start date for API data"
        )
    with col2:
        api_end_date = st.date_input(
            "End Date",
            value=datetime.now().date() - timedelta(days=1),
            max_value=datetime.now().date(),
            help="End date for API data"
        )
    
    # Show info about API
    st.sidebar.info(
        "📡 **Market Data API**\n"
        "- Using Yahoo Finance (yfinance)\n"
        "- Free, no API key required\n"
        "- Daily data (simulated to intraday)\n"
        "- More reliable than NSE scraping"
    )

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "📈 Charts", 
    "📋 Trade Log", 
    "🎯 Greeks Analysis",
    "📖 Documentation"
])

# Run Backtest Button
if st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True):
    with st.spinner("Running backtest... This may take a moment."):
        try:
            # Load or generate data
            if data_source == "Fetch from API (Yahoo Finance)":
                # Validate dates
                if api_start_date >= api_end_date:
                    st.sidebar.error("❌ Start date must be before end date")
                    st.stop()
                
                # Initialize API fetcher (use yfinance by default)
                fetcher = APIDataFetcher(use_yfinance=True)
                
                # Show progress
                progress_text = st.empty()
                progress_text.info(f"📡 Fetching data from Yahoo Finance ({api_start_date} to {api_end_date})...")
                
                try:
                    # Fetch and format data
                    price_data, metadata = fetcher.fetch_and_format(
                        start_date=api_start_date,
                        end_date=api_end_date,
                        timeframe=timeframe,
                        symbol="NIFTY 50"
                    )
                    
                    progress_text.success(
                        f"✅ Fetched {metadata['total_records']} records "
                        f"({metadata['daily_records']} daily bars converted to {timeframe})"
                    )
                    
                except Exception as e:
                    progress_text.error(f"❌ API Error: {str(e)}")
                    st.stop()
            
            elif data_source == "Upload CSV" and uploaded_file is not None:
                price_data = pd.read_csv(uploaded_file)
                
                # Clean Excel formula format (="value") from all string columns
                for col in price_data.columns:
                    if price_data[col].dtype == 'object':  # String columns
                        price_data[col] = price_data[col].astype(str).str.replace(r'^="|"$', '', regex=True)
                
                # Handle different column formats
                # Case 1: Separate 'date' and 'time' columns -> combine into 'timestamp'
                if 'date' in price_data.columns and 'time' in price_data.columns:
                    price_data['timestamp'] = pd.to_datetime(
                        price_data['date'].astype(str) + ' ' + price_data['time'].astype(str),
                        format='%d-%m-%y %H:%M:%S'
                    )
                # Case 2: Single 'timestamp' column -> parse it
                elif 'timestamp' in price_data.columns:
                    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                else:
                    raise ValueError("CSV must have either 'timestamp' column or 'date' and 'time' columns")
                
                # Handle 'spot' column (use it as 'close' if 'close' doesn't exist)
                if 'spot' in price_data.columns and 'close' not in price_data.columns:
                    price_data['close'] = price_data['spot']
                
                # Verify required columns
                required_cols = ['timestamp', 'close']
                missing_cols = [col for col in required_cols if col not in price_data.columns]
                if missing_cols:
                    raise ValueError(f"CSV missing required columns: {', '.join(missing_cols)}")
                
                # Add high/low if not present (use close as fallback)
                if 'high' not in price_data.columns:
                    price_data['high'] = price_data['close'] * 1.001  # Approximate 0.1% higher
                if 'low' not in price_data.columns:
                    price_data['low'] = price_data['close'] * 0.999   # Approximate 0.1% lower
                
                # Keep only needed columns
                price_data = price_data[['timestamp', 'close', 'high', 'low']].copy()
                
                # Sort by timestamp
                price_data = price_data.sort_values('timestamp').reset_index(drop=True)
                
                # Data validation: Remove rows with zero, negative, or NaN prices
                initial_rows = len(price_data)
                price_data = price_data[
                    (price_data['close'] > 0) & 
                    (price_data['high'] > 0) & 
                    (price_data['low'] > 0) &
                    price_data['close'].notna() &
                    price_data['high'].notna() &
                    price_data['low'].notna()
                ].reset_index(drop=True)
                
                if len(price_data) < initial_rows:
                    st.sidebar.warning(f"⚠️ Removed {initial_rows - len(price_data)} rows with invalid prices")
                
                if len(price_data) == 0:
                    st.sidebar.error("❌ No valid price data found after filtering")
                    st.stop()
                
            else:
                price_data = generate_sample_data(days=backtest_days, timeframe=timeframe)
            
            # Initialize strategy
            strategy = GammaScalpingStrategy(
                delta_threshold=delta_threshold,
                iv_entry_percentile=iv_entry_percentile,
                iv_exit_percentile=iv_exit_percentile,
                profit_target=profit_target,
                max_loss=max_loss,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate
            )
            
            # Run backtest
            backtest_results = strategy.run_backtest(price_data)
            
            # Store in session state
            st.session_state.price_data = price_data
            st.session_state.backtest_results = backtest_results
            st.session_state.trade_log = strategy.trade_log
            st.session_state.pnl_history = strategy.pnl_history
            st.session_state.strategy = strategy
            
            st.sidebar.success("✅ Backtest completed successfully!")
            
        except ZeroDivisionError as e:
            import traceback
            error_details = traceback.format_exc()
            st.sidebar.error(f"❌ Division by zero error:\n```\n{error_details}\n```")
            st.error("Division by zero error - see sidebar for details")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.sidebar.error(f"❌ Error running backtest: {str(e)}\n```\n{error_details}\n```")

# Tab 1: Dashboard
with tab1:
    if 'backtest_results' in st.session_state and not st.session_state.backtest_results.empty:
        results = st.session_state.backtest_results
        trade_log = st.session_state.trade_log
        pnl_history = st.session_state.pnl_history
        
        st.markdown("## 📊 Performance Summary")
        
        # Calculate metrics
        trades_df = pd.DataFrame([t for t in trade_log if 'EXIT' in t.get('action', '')])
        num_trades = len(trades_df)
        
        if num_trades > 0:
            winning_trades = trades_df[trades_df['total_pnl'] > 0]
            losing_trades = trades_df[trades_df['total_pnl'] <= 0]
            
            win_rate = len(winning_trades) / num_trades * 100
            total_pnl = sum(pnl_history) if pnl_history else 0
            avg_win = winning_trades['total_pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['total_pnl'].mean() if len(losing_trades) > 0 else 0
            
            # Correct profit factor: sum(profits) / abs(sum(losses))
            total_wins = winning_trades['total_pnl'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['total_pnl'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses != 0 else 0
            
            # Drawdown calculation
            cumulative_pnl = results['total_pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = drawdown.min()
            
            # Display metrics in columns
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total P&L", f"₹{total_pnl:,.0f}", 
                         delta=f"{(total_pnl/10000):.1f}% ROI" if total_pnl != 0 else None)
            
            with col2:
                st.metric("Win Rate", f"{win_rate:.1f}%",
                         delta=f"{len(winning_trades)}/{num_trades} trades")
            
            with col3:
                st.metric("Profit Factor", f"{profit_factor:.2f}",
                         delta="Good" if profit_factor > 1.5 else "Poor")
            
            with col4:
                st.metric("Avg Win", f"₹{avg_win:,.0f}",
                         delta=f"Avg Loss: ₹{avg_loss:,.0f}")
            
            with col5:
                st.metric("Max Drawdown", f"₹{max_drawdown:,.0f}",
                         delta=f"{(max_drawdown/total_pnl*100):.1f}%" if total_pnl != 0 else None,
                         delta_color="inverse")
            
            # Trade breakdown
            st.markdown("### Trade Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Exit Reasons")
                exit_reasons = trades_df['action'].value_counts()
                fig_reasons = go.Figure(data=[
                    go.Bar(
                        x=[r.replace('EXIT_STRADDLE_', '') for r in exit_reasons.index],
                        y=exit_reasons.values,
                        marker_color=['green' if 'PROFIT' in r else 'red' if 'LOSS' in r else 'blue' 
                                     for r in exit_reasons.index]
                    )
                ])
                fig_reasons.update_layout(
                    title="Exit Reasons Distribution",
                    xaxis_title="Exit Reason",
                    yaxis_title="Count",
                    height=300
                )
                st.plotly_chart(fig_reasons, use_container_width=True)
            
            with col2:
                st.markdown("#### P&L Distribution")
                fig_dist = go.Figure(data=[
                    go.Histogram(
                        x=trades_df['total_pnl'],
                        nbinsx=20,
                        marker_color='rgba(31, 119, 180, 0.7)',
                        name='P&L'
                    )
                ])
                fig_dist.add_vline(x=0, line_dash="dash", line_color="red")
                fig_dist.add_vline(x=trades_df['total_pnl'].mean(), 
                                  line_dash="dash", line_color="green",
                                  annotation_text=f"Mean: ₹{trades_df['total_pnl'].mean():.0f}")
                fig_dist.update_layout(
                    title="P&L Distribution",
                    xaxis_title="P&L (₹)",
                    yaxis_title="Frequency",
                    height=300
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Hedging statistics
            hedges = [t for t in trade_log if t.get('action') == 'REHEDGE']
            if hedges:
                st.markdown("### Hedging Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Rehedges", len(hedges))
                
                with col2:
                    avg_hedges = len(hedges) / num_trades if num_trades > 0 else 0
                    st.metric("Avg Rehedges/Trade", f"{avg_hedges:.1f}")
                
                with col3:
                    hedges_df = pd.DataFrame(hedges)
                    avg_delta = hedges_df['portfolio_delta'].abs().mean()
                    st.metric("Avg |Delta| at Rehedge", f"{avg_delta:.3f}")
        
        else:
            st.info("No trades executed in this backtest. Try adjusting entry parameters.")
    
    else:
        st.info("👈 Configure parameters in the sidebar and click 'Run Backtest' to get started!")
        
        st.markdown("""
        ### Welcome to Gamma Scalping Strategy Dashboard
        
        This application helps you backtest and analyze a gamma scalping (volatility scalping) 
        strategy for Nifty options.
        
        **Quick Start:**
        1. Use default parameters or select a preset (Conservative/Aggressive/High Frequency)
        2. Adjust parameters in the sidebar as needed
        3. Click "Run Backtest" to execute the strategy
        4. Explore results across different tabs
        
        **Key Features:**
        - Interactive parameter tuning
        - Real-time backtesting
        - Comprehensive performance metrics
        - Detailed trade analysis
        - Greeks visualization
        - CSV export functionality
        """)

# Tab 2: Charts
with tab2:
    if 'backtest_results' in st.session_state and not st.session_state.backtest_results.empty:
        results = st.session_state.backtest_results
        price_data = st.session_state.price_data
        trade_log = st.session_state.trade_log
        
        st.markdown("## 📈 Interactive Charts")
        
        # Price chart with trades
        st.markdown("### Price Chart with Entry/Exit Points")
        
        fig_price = go.Figure()
        
        # Price line
        fig_price.add_trace(go.Scatter(
            x=price_data['timestamp'],
            y=price_data['close'],
            mode='lines',
            name='Nifty Spot',
            line=dict(color='rgba(31, 119, 180, 0.8)', width=2)
        ))
        
        # Entry points
        entries = [t for t in trade_log if t['action'] == 'ENTER_STRADDLE']
        if entries:
            entry_times = [t['timestamp'] for t in entries]
            entry_prices = [t['spot'] for t in entries]
            fig_price.add_trace(go.Scatter(
                x=entry_times,
                y=entry_prices,
                mode='markers',
                name='Enter Straddle',
                marker=dict(size=12, color='green', symbol='triangle-up')
            ))
        
        # Exit points
        exits = [t for t in trade_log if 'EXIT' in t['action']]
        if exits:
            exit_times = [t['timestamp'] for t in exits]
            exit_prices = [t['spot'] for t in exits]
            exit_colors = ['red' if 'STOP_LOSS' in t['action'] else 'blue' for t in exits]
            fig_price.add_trace(go.Scatter(
                x=exit_times,
                y=exit_prices,
                mode='markers',
                name='Exit Position',
                marker=dict(size=12, color=exit_colors, symbol='triangle-down')
            ))
        
        fig_price.update_layout(
            title="Nifty Price with Trade Markers",
            xaxis_title="Time",
            yaxis_title="Price (₹)",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Cumulative P&L with Drawdown
        st.markdown("### Cumulative P&L and Drawdown")
        
        cumulative_pnl = results['total_pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        
        fig_pnl = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Cumulative P&L", "Drawdown"),
            row_heights=[0.7, 0.3]
        )
        
        # Cumulative P&L
        fig_pnl.add_trace(
            go.Scatter(
                x=results['timestamp'],
                y=cumulative_pnl,
                name='Cumulative P&L',
                line=dict(color='green', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 0, 0.1)'
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig_pnl.add_trace(
            go.Scatter(
                x=results['timestamp'],
                y=drawdown,
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ),
            row=2, col=1
        )
        
        fig_pnl.update_xaxes(title_text="Time", row=2, col=1)
        fig_pnl.update_yaxes(title_text="P&L (₹)", row=1, col=1)
        fig_pnl.update_yaxes(title_text="Drawdown (₹)", row=2, col=1)
        fig_pnl.update_layout(height=600, hovermode='x unified')
        
        st.plotly_chart(fig_pnl, use_container_width=True)
        
        # Portfolio Delta
        st.markdown("### Portfolio Delta Control")
        
        fig_delta = go.Figure()
        
        fig_delta.add_trace(go.Scatter(
            x=results['timestamp'],
            y=results['portfolio_delta'],
            name='Portfolio Delta',
            line=dict(color='purple', width=2)
        ))
        
        fig_delta.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig_delta.add_hline(y=delta_threshold, line_dash="dash", line_color="red", 
                           annotation_text="Rehedge Threshold")
        fig_delta.add_hline(y=-delta_threshold, line_dash="dash", line_color="red")
        
        # Add shaded region for neutral zone
        fig_delta.add_hrect(
            y0=-delta_threshold, y1=delta_threshold,
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
        )
        
        fig_delta.update_layout(
            title="Portfolio Delta (Target: Delta-Neutral)",
            xaxis_title="Time",
            yaxis_title="Delta",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_delta, use_container_width=True)
        
    else:
        st.info("Run a backtest to view charts")

# Tab 3: Trade Log
with tab3:
    if 'trade_log' in st.session_state and st.session_state.trade_log:
        st.markdown("## 📋 Detailed Trade Log")
        
        trade_log_df = pd.DataFrame(st.session_state.trade_log)
        
        # Format display
        if 'timestamp' in trade_log_df.columns:
            trade_log_df['timestamp'] = pd.to_datetime(trade_log_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Display trade log
        st.dataframe(
            trade_log_df,
            use_container_width=True,
            height=600
        )
        
        # Download button
        csv = trade_log_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade Log CSV",
            data=csv,
            file_name=f"gamma_scalping_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.info("Run a backtest to view trade log")

# Tab 4: Greeks Analysis
with tab4:
    if 'backtest_results' in st.session_state and not st.session_state.backtest_results.empty:
        results = st.session_state.backtest_results
        
        st.markdown("## 🎯 Greeks Analysis")
        
        # Gamma vs Theta
        st.markdown("### Gamma (Edge) vs Theta (Cost)")
        
        fig_greeks = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Gamma & Vega (Positive Exposure)", "Theta (Time Decay Cost)")
        )
        
        # Gamma and Vega
        fig_greeks.add_trace(
            go.Scatter(
                x=results['timestamp'],
                y=results['gamma'],
                name='Gamma',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
        
        fig_greeks.add_trace(
            go.Scatter(
                x=results['timestamp'],
                y=results['vega'],
                name='Vega',
                line=dict(color='purple', width=2)
            ),
            row=1, col=1
        )
        
        # Theta
        fig_greeks.add_trace(
            go.Scatter(
                x=results['timestamp'],
                y=results['theta'],
                name='Theta',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ),
            row=2, col=1
        )
        
        fig_greeks.update_xaxes(title_text="Time", row=2, col=1)
        fig_greeks.update_yaxes(title_text="Gamma / Vega", row=1, col=1)
        fig_greeks.update_yaxes(title_text="Theta (Daily Decay)", row=2, col=1)
        fig_greeks.update_layout(height=700, hovermode='x unified')
        
        st.plotly_chart(fig_greeks, use_container_width=True)
        
        # Cumulative Theta vs Total P&L
        st.markdown("### Cumulative Theta Decay vs Total P&L")
        
        fig_theta_pnl = go.Figure()
        
        fig_theta_pnl.add_trace(go.Scatter(
            x=results['timestamp'],
            y=results['theta'].cumsum(),
            name='Cumulative Theta (Cost)',
            line=dict(color='red', width=2)
        ))
        
        fig_theta_pnl.add_trace(go.Scatter(
            x=results['timestamp'],
            y=results['total_pnl'].cumsum(),
            name='Total P&L',
            line=dict(color='green', width=2)
        ))
        
        fig_theta_pnl.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig_theta_pnl.update_layout(
            title="Theta Decay vs P&L (Must overcome theta to profit)",
            xaxis_title="Time",
            yaxis_title="Cumulative (₹)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_theta_pnl, use_container_width=True)
        
        # P&L Components
        st.markdown("### P&L Components Breakdown")
        
        trades_df = pd.DataFrame([t for t in st.session_state.trade_log if 'EXIT' in t.get('action', '')])
        
        if len(trades_df) > 0 and 'options_pnl' in trades_df.columns:
            total_options_pnl = trades_df['options_pnl'].sum()
            total_futures_pnl = trades_df['futures_pnl'].sum() if 'futures_pnl' in trades_df else 0
            total_hedging_cost = trades_df['hedging_cost'].sum() if 'hedging_cost' in trades_df else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Options P&L", f"₹{total_options_pnl:,.0f}")
            
            with col2:
                st.metric("Futures P&L (FIFO)", f"₹{total_futures_pnl:,.0f}")
            
            with col3:
                st.metric("Hedging Cost (legacy)", f"₹{total_hedging_cost:,.0f}",
                         help="Legacy field - not used. FIFO accounting already includes all hedge economics")
            
            # Pie chart (only options and futures, no hedging_cost as it's double counting)
            fig_components = go.Figure(data=[go.Pie(
                labels=['Options P&L', 'Futures P&L'],
                values=[abs(total_options_pnl), abs(total_futures_pnl)],
                hole=0.3
            )])
            
            
            fig_components.update_layout(
                title="P&L Components (FIFO Accounting)",
                height=400
            )
            
            st.plotly_chart(fig_components, use_container_width=True)
        
        # 🎯 GREEKS P&L ATTRIBUTION (Professional Analysis)
        if 'gamma_pnl' in results.columns:
            st.markdown("### 🎯 Greeks P&L Attribution (Skill vs Luck)")
            
            # Calculate cumulative attribution
            cum_gamma_pnl = results['gamma_pnl'].sum()
            cum_vega_pnl = results['vega_pnl'].sum()
            cum_theta_pnl = results['theta_pnl'].sum()
            
            # Display attribution metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Gamma P&L (Skill)", f"₹{cum_gamma_pnl:,.0f}",
                         help="Profits from price movement scalping - repeatable skill")
            
            with col2:
                st.metric("Vega P&L (Luck)", f"₹{cum_vega_pnl:,.0f}",
                         help="Profits from IV changes - market luck, not scalable")
            
            with col3:
                st.metric("Theta P&L (Cost)", f"₹{cum_theta_pnl:,.0f}",
                         help="Time decay cost - always negative")
            
            # Attribution analysis
            total_options = cum_gamma_pnl + cum_vega_pnl + cum_theta_pnl
            if abs(total_options) > 0:
                gamma_pct = (cum_gamma_pnl / abs(total_options)) * 100
                vega_pct = (cum_vega_pnl / abs(total_options)) * 100
                theta_pct = (cum_theta_pnl / abs(total_options)) * 100
                
                if gamma_pct > 60:
                    st.success(f"✅ **Skill-Based**: {gamma_pct:.1f}% from gamma scalping - strategy is working!")
                elif vega_pct > 50:
                    st.warning(f"⚠️ **Luck-Based**: {vega_pct:.1f}% from vega - not repeatable/scalable")
                else:
                    st.info(f"📊 Mixed: Gamma {gamma_pct:.1f}% | Vega {vega_pct:.1f}% | Theta {theta_pct:.1f}%")
            
            # Waterfall chart showing attribution
            fig_waterfall = go.Figure(go.Waterfall(
                name="P&L Attribution",
                orientation="v",
                measure=["relative", "relative", "relative", "total"],
                x=["Gamma<br>(Scalping)", "Vega<br>(IV Change)", "Theta<br>(Decay)", "Total<br>Options P&L"],
                y=[cum_gamma_pnl, cum_vega_pnl, cum_theta_pnl, 0],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "red"}},
                increasing={"marker": {"color": "green"}},
                totals={"marker": {"color": "blue"}}
            ))
            
            fig_waterfall.update_layout(
                title="Options P&L Attribution Waterfall",
                yaxis_title="P&L (₹)",
                height=400
            )
            
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
    else:
        st.info("Run a backtest to view Greeks analysis")

# Tab 5: Documentation
with tab5:
    st.markdown("""
    ## 📖 Gamma Scalping Strategy Documentation
    
    ### What is Gamma Scalping?
    
    Gamma scalping (also called volatility scalping) is a market-neutral options strategy that:
    1. **Buys a straddle** (ATM call + ATM put) to gain positive gamma exposure
    2. **Delta hedges** by trading futures to maintain delta neutrality
    3. **Profits from volatility** when realized volatility exceeds implied volatility
    
    ### Core Principle
    
    **Profit = Realized Volatility - Implied Volatility - Transaction Costs - Theta Decay**
    
    ### Strategy Flow
    
    #### Entry
    - Enter when IV is relatively low (below specified percentile)
    - Buy ATM straddle (call + put at same strike)
    - Initial position is approximately delta-neutral
    
    #### Delta Hedging
    - Monitor portfolio delta continuously
    - When |delta| exceeds threshold:
        - **Positive delta**: Sell futures to neutralize
        - **Negative delta**: Buy futures to neutralize
    - This "scalps gamma" by buying low and selling high automatically
    
    #### Exit
    - Profit target hit (e.g., 50% of premium paid)
    - Stop loss hit (e.g., -30% of premium paid)
    - IV spikes (volatility already realized)
    - Time decay too high
    
    ### Parameter Guide
    
    #### Delta Threshold (0.05 - 0.30)
    - **Lower (0.08-0.12)**: More frequent hedging, more opportunities, higher costs
    - **Higher (0.15-0.20)**: Less hedging, lower costs, more directional risk
    - **Recommended**: 0.15 as starting point
    
    #### IV Entry Percentile (10 - 50)
    - **Lower (20-30)**: More selective, only cheap volatility
    - **Higher (35-45)**: More trades, pay more for volatility
    - **Recommended**: 25-35 for Nifty
    
    #### Profit Target (10% - 100%)
    - **Conservative (30-40%)**: Quick profits, higher win rate
    - **Aggressive (50-70%)**: Let winners run, lower win rate
    - **Recommended**: 40-50% for weekly options
    
    #### Max Loss (-50% - -10%)
    - **Tight (-20%)**: Limits losses, may exit early
    - **Wide (-40%)**: Gives room, higher risk
    - **Recommended**: -25% to -35%
    
    #### Days to Expiry (2 - 21)
    - **Short (2-5)**: Higher gamma, higher theta, more active
    - **Long (7-14)**: Lower gamma, lower theta, more stable
    - **Recommended**: 5-7 days for good gamma/theta ratio
    
    ### Understanding the Greeks
    
    - **Delta (Δ)**: Directional exposure - aim to keep near zero
    - **Gamma (Γ)**: Your edge - higher gamma = more scalping opportunities
    - **Theta (Θ)**: Your cost - daily time decay to overcome
    - **Vega (ν)**: IV sensitivity - profit if IV increases after entry
    
    ### When Strategy Works Best
    
    ✅ **Favorable Conditions:**
    - IV is low but about to realize
    - Choppy, range-bound markets
    - Around events (earnings, policy decisions)
    - Good gamma/theta ratio
    
    ❌ **Unfavorable Conditions:**
    - IV already high (expensive options)
    - Strong trending markets
    - Very low volatility (insufficient movement)
    - High transaction costs
    
    ### Risk Warnings
    
    ⚠️ **Key Risks:**
    - **Gap Risk**: Overnight gaps can breach hedges
    - **Theta Decay**: Time works against you daily
    - **Transaction Costs**: Over-hedging destroys profits
    - **Model Risk**: Black-Scholes assumptions may not hold
    - **Execution Risk**: Slippage in fast markets
    
    ### Tips for Success
    
    1. **Start Conservative**: Use conservative preset initially
    2. **Monitor Closely**: Watch real-time delta and gamma
    3. **Control Costs**: Optimize delta threshold for your cost structure
    4. **Test Thoroughly**: Backtest across different market conditions
    5. **Paper Trade First**: Never start with live money
    6. **Keep Records**: Track actual vs expected performance
    7. **Adjust Parameters**: Fine-tune based on results
    
    ### References
    
    📚 **Recommended Reading:**
    - "Dynamic Hedging" by Nassim Taleb
    - "Option Volatility and Pricing" by Sheldon Natenberg
    - "Volatility Trading" by Euan Sinclair
    
    ---
    
    **Disclaimer**: This is educational software for learning gamma scalping concepts. 
    Options trading involves substantial risk and is not suitable for all investors. 
    Test thoroughly with paper trading before risking real capital.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Gamma Scalping Strategy Dashboard v1.0</p>
        <p>For educational purposes only. Trade at your own risk.</p>
    </div>
    """, unsafe_allow_html=True)
