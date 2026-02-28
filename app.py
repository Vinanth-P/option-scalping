import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest_engine import (
    build_dataframes,
    generate_synthetic_data,
    run_parameter_sweep,
    run_2d_sweep,
    format_inr,
    DEFAULT_FEE, DEFAULT_K, DEFAULT_COOLDOWN,
    DEFAULT_MAX_HEDGES, DEFAULT_OPEN_FILTER, DEFAULT_IV_CRUSH,
)

st.set_page_config(
    page_title="Nifty50 Gamma Scalp 1Min",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    header[data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding-top: 0rem !important; margin-top: 0rem !important; }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem !important; }
    .main { padding: 0rem 1rem; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 0.9rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; font-weight: 600; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 12px 16px;
    }
    .metric-card { border-radius: 8px; padding: 14px 18px; margin-bottom: 8px; }
    .metric-card .metric-label { font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-bottom: 4px; }
    .metric-card .metric-value { font-size: 1.6rem; font-weight: 700; color: #FAFAFA; }
    .metric-green { border: 1px solid rgba(0, 200, 83, 0.4); background: rgba(0, 200, 83, 0.08); }
    .metric-red { border: 1px solid rgba(255, 23, 68, 0.4); background: rgba(255, 23, 68, 0.08); }
</style>
""", unsafe_allow_html=True)

def colored_metric(label, value, color="green"):
    css_class = "metric-green" if color == "green" else "metric-red"
    st.markdown(f"""
    <div class="metric-card {css_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### ⚡ Nifty50 Gamma Scalp 1Min")

# Sidebar
st.sidebar.header("📂 Data & Filters", help="1-lot Nifty margin requirement (~₹1.75L)")

uploaded_file = st.sidebar.file_uploader("Upload Raw Options Data CSV", type=["csv"])

fee_override = st.sidebar.number_input(
    "Fee Per Trade (₹)", value=20.0, min_value=0.0, step=5.0
)

k_factor_override = st.sidebar.slider(
    "Economic Gate (K-Factor)", min_value=1.0, max_value=10.0, value=float(DEFAULT_K), step=0.5,
    help="Multiplier for the economic gate threshold."
)

cooldown_override = st.sidebar.slider(
    "Hedge Cooldown (Mins)", min_value=0, max_value=30, value=int(DEFAULT_COOLDOWN), step=1,
    help="Minimum minutes between hedges."
)

# Hardcoded starting capital since it was removed from UI
starting_capital = 175000

# Data Loading & Backtest
@st.cache_data(show_spinner="Running backtest engine...")
def load_and_run(raw_df_bytes, fee, k_factor, cooldown):
    raw_df = pd.read_csv(raw_df_bytes)
    if 'date' in raw_df.columns:
        raw_df['date'] = raw_df['date'].astype(str).str.replace(r'^="?|"$', '', regex=True).str.strip()
    raw_df['date_parsed'] = pd.to_datetime(raw_df['date'], format='mixed', dayfirst=True).dt.date
    total_raw_sessions = raw_df['date_parsed'].nunique()
    df_trades, df_sessions = build_dataframes(raw_df, fee=fee, k_factor=k_factor, cooldown_min=cooldown)
    n_processed_sessions = df_sessions['date'].nunique() if not df_sessions.empty else 0
    removed_sessions = total_raw_sessions - n_processed_sessions
    return df_trades, df_sessions, raw_df, removed_sessions

using_synthetic = False
default_file_path = "FINAL_NIFTY_MASTER_ATM.csv"

if uploaded_file is not None:
    df_trades, df_sessions, raw_df, removed_sessions = load_and_run(uploaded_file, fee_override, k_factor_override, cooldown_override)
elif os.path.exists(default_file_path):
    st.sidebar.success(f"📂 Loaded provided data ({default_file_path})")
    df_trades, df_sessions, raw_df, removed_sessions = load_and_run(default_file_path, fee_override, k_factor_override, cooldown_override)
else:
    st.sidebar.warning("⚠️ No CSV uploaded and default data not found. Using synthetic sample data.")
    using_synthetic = True
    raw_df = generate_synthetic_data(n_sessions=30)
    df_trades, df_sessions = build_dataframes(raw_df, fee=fee_override, k_factor=k_factor_override, cooldown_min=cooldown_override)
    removed_sessions = 0

if using_synthetic:
    st.warning("⚠️ Showing synthetic sample data. Upload your CSV to populate real results.")

# Date range filter
if not df_sessions.empty and 'date' in df_sessions.columns:
    all_dates = pd.to_datetime(df_sessions['date']).dt.date
    date_min, date_max = all_dates.min(), all_dates.max()
    date_range = st.sidebar.date_input(
        "Date Range Filter", value=[date_min, date_max],
        min_value=date_min, max_value=date_max,
    )
    if len(date_range) == 2:
        mask = (all_dates >= date_range[0]) & (all_dates <= date_range[1])
        df_sessions = df_sessions[mask.values].reset_index(drop=True)
        if not df_trades.empty and 'date' in df_trades.columns:
            trade_dates = pd.to_datetime(df_trades['date']).dt.date
            trade_mask = (trade_dates >= date_range[0]) & (trade_dates <= date_range[1])
            df_trades = df_trades[trade_mask.values].reset_index(drop=True)

# Day type filter
if not df_sessions.empty and 'day_type' in df_sessions.columns:
    available_types = [t for t in df_sessions['day_type'].unique() if t != 'skipped']
    day_type_filter = st.sidebar.multiselect(
        "Day Type Filter", options=available_types, default=available_types,
    )
    if day_type_filter:
        df_sessions_filtered = df_sessions[
            (df_sessions['day_type'].isin(day_type_filter)) | (df_sessions['day_type'] == 'skipped')
        ]
    else:
        df_sessions_filtered = df_sessions
else:
    df_sessions_filtered = df_sessions

entered = df_sessions_filtered[df_sessions_filtered['session_entered'] == True] if not df_sessions_filtered.empty else pd.DataFrame()
fired = df_trades[df_trades['was_blocked'] == False] if not df_trades.empty else pd.DataFrame()
blocked = df_trades[df_trades['was_blocked'] == True] if not df_trades.empty else pd.DataFrame()


if not entered.empty:
    net_pnl = entered['net_pnl'].sum()
    gross_pnl = entered['gross_pnl'].sum()
    total_fees = entered['total_fees'].sum()
    fee_drag = (total_fees / gross_pnl * 100) if gross_pnl != 0 else 0

    total_hedges = int(entered['hedge_count'].sum())
    n_sessions_entered = len(entered)
    avg_hedges_day = total_hedges / n_sessions_entered if n_sessions_entered > 0 else 0
    avg_pnl_hedge = net_pnl / total_hedges if total_hedges > 0 else 0

    daily_pnls = entered['net_pnl'].values
    sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252) if len(daily_pnls) > 1 and np.std(daily_pnls) > 0 else 0
    cum_pnl = np.cumsum(daily_pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max
    max_dd = np.min(drawdowns)
    win_rate = (fired['net_pnl'] > 0).mean() * 100 if not fired.empty else 0

    # Calmar = CAGR / Max DD
    n_years = len(daily_pnls) / 252
    cagr_pct = (net_pnl / starting_capital) / n_years * 100 if n_years > 0 else 0
    calmar = abs(cagr_pct / (max_dd / starting_capital * 100)) if max_dd != 0 else 0

    # Profit Factor = gross wins / gross losses
    winning_pnl = daily_pnls[daily_pnls > 0].sum()
    losing_pnl = abs(daily_pnls[daily_pnls < 0].sum())
    profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

    # Row 1: P&L
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1: colored_metric("Net P&L After Fees", format_inr(net_pnl), "green")
    with r1c2: colored_metric("Gross P&L Before Fees", format_inr(gross_pnl), "green")
    with r1c3: colored_metric("Fee Drag %", f"{fee_drag:.1f}%", "red")

    # Row 2: Risk-Adjusted Returns
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1: colored_metric("Sharpe Ratio", f"{sharpe:.2f}", "green")
    with r2c2: colored_metric("Calmar Ratio", f"{calmar:.2f}", "green")
    with r2c3: colored_metric("Profit Factor", f"{profit_factor:.2f}", "green")

    # Max consecutive losing sessions
    max_consec_loss = 0
    cs = 0
    for p in daily_pnls:
        if p < 0:
            cs += 1
            max_consec_loss = max(max_consec_loss, cs)
        else:
            cs = 0

    # Row 3: Risk & Execution
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    with r3c1: colored_metric("Max Drawdown", format_inr(max_dd), "red")
    with r3c2: colored_metric("Max Consecutive Losers", str(max_consec_loss), "red")
    with r3c3: colored_metric("Net Win Rate", f"{win_rate:.1f}%", "green")
    with r3c4: colored_metric("Avg Net P&L/Hedge", format_inr(avg_pnl_hedge), "green")
else:
    st.info("No sessions entered. Adjust filters or upload data.")

st.markdown("---")

PLOTLY_TEMPLATE = "plotly_dark"

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 P&L Analysis",
    "🌍 Regime Breakdown",
    "🔧 Param Robustness",
    "📜 Hedge Log"
])

# Tab 1: P&L Analysis
with tab1:
    if entered.empty:
        st.info("No session data to display.")
    else:
        st.subheader("Cumulative Equity Curve & Drawdown")

        cum_pnl_series = entered['net_pnl'].cumsum()
        cum_pnl_arr = cum_pnl_series.values
        running_max_arr = np.maximum.accumulate(cum_pnl_arr)
        dd_arr = cum_pnl_arr - running_max_arr

        col_eq, col_dd = st.columns(2)
        with col_eq:
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=entered['date'].astype(str), y=cum_pnl_series,
                mode='lines', name='Strategy',
                line=dict(color='#00E5FF', width=2.5),
                fill='tozeroy', fillcolor='rgba(0,229,255,0.15)',
                hovertemplate="Date: %{x}<br>Cumulative: ₹%{y:,.0f}<extra></extra>"
            ))
            fig_cum.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Breakeven")
            fig_cum.update_layout(
                template=PLOTLY_TEMPLATE,
                xaxis_title="Date", yaxis_title="Cumulative Net P&L (₹)",
                height=400, margin=dict(t=30)
            )
            st.plotly_chart(fig_cum, use_container_width=True)

        with col_dd:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=entered['date'].astype(str), y=dd_arr,
                fill='tozeroy', fillcolor='rgba(255,23,68,0.3)',
                line=dict(color='#FF1744', width=2), name='Drawdown',
                hovertemplate="Date: %{x}<br>Drawdown: ₹%{y:,.0f}<extra></extra>"
            ))
            max_dd_idx = np.argmin(dd_arr)
            fig_dd.add_annotation(
                x=entered['date'].astype(str).iloc[max_dd_idx], y=dd_arr[max_dd_idx],
                text=f"Max DD: {format_inr(dd_arr[max_dd_idx])}",
                showarrow=True, arrowhead=2, arrowcolor='white',
                font=dict(color='white', size=11)
            )
            fig_dd.update_layout(
                template=PLOTLY_TEMPLATE,
                xaxis_title="Date", yaxis_title="Drawdown (₹)",
                height=400, margin=dict(t=30)
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("---")

        # Economic Gate Analysis
        if not fired.empty or not blocked.empty:
            st.subheader("Economic Gate Analysis")
            gate_data = []
            if not fired.empty:
                gate_data.append({
                    'Category': 'Hedges Fired', 'Count': len(fired),
                    'Avg Expected Capture (₹)': f'₹{fired["expected_capture"].mean():,.0f}',
                    'Avg Actual Capture (₹)': f'₹{fired["actual_capture"].mean():,.0f}',
                    'Avg Net P&L (₹)': f'₹{fired["net_pnl"].mean():,.0f}',
                })
            if not blocked.empty:
                gate_data.append({
                    'Category': 'Hedges Blocked by Gate', 'Count': len(blocked),
                    'Avg Expected Capture (₹)': f'₹{blocked["blocked_expected_capture"].mean():,.0f}',
                    'Avg Actual Capture (₹)': 'N/A', 'Avg Net P&L (₹)': 'N/A',
                })
            if gate_data:
                st.table(pd.DataFrame(gate_data))

        # Performance by Time Band
        if not fired.empty and 'session_time_band' in fired.columns:
            st.subheader("Performance by Time Band")
            band_agg = fired.groupby('session_time_band').agg(
                count=('net_pnl', 'count'),
                avg_pnl=('net_pnl', 'mean'),
                win_rate=('net_pnl', lambda x: (x > 0).mean() * 100)
            ).reset_index()

            fig_band = go.Figure()
            fig_band.add_trace(go.Bar(
                x=band_agg['session_time_band'], y=band_agg['avg_pnl'],
                marker_color=['#00C853' if x > 0 else '#FF1744' for x in band_agg['avg_pnl']],
                text=[f"n={c}, WR={w:.0f}%" for c, w in zip(band_agg['count'], band_agg['win_rate'])],
                textposition='outside',
            ))
            fig_band.update_layout(
                template=PLOTLY_TEMPLATE,
                xaxis_title="Time Band", yaxis_title="Avg Net P&L (₹)",
                height=350, margin=dict(t=30)
            )
            st.plotly_chart(fig_band, use_container_width=True)

        # Greeks Analysis
        if not fired.empty and 'net_delta' in fired.columns:
            st.subheader("Greeks Analysis: Delta Distribution & P&L at Hedge")
            col_greek1, col_greek2 = st.columns([3, 1])
            with col_greek1:
                tab_g1, tab_g2 = st.tabs(["Delta vs Capture", "Delta vs Net P&L"])
                with tab_g1:
                    fig_greeks1 = px.scatter(
                        fired, x='net_delta', y='expected_capture', color='trigger_type',
                        template=PLOTLY_TEMPLATE, height=350,
                        labels={'net_delta': 'Net Delta', 'expected_capture': 'Expected Capture (₹)', 'trigger_type': 'Trigger'},
                        title="Hedge Triggers: Delta vs Capture Size"
                    )
                    fig_greeks1.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    st.plotly_chart(fig_greeks1, use_container_width=True)
                with tab_g2:
                    fig_greeks2 = px.scatter(
                        fired, x='net_delta', y='net_pnl', color='trigger_type',
                        template=PLOTLY_TEMPLATE, height=350,
                        labels={'net_delta': 'Net Delta', 'net_pnl': 'Net P&L (₹)', 'trigger_type': 'Trigger'},
                        title="Hedge Triggers: Delta vs Net P&L"
                    )
                    fig_greeks2.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    fig_greeks2.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Breakeven")
                    st.plotly_chart(fig_greeks2, use_container_width=True)
            with col_greek2:
                st.metric("Avg Abs Delta at Hedge", f"{fired['net_delta'].abs().mean():.3f}")
                st.metric("Max Abs Delta Hit", f"{fired['net_delta'].abs().max():.3f}")
                
        st.markdown("---")

        # Slippage Sensitivity
        st.subheader("Slippage Sensitivity")
        total_hedges_all = int(entered['hedge_count'].sum())
        baseline_pnl = entered['net_pnl'].sum()
        extra_15x = total_hedges_all * (fee_override * 0.5)
        extra_2x = total_hedges_all * fee_override
        pnl_15x = baseline_pnl - extra_15x
        pnl_2x = baseline_pnl - extra_2x

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric(f"Baseline (₹{fee_override})", format_inr(baseline_pnl))
        sc2.metric(f"1.5× Fee (₹{int(fee_override*1.5)})", format_inr(pnl_15x),
                   delta=format_inr(pnl_15x - baseline_pnl))
        sc3.metric(f"2× Fee (₹{fee_override*2})", format_inr(pnl_2x),
                   delta=format_inr(pnl_2x - baseline_pnl))

        if pnl_2x < 0:
            st.error("⚠️ Strategy turns unprofitable at 2× slippage.")

        st.markdown("---")

        # Best 5 vs Worst 5
        st.subheader("Best 5 vs Worst 5 Sessions")
        display_cols = ['date', 'net_pnl', 'hedge_count', 'entry_iv', 'rv_iv_ratio', 'day_type', 'exit_reason']
        avail_cols = [c for c in display_cols if c in entered.columns]

        col_best, col_worst = st.columns(2)
        with col_best:
            st.markdown("**🟢 Best 5 Sessions**")
            best5 = entered.nlargest(5, 'net_pnl')[avail_cols].copy()
            best5['net_pnl'] = best5['net_pnl'].apply(format_inr)
            st.table(best5)
        with col_worst:
            st.markdown("**🔴 Worst 5 Sessions**")
            worst5 = entered.nsmallest(5, 'net_pnl')[avail_cols].copy()
            worst5['net_pnl'] = worst5['net_pnl'].apply(format_inr)
            st.table(worst5)

# Tab 4: Hedge Log
with tab4:
    st.subheader("Hedge Log")
    display_trades = df_trades
    if not display_trades.empty:
        if 'date' in display_trades.columns:
            dates_avail = display_trades['date'].unique()
            selected_date = st.selectbox("Filter by Date", options=['All'] + list(dates_avail))
            if selected_date != 'All':
                display_trades = display_trades[display_trades['date'] == selected_date]

        log_cols = ['date', 'time', 'spot_at_hedge', 'net_delta', 'expected_capture',
                   'actual_capture', 'fee', 'net_pnl', 'trigger_type',
                   'session_time_band', 'was_blocked', 'blocked_reasons']
        available_cols = [c for c in log_cols if c in display_trades.columns]
        st.dataframe(
            display_trades[available_cols].style.applymap(
                lambda x: 'color: #00C853' if isinstance(x, (int, float)) and x > 0
                else ('color: #FF1744' if isinstance(x, (int, float)) and x < 0 else ''),
                subset=[c for c in ['net_pnl', 'actual_capture'] if c in available_cols]
            ),
            use_container_width=True, height=600
        )

# Tab 2: Regime Breakdown
with tab2:
    if entered.empty or 'day_type' not in entered.columns:
        st.info("No regime data to display.")
    else:
        st.subheader("Regime Performance Summary")
        regimes = entered.groupby('day_type').agg(
            sessions=('net_pnl', 'count'),
            avg_pnl=('net_pnl', 'mean'),
            avg_hedges=('hedge_count', 'mean'),
            total_pnl=('net_pnl', 'sum'),
        ).reset_index()

        regime_wr = entered.groupby('day_type')['net_pnl'].apply(
            lambda x: (x > 0).mean() * 100
        ).reset_index(name='win_rate')
        regimes = regimes.merge(regime_wr, on='day_type')

        fig_regime = make_subplots(specs=[[{"secondary_y": True}]])
        fig_regime.add_trace(
            go.Bar(x=regimes['day_type'], y=regimes['avg_pnl'],
                   name='Avg Daily Net P&L (₹)',
                   marker_color=['#00C853' if x > 0 else '#FF1744' for x in regimes['avg_pnl']]),
            secondary_y=False
        )
        fig_regime.add_trace(
            go.Scatter(x=regimes['day_type'], y=regimes['win_rate'],
                       name='Win Rate %', mode='lines+markers',
                       line=dict(color='#FFAB00', width=3)),
            secondary_y=True
        )
        fig_regime.update_layout(template=PLOTLY_TEMPLATE, height=400, margin=dict(t=30))
        fig_regime.update_yaxes(title_text="Avg P&L (₹)", secondary_y=False)
        fig_regime.update_yaxes(title_text="Win Rate %", secondary_y=True)
        st.plotly_chart(fig_regime, use_container_width=True)

        st.subheader("Regime Summary Table")
        display_regimes = regimes.copy()
        display_regimes['avg_pnl'] = display_regimes['avg_pnl'].apply(lambda x: format_inr(x))
        display_regimes['total_pnl'] = display_regimes['total_pnl'].apply(lambda x: format_inr(x))
        display_regimes['win_rate'] = display_regimes['win_rate'].apply(lambda x: f"{x:.1f}%")
        display_regimes['avg_hedges'] = display_regimes['avg_hedges'].apply(lambda x: f"{x:.1f}")
        st.table(display_regimes)

        # RV/IV Ratio vs P&L
        st.subheader("RV/IV Ratio vs Session P&L")
        if 'rv_iv_ratio' in entered.columns:
            fig_rv = px.scatter(
                entered, x='rv_iv_ratio', y='net_pnl', color='day_type',
                template=PLOTLY_TEMPLATE,
                labels={'rv_iv_ratio': 'RV/IV Ratio', 'net_pnl': 'Net P&L (₹)'},
                height=400,
            )
            valid = entered[['rv_iv_ratio', 'net_pnl']].dropna()
            if len(valid) > 2:
                coeffs = np.polyfit(valid['rv_iv_ratio'], valid['net_pnl'], 1)
                x_range = np.linspace(valid['rv_iv_ratio'].min(), valid['rv_iv_ratio'].max(), 50)
                y_trend = np.polyval(coeffs, x_range)
                fig_rv.add_trace(go.Scatter(
                    x=x_range, y=y_trend, mode='lines',
                    name=f'Trend (slope={coeffs[0]:.0f})',
                    line=dict(color='white', width=2, dash='dash')
                ))
            st.plotly_chart(fig_rv, use_container_width=True)

        # Weekday & Expiry Analysis
        st.markdown("---")
        st.subheader("📅 Weekday & Expiry Analysis")

        if 'day_of_week' in entered.columns:
            day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
            entered_dow = entered.copy()
            entered_dow['day_of_week'] = pd.Categorical(
                entered_dow['day_of_week'], categories=day_order, ordered=True
            )

            dow_agg = entered_dow.groupby('day_of_week', observed=True).agg(
                sessions=('net_pnl', 'count'),
                avg_pnl=('net_pnl', 'mean'),
                total_pnl=('net_pnl', 'sum'),
                avg_hedges=('hedge_count', 'mean'),
            ).reset_index()

            dow_wr = entered_dow.groupby('day_of_week', observed=True)['net_pnl'].apply(
                lambda x: (x > 0).mean() * 100
            ).reset_index(name='win_rate')
            dow_agg = dow_agg.merge(dow_wr, on='day_of_week')

            fig_dow = make_subplots(specs=[[{"secondary_y": True}]])
            fig_dow.add_trace(
                go.Bar(
                    x=dow_agg['day_of_week'], y=dow_agg['avg_pnl'], name='Avg P&L (₹)',
                    marker_color=['#FFD600' if d == 'Thu' else ('#00C853' if p > 0 else '#FF1744')
                                  for d, p in zip(dow_agg['day_of_week'], dow_agg['avg_pnl'])],
                    text=[f"n={n}" for n in dow_agg['sessions']],
                    textposition='outside',
                ),
                secondary_y=False
            )
            fig_dow.add_trace(
                go.Scatter(
                    x=dow_agg['day_of_week'], y=dow_agg['win_rate'],
                    name='Win Rate %', mode='lines+markers',
                    line=dict(color='#00E5FF', width=3)
                ),
                secondary_y=True
            )
            fig_dow.update_layout(
                template=PLOTLY_TEMPLATE, height=400, margin=dict(t=30),
                title="P&L by Day of Week (Thursday = Expiry Day, highlighted yellow)"
            )
            fig_dow.update_yaxes(title_text="Avg P&L (₹)", secondary_y=False)
            fig_dow.update_yaxes(title_text="Win Rate %", secondary_y=True)
            st.plotly_chart(fig_dow, use_container_width=True)

            st.subheader("Weekday Summary Table")
            display_dow = dow_agg.copy()
            display_dow['avg_pnl'] = display_dow['avg_pnl'].apply(lambda x: format_inr(x))
            display_dow['total_pnl'] = display_dow['total_pnl'].apply(lambda x: format_inr(x))
            display_dow['win_rate'] = display_dow['win_rate'].apply(lambda x: f"{x:.1f}%")
            display_dow['avg_hedges'] = display_dow['avg_hedges'].apply(lambda x: f"{x:.1f}")
            st.table(display_dow)

            # Thursday vs Non-Thursday
            if 'is_thursday' in entered.columns:
                st.subheader("Thursday (Expiry) vs Other Days")
                thu = entered[entered['is_thursday'] == True]
                non_thu = entered[entered['is_thursday'] == False]

                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Thu Avg P&L", format_inr(thu['net_pnl'].mean()) if not thu.empty else "N/A")
                tc2.metric("Other Avg P&L", format_inr(non_thu['net_pnl'].mean()) if not non_thu.empty else "N/A")
                tc3.metric("Thu Win Rate", f"{(thu['net_pnl'] > 0).mean() * 100:.1f}%" if not thu.empty else "N/A")
                tc4.metric("Other Win Rate", f"{(non_thu['net_pnl'] > 0).mean() * 100:.1f}%" if not non_thu.empty else "N/A")

                tc5, tc6, tc7, tc8 = st.columns(4)
                tc5.metric("Thu Sessions", len(thu))
                tc6.metric("Other Sessions", len(non_thu))
                tc7.metric("Thu Total P&L", format_inr(thu['net_pnl'].sum()) if not thu.empty else "N/A")
                tc8.metric("Other Total P&L", format_inr(non_thu['net_pnl'].sum()) if not non_thu.empty else "N/A")

                tc9, tc10 = st.columns(2)
                tc9.metric("Thu Avg Hedges/Day", f"{thu['hedge_count'].mean():.1f}" if not thu.empty else "N/A")
                tc10.metric("Other Avg Hedges/Day", f"{non_thu['hedge_count'].mean():.1f}" if not non_thu.empty else "N/A")

            # Days-to-Expiry P&L
            if 'days_to_expiry' in entered.columns:
                st.subheader("P&L by Days to Thursday Expiry")
                dte_agg = entered.groupby('days_to_expiry').agg(
                    sessions=('net_pnl', 'count'),
                    avg_pnl=('net_pnl', 'mean'),
                    total_pnl=('net_pnl', 'sum'),
                ).reset_index()
                dte_agg['day_label'] = dte_agg['days_to_expiry'].map({
                    0: 'Thu (0)', 1: 'Wed (1)', 2: 'Tue (2)', 3: 'Mon (3)', 4: 'Fri (4)',
                })

                fig_dte = go.Figure(go.Bar(
                    x=dte_agg['day_label'], y=dte_agg['avg_pnl'],
                    marker_color=['#FFD600' if d == 0 else ('#00C853' if p > 0 else '#FF1744')
                                  for d, p in zip(dte_agg['days_to_expiry'], dte_agg['avg_pnl'])],
                    text=[f"n={n}, total={format_inr(t)}" for n, t in zip(dte_agg['sessions'], dte_agg['total_pnl'])],
                    textposition='outside',
                ))
                fig_dte.update_layout(
                    template=PLOTLY_TEMPLATE, height=400, margin=dict(t=30),
                    xaxis_title="Days to Thursday Expiry",
                    yaxis_title="Avg Session P&L (₹)",
                    title="Does proximity to expiry affect P&L?"
                )
                st.plotly_chart(fig_dte, use_container_width=True)

# Tab 3: Parameter Robustness
with tab3:
    st.subheader("Parameter Robustness Heatmap")
    run_sweeps = st.button("🔬 Generate Robustness Heatmap", type="primary")

    if run_sweeps:
        with st.spinner("Generating K × Cooldown heatmap..."):
            k_sweep = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
            cd_sweep = [5, 7, 9, 11, 13, 15]
            heatmap_results = run_2d_sweep(
                raw_df, 'k_factor', k_sweep, 'cooldown_min', cd_sweep
            )

        if not heatmap_results.empty:
            st.subheader("K × Cooldown Heatmap (Sharpe Ratio)")
            pivot = heatmap_results.pivot(index='param1', columns='param2', values='sharpe')
            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=[str(r) for r in pivot.index],
                colorscale='RdYlGn',
                text=np.round(pivot.values, 2),
                texttemplate='%{text}',
                hovertemplate='K=%{y}, Cooldown=%{x}min<br>Sharpe: %{z:.2f}<extra></extra>'
            ))
            fig_hm.update_layout(
                template=PLOTLY_TEMPLATE,
                xaxis_title="Cooldown (min)", yaxis_title="K Factor",
                height=450, margin=dict(t=30),
                title="Parameter Grid: Sharpe Ratio by K Factor × Cooldown"
            )
            st.plotly_chart(fig_hm, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 0.8em;'>"
    "Nifty50 Gamma Scalp 1Min — V1.0"
    "</div>",
    unsafe_allow_html=True
)
