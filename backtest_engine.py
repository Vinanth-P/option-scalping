"""
Backtest Engine — Nifty50 ATM Straddle Gamma Scalping

Preprocesses raw 1-minute options CSV, runs session-by-session backtest,
builds df_trades and df_sessions DataFrames for the dashboard,
and supports parameter sensitivity sweeps.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


# CONSTANTS
RISK_FREE_RATE = 0.065          # India risk-free rate
TRADING_MINUTES_PER_DAY = 375   # 9:15 AM to 3:30 PM
TRADING_DAYS_PER_YEAR = 252
LOT_SIZE = 50
DEFAULT_FEE = 20.0              # Execution fee per hedge (₹20 buy, ₹20 sell = ₹40 total, but code deducts per leg)
GATE_BASIS_COST = 40.0          # Basis for economic gate (K * ₹40 threshold)
DEFAULT_K = 6.0                 # Economic multiplier (optimized via heatmap: K=6, CD=15 → Sharpe 3.30)
DEFAULT_COOLDOWN = 15           # Minutes between hedges (optimized via heatmap)
MAX_COOLDOWN_MINUTES = 15       # Hard cap — never exceed 15 min cooldown
DEFAULT_MAX_HEDGES = 30         # Daily cap
DEFAULT_OPEN_FILTER = 0.0010    # First 15-min move threshold (0.10%, optimized via sweep)
DEFAULT_IV_CRUSH = 12.0         # IV drop % for exit (optimized via sweep +25k P&L, 1250 hedges)
STRADDLE_ENTRY_EXIT_FEE = 80.0  # Brokerage for straddle open + close (₹20 x 4)


# BLACK-SCHOLES HELPERS
def bs_d1(spot, strike, T, r, iv):
    """Calculate d1 for Black-Scholes."""
    if T <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    return (np.log(spot / strike) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))


def bs_delta_call(spot, strike, T, r, iv):
    """Call delta = N(d1)."""
    d1 = bs_d1(spot, strike, T, r, iv)
    return norm.cdf(d1)


def bs_delta_put(spot, strike, T, r, iv):
    """Put delta = N(d1) - 1."""
    d1 = bs_d1(spot, strike, T, r, iv)
    return norm.cdf(d1) - 1.0


# DATA PREPROCESSING
def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw options CSV into a unified bar-by-bar DataFrame.
    
    Input columns: date, time, symbol, option_type, type, strike_offset,
                   open, high, low, close, volume, oi, iv, spot
    
    Returns DataFrame with one row per bar, containing:
    - spot, close_CE, close_PE, straddle_price
    - iv_CE, iv_PE, avg_iv
    - delta_CE, delta_PE, net_delta
    - atr_5 (5-bar ATR on spot)
    - realized_var (intraday cumulative)
    - volume_CE, volume_PE, oi_CE, oi_PE
    """
    df = df.copy()
    
    # Check if data is already in wide format (has CE_close, PE_close, etc.)
    if 'CE_close' in df.columns and 'PE_close' in df.columns:
        # Rename to match expected format
        df = df.rename(columns={
            'CE_close': 'close_CE',
            'PE_close': 'close_PE',
            'CE_open': 'open_CE',
            'PE_open': 'open_PE',
            'CE_high': 'high_CE',
            'PE_high': 'high_PE',
            'CE_low': 'low_CE',
            'PE_low': 'low_PE',
            'CE_volume': 'volume_CE',
            'PE_volume': 'volume_PE',
            'CE_oi': 'oi_CE',
            'PE_oi': 'oi_PE',
        })
        if 'iv' in df.columns and 'iv_CE' not in df.columns:
            df['iv_CE'] = df['iv']
            df['iv_PE'] = df['iv']
            
        # Clean Excel-style date/time wrappers like ="03-02-26"
        for col in ['date', 'time']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'^="?|"$', '', regex=True).str.strip()
                
        # Create timestamp
        if 'timestamp' not in df.columns:
            if 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            else:
                df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='mixed', dayfirst=True)
                
        df['date_parsed'] = df['timestamp'].dt.date
        df['time_parsed'] = df['timestamp'].dt.time
        
        # Ensure straddle_price exists
        if 'Straddle_Price' in df.columns:
            df['straddle_price'] = df['Straddle_Price']
        elif 'straddle_price' not in df.columns:
            df['straddle_price'] = df['close_CE'] + df['close_PE']
            
        processed = df.sort_values('timestamp').reset_index(drop=True)
    else:
        # Clean Excel-style date formatting
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str).str.replace(r'^="?|"?$', '', regex=True)
        
        # Normalize option_type
        if 'option_type' in df.columns:
            df['option_type'] = df['option_type'].str.upper().str.strip()
        elif 'type' in df.columns:
            df['option_type'] = df['type'].str.upper().str.strip()
        
        # Filter ATM straddle only (strike_offset == 0 or "ATM")
        if 'strike_offset' in df.columns:
            # Handle various ATM representations
            atm_mask = (
                (df['strike_offset'] == 0) |
                (df['strike_offset'] == '0') |
                (df['strike_offset'].astype(str).str.upper() == 'ATM')
            )
            df = df[atm_mask].copy()
        
        # Clean Excel-style date/time wrappers like ="03-02-26"
        for col in ['date', 'time']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'^="?|"$', '', regex=True).str.strip()
        
        # Create timestamp
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='mixed', dayfirst=True)
        df['date_parsed'] = df['timestamp'].dt.date
        df['time_parsed'] = df['timestamp'].dt.time
        
        # Separate CE and PE
        ce = df[df['option_type'].isin(['CE', 'CALL'])].copy()
        pe = df[df['option_type'].isin(['PE', 'PUT'])].copy()
        
        # Create merge key
        ce['merge_key'] = ce['date'].astype(str) + '_' + ce['time'].astype(str)
        pe['merge_key'] = pe['date'].astype(str) + '_' + pe['time'].astype(str)
        
        # Merge CE and PE on same bar
        merged = ce.merge(
            pe[['merge_key', 'close', 'open', 'high', 'low', 'volume', 'oi', 'iv']],
            on='merge_key',
            suffixes=('_CE', '_PE'),
            how='inner'
        )
        
        # Rename for clarity
        merged = merged.rename(columns={
            'close_CE': 'close_CE',
            'close_PE': 'close_PE',
            'iv_CE': 'iv_CE',
            'iv_PE': 'iv_PE',
            'volume_CE': 'volume_CE',
            'volume_PE': 'volume_PE',
            'oi_CE': 'oi_CE',
            'oi_PE': 'oi_PE',
            'open_CE': 'open_CE',
            'open_PE': 'open_PE',
            'high_CE': 'high_CE',
            'high_PE': 'high_PE',
            'low_CE': 'low_CE',
            'low_PE': 'low_PE',
        })
        
        processed = merged.sort_values('timestamp').reset_index(drop=True)
        
        # Straddle price
        processed['straddle_price'] = processed['close_CE'] + processed['close_PE']
    
    # Average IV (ensure it's in decimal form)
    for col in ['iv_CE', 'iv_PE']:
        if processed[col].median() > 1:  # Likely in percentage form
            processed[col] = processed[col] / 100.0
    processed['avg_iv'] = (processed['iv_CE'] + processed['iv_PE']) / 2
    
    # Data Quality Checks
    # 1. Dynamic volume guard threshold (20th percentile with floor 100)
    if 'volume_CE' in processed.columns:
        vol_guard_threshold = max(
            processed['volume_CE'].quantile(0.20),
            100  # Absolute floor — never go below 100
        )
        processed['low_volume'] = processed['volume_CE'] < vol_guard_threshold
        processed['unreliable_iv'] = processed['low_volume']
    else:
        vol_guard_threshold = 100
        processed['low_volume'] = False
        processed['unreliable_iv'] = False
        
    # 2. Flag bars where iv <= 0 or iv > 2.0 (data errors)
    processed['iv_error'] = (processed['avg_iv'] <= 0) | (processed['avg_iv'] > 2.0)
    
    # 3. Flag bars where |spot_t - spot_{t-1}| > 1% (potential data spike)
    processed['spot_spike'] = processed['spot'].pct_change().abs() > 0.01
    
    # 4. Confirm bars exist per session
    session_counts = processed.groupby('date_parsed').size()
    complete_sessions = session_counts[session_counts >= 300].index
    processed = processed[processed['date_parsed'].isin(complete_sessions)].copy()
    
    # --- Per-session derived columns ---
    all_results = []
    sessions = processed['date_parsed'].unique()
    
    for session_date in sessions:
        session = processed[processed['date_parsed'] == session_date].copy()
        if session.empty: continue
        
        n = len(session)
        spots = session['spot'].values
        strike = round(spots[0] / 50) * 50
        session['strike'] = strike
        
        # T values
        bars_from_start = np.arange(n)
        rem_min = np.maximum(TRADING_MINUTES_PER_DAY - bars_from_start, 1)
        session['T'] = rem_min / (TRADING_MINUTES_PER_DAY * TRADING_DAYS_PER_YEAR)
        T_values = session['T'].values
        
        # Deltas
        iv_ce = session['iv_CE'].values
        iv_pe = session['iv_PE'].values
        session['delta_CE'] = [bs_delta_call(spots[i], strike, T_values[i], RISK_FREE_RATE, iv_ce[i]) for i in range(n)]
        session['delta_PE'] = [bs_delta_put(spots[i], strike, T_values[i], RISK_FREE_RATE, iv_pe[i]) for i in range(n)]
        session['net_delta'] = session['delta_CE'] + session['delta_PE']
        
        # ATR (5-bar on spot)
        spot_changes = np.abs(np.diff(spots, prepend=spots[0]))
        atr_values = np.full(n, spots[0] * 0.0025)
        for i in range(5, n):
            atr_values[i] = np.mean(spot_changes[i-5:i])
        session['atr_5'] = atr_values
        
        # Realized variance
        log_returns = np.zeros(n)
        log_returns[1:] = np.log(spots[1:] / spots[:-1])
        cum_rv = np.cumsum(log_returns**2)
        session['realized_var'] = cum_rv * TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY
        
        # Entry IV and drop
        session['entry_iv'] = iv_ce[0]
        session['iv_drop_pct'] = (iv_ce[0] - iv_ce) / iv_ce[0] * 100
        session['bar_idx'] = np.arange(n)
        
        all_results.append(session)
    
    if not all_results: return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


# SESSION BACKTEST LOGIC
def run_session_backtest(
    session_df: pd.DataFrame,
    fee: float = DEFAULT_FEE,
    k_factor: float = DEFAULT_K,
    cooldown_min: int = min(DEFAULT_COOLDOWN, MAX_COOLDOWN_MINUTES),
    max_hedges: int = DEFAULT_MAX_HEDGES,
    open_filter: float = DEFAULT_OPEN_FILTER,
    iv_crush_threshold: float = DEFAULT_IV_CRUSH,
    noon_rv_ratio: float = 0.40,
    afternoon_rv_ratio: float = 0.70,
    median_session_iv: float = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run backtest on a single session (one day).
    
    Returns:
        trades_df: DataFrame of hedge events (fired + blocked)
        session_summary: Dict with session-level metrics
    """
    if session_df.empty or len(session_df) < 15:
        return pd.DataFrame(), {}
    
    session_date = session_df['date_parsed'].iloc[0]
    spots = session_df['spot'].values
    timestamps = session_df['timestamp'].values
    net_deltas = session_df['net_delta'].values
    straddle_prices = session_df['straddle_price'].values
    iv_ce = session_df['iv_CE'].values
    atr_values = session_df['atr_5'].values
    realized_vars = session_df['realized_var'].values
    iv_drops = session_df['iv_drop_pct'].values
    unreliable_iv = session_df['unreliable_iv'].values
    iv_errors = session_df['iv_error'].values
    n = len(session_df)
    
    entry_iv = iv_ce[0]
    entry_spot = spots[0]
    
    # --- Fix 1: IV-Adjusted Delta Bands ---
    if median_session_iv is not None and median_session_iv > 0:
        iv_adjustment = min(entry_iv / median_session_iv, 1.0)
    else:
        iv_adjustment = 1.0
    
    # --- Patch 0: First 15-min movement check ---
    first_15_bars = min(15, n)
    first_15m_high = np.max(spots[:first_15_bars])
    first_15m_low = np.min(spots[:first_15_bars])
    first_15m_move = (first_15m_high - first_15m_low) / spots[0] if spots[0] > 0 else 0
    
    session_entered = first_15m_move >= open_filter
    
    # --- High-IV Regime Filter ---
    # Skip sessions where entry IV > median IV across all sessions.
    # Data shows high-IV sessions (both chop and trend) consistently lose money
    # because expensive straddles bleed theta faster than gamma can recover.
    high_iv_skipped = False
    if session_entered and median_session_iv is not None and median_session_iv > 0:
        if entry_iv > median_session_iv:
            session_entered = False
            high_iv_skipped = True
    
    summary = {
        'date': session_date,
        'entry_iv': entry_iv,
        'entry_spot': entry_spot,
        'first_15min_move_pct': first_15m_move * 100,
        'session_entered': session_entered,
    }
    
    if not session_entered:
        exit_reason = 'high_iv_skip' if high_iv_skipped else 'not_entered'
        summary.update({
            'exit_iv': iv_ce[-1],
            'iv_drop_pct': iv_drops[-1],
            'gross_pnl': 0, 'total_fees': 0, 'net_pnl': 0,
            'straddle_pnl': 0, 'hedge_pnl_total': 0,
            'hedge_count': 0, 'realized_var': realized_vars[-1],
            'implied_var': entry_iv**2,
            'rv_iv_ratio': 0,
            'iv_crush_exit': False, 'cut_50pct': False,
            'full_exit_14h30': False, 'daily_cap_hit': False,
            'day_type': 'skipped', 'exit_reason': exit_reason,
            'high_iv_skipped': high_iv_skipped,
            'spot_close': spots[-1], 'spot_high': np.max(spots),
            'spot_low': np.min(spots),
        })
        return pd.DataFrame(), summary
    
    # --- Session entered at bar 15 (9:30 AM) ---
    entry_bar = first_15_bars
    entry_price = straddle_prices[entry_bar] if entry_bar < n else straddle_prices[-1]
    entry_spot_actual = spots[entry_bar] if entry_bar < n else spots[-1]
    
    trades = []
    hedge_count = 0
    
    # Fix prompt Root Cause 2: Initialize last_hedge_spot to entry_spot
    # This ensures the first hedge can fire based on accumulated move from entry
    last_hedge_spot = entry_spot_actual
    
    last_hedge_time = timestamps[entry_bar]
    position_scale = 1.0
    iv_crush_exit = False
    cut_50pct = False
    full_exit_14h30 = False
    daily_cap_hit = False
    exit_bar = n - 1  # Default: exit at session close
    exit_reason = 'session_close'
    total_hedge_pnl = 0
    
    for i in range(entry_bar + 1, n):
        ts = pd.Timestamp(timestamps[i])
        current_time = ts.time() if hasattr(ts, 'time') else ts.to_pydatetime().time()
        spot = spots[i]
        delta = net_deltas[i]
        atr = atr_values[i]
        
        # --- Volume Guard: Skip low-volume bars entirely ---
        if 'low_volume' in session_df.columns and session_df['low_volume'].values[i]:
            continue  # skip delta computation and hedge evaluation
        
        # --- Patch 7: IV Crush Exit (volume-guarded) ---
        if iv_drops[i] >= iv_crush_threshold:
            # Only trust IV crush on bars with sufficient volume
            if 'low_volume' not in session_df.columns or not session_df['low_volume'].values[i]:
                iv_crush_exit = True
                exit_bar = i
                exit_reason = 'iv_crush'
                break
        
        # --- Patch 6: Variance checks ---
        bars_elapsed = i - entry_bar
        implied_var_so_far = entry_iv**2 * (bars_elapsed / TRADING_MINUTES_PER_DAY)
        
        hour, minute = current_time.hour, current_time.minute
        
        if hour == 12 and minute >= 30 and not cut_50pct:
            if implied_var_so_far > 0 and realized_vars[i] < noon_rv_ratio * implied_var_so_far:
                cut_50pct = True
                position_scale = 0.5
        
        if hour == 14 and minute >= 30:
            if implied_var_so_far > 0 and realized_vars[i] < afternoon_rv_ratio * implied_var_so_far:
                full_exit_14h30 = True
                exit_bar = i
                exit_reason = 'rv_exit_1430'
                break
        
        # --- Patch 5: Daily hedge cap ---
        if hedge_count >= max_hedges:
            daily_cap_hit = True
            continue  # Keep straddle open but no more hedging
        
        # --- Hedge logic (Patches 1-4) ---
        # Patch 2: Time-adaptive delta band (with IV adjustment from Fix 1)
        if hour >= 14:
            delta_band = 0.35 * iv_adjustment
            time_band = 'late'
        elif hour >= 12:
            delta_band = 0.30 * iv_adjustment
            time_band = 'midday'
        else:
            delta_band = 0.25 * iv_adjustment
            time_band = 'morning'
        
        abs_delta = abs(delta)
        move_since_hedge = abs(spot - (last_hedge_spot if last_hedge_spot is not None else entry_spot_actual))
        
        # Check delta condition
        delta_triggered = abs_delta > delta_band
        
        # Patch 3: ATR move filter (with warm-up lockout)
        if (i - entry_bar) < 25:
            # Phase 1: Before ~9:40 AM - use static 0.25% filter
            required_move = spot * 0.0025
        else:
            # Phase 2: After warm-up - ATR is the primary filter
            # ATR scales naturally with volatility; static floor removed
            # to allow realistic hedge frequency (target: 3-15/session)
            required_move = 1.5 * atr if atr > 0 else spot * 0.0025
        atr_triggered = move_since_hedge >= required_move
        
        # Patch 4: Cooldown (7 minutes between hedges)
        elapsed_minutes = (pd.Timestamp(timestamps[i]) - pd.Timestamp(last_hedge_time)).total_seconds() / 60
        cooldown_ok = elapsed_minutes >= cooldown_min
        
        # Patch 1: Economic gate (Required move must be large enough)
        expected_capture = abs_delta * move_since_hedge * LOT_SIZE
        economic_gate_ok = expected_capture >= k_factor * GATE_BASIS_COST
        
        # Fix 6: First Hedge Gate Bypass
        # The first hedge of a session always passes the gate.
        # Without this, last_hedge_spot = entry_spot → move ≈ 0 → gate blocks → cycle repeats.
        if hedge_count == 0:
            economic_gate_ok = True
            
        # Section 10: Diagnostic Block Logging
        is_blocked = False
        block_reasons = []
        
        if delta_triggered and atr_triggered and cooldown_ok:
            if not economic_gate_ok:
                is_blocked = True
                block_reasons.append(f"GATE | capture={expected_capture:.0f} < {k_factor*GATE_BASIS_COST:.0f}")
        
        # Determine trigger type
        if delta_triggered and atr_triggered:
            trigger_type = 'both'
        elif delta_triggered:
            trigger_type = 'delta_band'
        elif atr_triggered:
            trigger_type = 'atr_move'
        else:
            trigger_type = 'none'
        
        # Execution condition
        if delta_triggered and atr_triggered and cooldown_ok and economic_gate_ok:
            # Execute hedge
            hedge_pnl = delta * (spot - last_hedge_spot) * LOT_SIZE * position_scale
            net_hedge_pnl = hedge_pnl - fee
            total_hedge_pnl += net_hedge_pnl
            hedge_count += 1
            
            trades.append({
                'date': session_date,
                'time': str(current_time),
                'timestamp': timestamps[i],
                'spot_at_hedge': spot,
                'net_delta': delta,
                'expected_capture': expected_capture,
                'actual_capture': hedge_pnl,
                'fee': fee,
                'net_pnl': net_hedge_pnl,
                'trigger_type': trigger_type,
                'session_time_band': time_band,
                'was_blocked': False,
                'blocked_reasons': "",
                'blocked_expected_capture': 0,
                'volume_at_bar': session_df['volume_CE'].values[i] if 'volume_CE' in session_df.columns else 0,
                'oi_at_bar': session_df['oi_CE'].values[i] if 'oi_CE' in session_df.columns else 0,
            })
            
            last_hedge_spot = spot
            last_hedge_time = timestamps[i]
        elif is_blocked:
            # Blocked by economic gate
            trades.append({
                'date': session_date,
                'time': str(current_time),
                'timestamp': timestamps[i],
                'spot_at_hedge': spot,
                'net_delta': delta,
                'expected_capture': expected_capture,
                'actual_capture': 0,
                'fee': 0,
                'net_pnl': 0,
                'trigger_type': trigger_type,
                'session_time_band': time_band,
                'was_blocked': True,
                'blocked_reasons': " | ".join(block_reasons),
                'blocked_expected_capture': expected_capture,
                'volume_at_bar': session_df['volume_CE'].values[i] if 'volume_CE' in session_df.columns else 0,
                'oi_at_bar': session_df['oi_CE'].values[i] if 'oi_CE' in session_df.columns else 0,
            })
        
    # Session exit
    exit_price = straddle_prices[exit_bar] if exit_bar < n else straddle_prices[-1]
    straddle_pnl = (exit_price - entry_price) * LOT_SIZE * position_scale
    entry_exit_fees = STRADDLE_ENTRY_EXIT_FEE  # Straddle open + close brokerage
    total_fees = hedge_count * fee + entry_exit_fees
    gross_pnl = straddle_pnl + total_hedge_pnl + hedge_count * fee  # Add back hedge fees for gross
    net_pnl = straddle_pnl + total_hedge_pnl - entry_exit_fees
    
    # RV/IV ratio
    final_rv = realized_vars[exit_bar] if exit_bar < len(realized_vars) else 0
    implied_var = entry_iv**2
    rv_iv_ratio = final_rv / implied_var if implied_var > 0 else 0
    
    # Day type classification
    spot_open = spots[entry_bar]
    spot_close = spots[exit_bar] if exit_bar < n else spots[-1]
    spot_high = np.max(spots[entry_bar:exit_bar+1]) if exit_bar < n else np.max(spots[entry_bar:])
    spot_low = np.min(spots[entry_bar:exit_bar+1]) if exit_bar < n else np.min(spots[entry_bar:])
    
    trending = abs(spot_close - spot_open) / max(spot_high - spot_low, 0.01) > 0.5
    
    summary.update({
        'exit_iv': iv_ce[exit_bar] if exit_bar < len(iv_ce) else iv_ce[-1],
        'iv_drop_pct': iv_drops[exit_bar] if exit_bar < len(iv_drops) else iv_drops[-1],
        'gross_pnl': gross_pnl,
        'total_fees': total_fees,
        'net_pnl': net_pnl,
        'hedge_count': hedge_count,
        'realized_var': final_rv,
        'implied_var': implied_var,
        'rv_iv_ratio': rv_iv_ratio,
        'iv_crush_exit': iv_crush_exit,
        'cut_50pct': cut_50pct,
        'full_exit_14h30': full_exit_14h30,
        'daily_cap_hit': daily_cap_hit,
        'exit_reason': exit_reason,
        'spot_close': spot_close,
        'spot_high': spot_high,
        'spot_low': spot_low,
        'straddle_entry': entry_price,
        'straddle_exit': exit_price,
        'straddle_pnl': straddle_pnl,
        'hedge_pnl_total': total_hedge_pnl,
    })
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return trades_df, summary


def classify_day_type(session_summaries: List[Dict]) -> List[Dict]:
    """Add day_type, day_of_week, is_thursday, is_expiry_week classification."""
    entered = [s for s in session_summaries if s.get('session_entered', False)]
    if not entered:
        return session_summaries
    
    median_iv = np.median([s['entry_iv'] for s in entered])
    
    # Day name mapping
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    for s in session_summaries:
        # --- Fix 5: Day-of-week and expiry flags ---
        session_date = s.get('date')
        if session_date is not None:
            if not isinstance(session_date, pd.Timestamp):
                session_date = pd.Timestamp(session_date)
            dow = session_date.dayofweek  # 0=Mon, 3=Thu, 4=Fri
            s['day_of_week'] = day_names[dow]
            s['is_thursday'] = (dow == 3)
            # Nifty has weekly expiry on Thursday — every week is expiry week
            # But flag if this specific day is within 2 days of Thursday expiry
            s['is_expiry_week'] = True  # Weekly expiries = always
            s['days_to_expiry'] = (3 - dow) % 7  # Days until next Thursday
            s['is_expiry_day'] = (dow == 3)
        else:
            s['day_of_week'] = 'Unknown'
            s['is_thursday'] = False
            s['is_expiry_week'] = True
            s['days_to_expiry'] = -1
            s['is_expiry_day'] = False
        
        if not s.get('session_entered', False):
            s['day_type'] = 'skipped'
            continue
        
        high_iv = s['entry_iv'] > median_iv
        # Trending check
        sp_open = s.get('entry_spot', 0)
        sp_close = s.get('spot_close', sp_open)
        sp_high = s.get('spot_high', sp_open)
        sp_low = s.get('spot_low', sp_open)
        range_val = max(sp_high - sp_low, 0.01)
        trending = abs(sp_close - sp_open) / range_val > 0.5
        
        if high_iv and trending:
            s['day_type'] = 'high_iv_trend'
        elif high_iv and not trending:
            s['day_type'] = 'high_iv_chop'
        elif not high_iv and trending:
            s['day_type'] = 'low_iv_trend'
        else:
            s['day_type'] = 'low_iv_chop'
    
    return session_summaries


# ORCHESTRATOR
def build_dataframes(
    df: pd.DataFrame,
    fee: float = DEFAULT_FEE,
    k_factor: float = DEFAULT_K,
    cooldown_min: int = DEFAULT_COOLDOWN,
    max_hedges: int = DEFAULT_MAX_HEDGES,
    open_filter: float = DEFAULT_OPEN_FILTER,
    iv_crush_threshold: float = DEFAULT_IV_CRUSH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: preprocess → run all sessions → return (df_trades, df_sessions).
    """
    processed = preprocess_raw_data(df)
    if processed.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    sessions = processed['date_parsed'].unique()
    all_trades = []
    all_summaries = []
    
    # Compute median session IV across all sessions for IV adjustment (Fix 1)
    session_entry_ivs = []
    for session_date in sessions:
        session_df = processed[processed['date_parsed'] == session_date]
        if len(session_df) >= 15 and 'iv_CE' in session_df.columns:
            session_entry_ivs.append(session_df['iv_CE'].values[0])
    median_session_iv = np.median(session_entry_ivs) if session_entry_ivs else None
    
    for session_date in sessions:
        session_df = processed[processed['date_parsed'] == session_date].copy()
        trades_df, summary = run_session_backtest(
            session_df,
            fee=fee,
            k_factor=k_factor,
            cooldown_min=cooldown_min,
            max_hedges=max_hedges,
            open_filter=open_filter,
            iv_crush_threshold=iv_crush_threshold,
            median_session_iv=median_session_iv,
        )
        if not trades_df.empty:
            all_trades.append(trades_df)
        if summary:
            all_summaries.append(summary)
    
    # Classify day types
    all_summaries = classify_day_type(all_summaries)
    
    df_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    df_sessions = pd.DataFrame(all_summaries) if all_summaries else pd.DataFrame()
    
    return df_trades, df_sessions


# PARAMETER SENSITIVITY SWEEPS
def run_parameter_sweep(
    df: pd.DataFrame,
    param_name: str,
    values: list,
    base_params: Dict = None,
) -> pd.DataFrame:
    """
    Sweep one parameter while holding others at base values.
    Returns DataFrame with columns: param_value, net_pnl, hedge_count, sharpe, sessions_entered.
    """
    if base_params is None:
        base_params = {
            'fee': DEFAULT_FEE,
            'k_factor': DEFAULT_K,
            'cooldown_min': DEFAULT_COOLDOWN,
            'max_hedges': DEFAULT_MAX_HEDGES,
            'open_filter': DEFAULT_OPEN_FILTER,
            'iv_crush_threshold': DEFAULT_IV_CRUSH,
        }
    
    processed = preprocess_raw_data(df)
    if processed.empty:
        return pd.DataFrame()
    
    sessions = processed['date_parsed'].unique()
    
    # Compute median session IV across all sessions for IV adjustment (Fix 1)
    session_entry_ivs = []
    for session_date in sessions:
        session_df = processed[processed['date_parsed'] == session_date]
        if len(session_df) >= 15 and 'iv_CE' in session_df.columns:
            session_entry_ivs.append(session_df['iv_CE'].values[0])
    median_session_iv = np.median(session_entry_ivs) if session_entry_ivs else None
    
    results = []
    
    for val in values:
        params = base_params.copy()
        params[param_name] = val
        params['median_session_iv'] = median_session_iv
        
        total_pnl = 0
        total_hedges = 0
        daily_pnls = []
        sessions_entered = 0
        
        for session_date in sessions:
            session_df = processed[processed['date_parsed'] == session_date].copy()
            _, summary = run_session_backtest(session_df, **params)
            if summary:
                total_pnl += summary.get('net_pnl', 0)
                total_hedges += summary.get('hedge_count', 0)
                if summary.get('session_entered', False):
                    daily_pnls.append(summary.get('net_pnl', 0))
                    sessions_entered += 1
        
        # Sharpe ratio
        if len(daily_pnls) > 1:
            sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252) if np.std(daily_pnls) > 0 else 0
        else:
            sharpe = 0
        
        results.append({
            'param_value': val,
            'net_pnl': total_pnl,
            'hedge_count': total_hedges,
            'sharpe': sharpe,
            'sessions_entered': sessions_entered,
        })
    
    return pd.DataFrame(results)


def run_2d_sweep(
    df: pd.DataFrame,
    param1_name: str, param1_values: list,
    param2_name: str, param2_values: list,
    base_params: Dict = None,
) -> pd.DataFrame:
    """
    2D parameter sweep for heatmap (e.g., K × Cooldown).
    Returns DataFrame with columns: param1, param2, net_pnl, sharpe.
    """
    if base_params is None:
        base_params = {
            'fee': DEFAULT_FEE,
            'k_factor': DEFAULT_K,
            'cooldown_min': DEFAULT_COOLDOWN,
            'max_hedges': DEFAULT_MAX_HEDGES,
            'open_filter': DEFAULT_OPEN_FILTER,
            'iv_crush_threshold': DEFAULT_IV_CRUSH,
        }
    
    processed = preprocess_raw_data(df)
    if processed.empty:
        return pd.DataFrame()
    
    sessions = processed['date_parsed'].unique()
    
    # Compute median session IV across all sessions for IV adjustment (Fix 1)
    session_entry_ivs = []
    for session_date in sessions:
        session_df = processed[processed['date_parsed'] == session_date]
        if len(session_df) >= 15 and 'iv_CE' in session_df.columns:
            session_entry_ivs.append(session_df['iv_CE'].values[0])
    median_session_iv = np.median(session_entry_ivs) if session_entry_ivs else None
    
    results = []
    
    for v1 in param1_values:
        for v2 in param2_values:
            params = base_params.copy()
            params[param1_name] = v1
            params[param2_name] = v2
            params['median_session_iv'] = median_session_iv
            
            daily_pnls = []
            total_pnl = 0
            
            for session_date in sessions:
                session_df = processed[processed['date_parsed'] == session_date].copy()
                _, summary = run_session_backtest(session_df, **params)
                if summary and summary.get('session_entered', False):
                    pnl = summary.get('net_pnl', 0)
                    daily_pnls.append(pnl)
                    total_pnl += pnl
            
            sharpe = 0
            if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
                sharpe = (np.mean(daily_pnls) / np.std(daily_pnls)) * np.sqrt(252)
            
            results.append({
                'param1': v1,
                'param2': v2,
                'net_pnl': total_pnl,
                'sharpe': sharpe,
            })
    
    return pd.DataFrame(results)


# SYNTHETIC DATA GENERATOR
def generate_synthetic_data(n_sessions: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic raw options CSV data for testing.
    Mimics real NIFTY options data format.
    """
    np.random.seed(seed)
    
    rows = []
    base_spot = 24000
    base_date = datetime(2025, 1, 6)
    
    for day in range(n_sessions):
        session_date = base_date + timedelta(days=day)
        if session_date.weekday() >= 5:  # Skip weekends
            continue
        
        date_str = session_date.strftime('%d-%m-%y')
        spot = base_spot + np.random.normal(0, 200)
        iv_base = np.random.uniform(0.10, 0.20)
        
        for minute in range(TRADING_MINUTES_PER_DAY):
            hour = 9 + (minute + 15) // 60
            min_val = (minute + 15) % 60
            time_str = f"{hour:02d}:{min_val:02d}:00"
            
            # Spot drift
            spot += np.random.normal(0, spot * 0.0003)
            iv = iv_base + np.random.normal(0, 0.005)
            iv = max(0.05, iv)
            
            strike = round(spot / 50) * 50
            
            # ATM CE price approximation
            ce_price = max(1, spot - strike + spot * iv * np.sqrt(1/252) * 0.5 + np.random.normal(0, 2))
            pe_price = max(1, strike - spot + spot * iv * np.sqrt(1/252) * 0.5 + np.random.normal(0, 2))
            
            vol = int(np.random.exponential(500000))
            oi_val = int(np.random.exponential(2000000))
            
            for otype, price in [('CE', ce_price), ('PE', pe_price)]:
                rows.append({
                    'date': f'="{date_str}"',
                    'time': time_str,
                    'symbol': 'NIFTY',
                    'option_type': otype,
                    'type': otype,
                    'strike_offset': 'ATM',
                    'open': round(price + np.random.normal(0, 1), 2),
                    'high': round(price + abs(np.random.normal(0, 2)), 2),
                    'low': round(price - abs(np.random.normal(0, 2)), 2),
                    'close': round(price, 2),
                    'volume': vol,
                    'oi': oi_val,
                    'iv': round(iv, 4),
                    'spot': round(spot, 2),
                })
    
    return pd.DataFrame(rows)


# UTILITY: Indian Number Formatting
def format_inr(value: float) -> str:
    """Format a number in Indian numbering system with ₹ symbol."""
    if abs(value) >= 10000000:  # 1 Cr
        return f"₹{value/10000000:.2f} Cr"
    elif abs(value) >= 100000:  # 1 Lakh
        return f"₹{value/100000:.2f} L"
    else:
        sign = '-' if value < 0 else ''
        value = abs(value)
        s = f"{value:,.2f}"
        return f"{sign}₹{s}"
