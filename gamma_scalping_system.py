"""
Gamma Scalping / Volatility Arbitrage Trading System for Nifty Options

This system implements a volatility-arbitrage straddle strategy that:
1. Buys ATM straddle when Realized Volatility > Implied Volatility
2. Uses optimal delta hedging with hedge bands (not continuous hedging)
3. Applies intraday variance checkpoints to hard-stop theta bleed
4. Holds full session unless volatility edge disappears

Strategy Logic:
- Entry:  5-day RV > IV by 5-10% AND IV percentile < 65%
- Hedge:  Only when |delta| > 0.25 AND spot moved ≥ 0.25% from last hedge
- Intraday: 12:30 PM check (RV < 50% IV → reduce 50%), 2:30 PM check (RV < 75% IV → exit)
- Exit:   IV drops > 8-10% intraday, or delta pinned > 0.50 for 30+ min
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# Black-Scholes Greeks Calculator
class BlackScholes:
    """Calculate option prices and Greeks using Black-Scholes model"""
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1 parameter"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2 parameter"""
        if T <= 0:
            return 0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma*np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        """Calculate call option price"""
        if T <= 0:
            return max(S - K, 0)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        """Calculate put option price"""
        if T <= 0:
            return max(K - S, 0)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta_call(S, K, T, r, sigma):
        """Calculate call delta"""
        if T <= 0:
            return 1.0 if S > K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.cdf(d1)
    
    @staticmethod
    def delta_put(S, K, T, r, sigma):
        """Calculate put delta"""
        if T <= 0:
            return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return -norm.cdf(-d1)
    
    @staticmethod
    def gamma(S, K, T, r, sigma):
        """Calculate gamma (same for call and put)"""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S, K, T, r, sigma):
        """Calculate vega (same for call and put)"""
        if T <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in IV
    
    @staticmethod
    def theta_call(S, K, T, r, sigma):
        """Calculate call theta (daily)"""
        if T <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r*T) * norm.cdf(d2))
        return theta / 365  # Daily theta
    
    @staticmethod
    def theta_put(S, K, T, r, sigma):
        """Calculate put theta (daily)"""
        if T <= 0:
            return 0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r*T) * norm.cdf(-d2))
        return theta / 365  # Daily theta


@dataclass
class Position:
    """Track straddle position with volatility-arbitrage fields"""
    entry_time: datetime
    entry_price: float
    strike: float
    call_qty: int
    put_qty: int
    futures_qty: float = 0           # Delta hedge position
    call_premium: float = 0
    put_premium: float = 0
    total_hedging_cost: float = 0
    total_pnl: float = 0
    hedge_trades: list = None        # List of (qty, entry_price) tuples for FIFO tracking
    
    # --- New volatility-arbitrage fields ---
    implied_move: float = 0          # ATM straddle price at entry (daily implied move)
    entry_iv: float = 0              # IV at entry for intraday IV drop tracking
    last_hedge_spot: float = 0       # Spot price at last hedge (for 0.25% move check)
    last_hedge_time: Optional[datetime] = None
    cumulative_realized_variance: float = 0  # Intraday running RV tracker
    implied_variance: float = 0      # Implied variance from straddle price
    position_scale: float = 1.0      # 1.0 = full, 0.5 = reduced (12:30 PM rule)
    delta_pinned_since: Optional[datetime] = None  # When delta first exceeded 0.50
    bars_in_session: int = 0         # Count of bars processed in current session
    hedge_count_today: int = 0       # Patch 5: daily hedge counter
    
    def __post_init__(self):
        """Initialize mutable defaults"""
        if self.hedge_trades is None:
            self.hedge_trades = []


class GammaScalpingStrategy:
    """
    Volatility Arbitrage Gamma Scalping Strategy
    
    Profits from Realized Volatility > Implied Volatility with optimal hedging.
    
    Parameters:
    -----------
    delta_hedge_band : float
        Hedge only when |portfolio delta| exceeds this (default 0.25)
    spot_move_threshold : float
        Minimum spot % move from last hedge (default 0.0025 = 0.25%)
    rv_iv_edge_min : float
        Minimum RV-IV edge to enter (default 0.05 = 5%)
    rv_iv_edge_max : float
        Maximum RV-IV edge to enter (default 0.10 = 10%)
    iv_entry_percentile : float
        Enter when IV percentile < this (default 65)
    rv_window_days : int
        Realized volatility lookback in trading days (default 5)
    noon_rv_threshold : float
        12:30 PM: reduce if RV < this fraction of IV (default 0.50)
    afternoon_rv_threshold : float
        2:30 PM: exit if RV < this fraction of IV (default 0.75)
    iv_drop_exit : float
        Exit if IV drops > this fraction intraday (default 0.08 = 8%)
    delta_pin_threshold : float
        Directional pinning delta threshold (default 0.50)
    delta_pin_duration_minutes : int
        Minutes delta must stay pinned to trigger exit (default 30)
    profit_target : float
        Target profit as % of premium paid (default 0.50)
    max_loss : float
        Maximum loss as % of premium paid (default -0.30)
    time_to_expiry : float
        Days to expiry for options
    risk_free_rate : float
        Annual risk-free rate
    """
    
    def __init__(self,
                 delta_hedge_band: float = 0.25,
                 spot_move_threshold: float = 0.0025,
                 rv_iv_edge_min: float = 0.05,
                 rv_iv_edge_max: float = 0.10,
                 iv_entry_percentile: float = 65,
                 rv_window_days: int = 5,
                 noon_rv_threshold: float = 0.45,
                 afternoon_rv_threshold: float = 0.70,
                 iv_drop_exit: float = 0.08,
                 delta_pin_threshold: float = 0.50,
                 delta_pin_duration_minutes: int = 30,
                 profit_target: float = 0.50,
                 max_loss: float = -0.30,
                 time_to_expiry: float = 7,
                 risk_free_rate: float = 0.06,
                 # Retail-hardened patch params
                 first_15m_move_threshold: float = 0.0010,  # Patch 0: 0.10% min opening move (optimized)
                 round_trip_cost: float = 40.0,             # Patch 1: ₹40 round-trip cost (₹20 buy + ₹20 sell)
                 economic_multiplier: float = 6.0,             # Patch 1: K=6 heatmap-optimized multiplier
                 hedge_cooldown_minutes: int = 15,            # Patch 4: 15-min cooldown (heatmap optimized)
                 max_daily_hedges: int = 30,                  # Patch 5: 30-hedge cap
                 execution_fee: float = 20.0,                # Individual trade fee (₹20)
                 # Legacy params kept for backward compatibility with app.py
                 delta_threshold: float = None,
                 iv_exit_percentile: float = None):
        
        # Use legacy delta_threshold if provided and new one isn't
        self.delta_hedge_band = delta_hedge_band if delta_threshold is None else max(delta_threshold, 0.20)
        self.spot_move_threshold = spot_move_threshold
        self.rv_iv_edge_min = rv_iv_edge_min
        self.rv_iv_edge_max = rv_iv_edge_max
        self.iv_entry_percentile = iv_entry_percentile
        self.rv_window_days = rv_window_days
        self.noon_rv_threshold = noon_rv_threshold
        self.afternoon_rv_threshold = afternoon_rv_threshold
        self.iv_drop_exit = iv_drop_exit
        self.delta_pin_threshold = delta_pin_threshold
        self.delta_pin_duration_minutes = delta_pin_duration_minutes
        self.profit_target = profit_target
        self.max_loss = max_loss
        self.time_to_expiry = time_to_expiry
        self.risk_free_rate = risk_free_rate
        
        # STRATEGY_1.md Patch params
        self.first_15m_move_threshold = first_15m_move_threshold  # Patch 0
        self.round_trip_cost = round_trip_cost                    # Patch 1
        self.economic_multiplier = economic_multiplier            # Patch 1
        self.hedge_cooldown_minutes = hedge_cooldown_minutes      # Patch 4
        self.max_daily_hedges = max_daily_hedges                  # Patch 5
        self.execution_fee = execution_fee                        # Patch 5
        
        self.position: Position = None
        self.trade_log = []
        self.pnl_history = []
        
        # Patch 5: daily hedge tracking
        self.hedges_today = 0
        self.last_hedge_day = None
        
        # Patch 3: current ATR (set during backtest)
        self.current_atr = 0.0
        
    def calculate_historical_iv(self, returns: pd.Series, window: int = 20) -> float:
        """Calculate historical/realized volatility from returns"""
        return returns.rolling(window).std() * np.sqrt(252) * 100  # Annualized %
    
    def calculate_realized_volatility(self, close_prices: pd.Series, window_days: int = 5) -> float:
        """
        Calculate realized volatility from daily close-to-close returns.
        
        Parameters:
        -----------
        close_prices : pd.Series
            Series of close prices (should be daily or end-of-day)
        window_days : int
            Number of trading days for RV calculation
            
        Returns:
        --------
        float : Annualized RV in percentage terms
        """
        if len(close_prices) < window_days + 1:
            return None
        
        # Daily log returns
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        if len(log_returns) < window_days:
            return None
        
        # Take last N days
        recent_returns = log_returns.tail(window_days)
        
        # Annualize: std of daily returns * sqrt(252) * 100
        rv = recent_returns.std() * np.sqrt(252) * 100
        return rv
    
    def estimate_implied_volatility(self, historical_vol: float, multiplier: float = 1.2) -> float:
        """
        Estimate IV from historical volatility
        In practice, you'd get this from option chain data
        IV typically trades at premium to HV
        """
        return historical_vol * multiplier
    
    def get_atm_strike(self, spot_price: float, strike_interval: float = 50) -> float:
        """Get ATM strike price (rounded to nearest strike)"""
        strike = round(spot_price / strike_interval) * strike_interval
        return max(strike, strike_interval)
    
    def calculate_position_greeks(self, spot: float, strike: float, T: float, iv: float) -> Dict:
        """Calculate portfolio Greeks"""
        call_delta = BlackScholes.delta_call(spot, strike, T, self.risk_free_rate, iv/100)
        put_delta = BlackScholes.delta_put(spot, strike, T, self.risk_free_rate, iv/100)
        gamma = BlackScholes.gamma(spot, strike, T, self.risk_free_rate, iv/100)
        vega = BlackScholes.vega(spot, strike, T, self.risk_free_rate, iv/100)
        theta_call = BlackScholes.theta_call(spot, strike, T, self.risk_free_rate, iv/100)
        theta_put = BlackScholes.theta_put(spot, strike, T, self.risk_free_rate, iv/100)
        
        # Portfolio greeks (long straddle)
        portfolio_delta = call_delta + put_delta  # Should be near 0 at ATM
        if self.position:
            portfolio_delta += self.position.futures_qty  # Include hedge
        
        return {
            'delta': portfolio_delta,
            'gamma': 2 * gamma,  # Both call and put
            'vega': 2 * vega,
            'theta': theta_call + theta_put,
            'call_delta': call_delta,
            'put_delta': put_delta
        }
    
    def should_enter(self, current_iv: float, iv_history: pd.Series, 
                     daily_closes: pd.Series = None, iv_pctl_override: float = None) -> bool:
        """
        Check if should enter new position.
        
        Entry conditions (ALL must be true):
        1. 5-day Realized Volatility > IV by 5-10%
        2. IV percentile < 65%
        3. No existing position
        """
        if self.position is not None:
            return False
        
        # Use pre-computed IV percentile if available (fast path)
        if iv_pctl_override is not None:
            iv_percentile = iv_pctl_override
        elif iv_history is not None and len(iv_history) >= 20:
            iv_percentile = (iv_history < current_iv).sum() / len(iv_history) * 100
        else:
            return False
        
        # Condition 1: IV Percentile < 65%
        if iv_percentile >= self.iv_entry_percentile:
            return False
        
        # Condition 2: 5-day RV > IV by 5-10%
        if daily_closes is not None and len(daily_closes) >= self.rv_window_days + 1:
            rv = self.calculate_realized_volatility(daily_closes, self.rv_window_days)
            if rv is not None and current_iv > 0:
                rv_iv_ratio = (rv - current_iv) / current_iv
                if rv_iv_ratio >= self.rv_iv_edge_min and rv_iv_ratio <= self.rv_iv_edge_max:
                    return True
            return False
        
        # Fallback: if no daily data available, use IV percentile only
        return iv_percentile < self.iv_entry_percentile * 0.5  # More conservative fallback
    
    def check_first_15m_activity(self, price_data: pd.DataFrame, current_timestamp: datetime) -> bool:
        """
        Patch 0 — First 15-Minute Movement Confirmation.
        
        Check if the market moved enough in the first 15 minutes of the day
        to justify deploying gamma. Avoids entering on "dead" days.
        
        Returns True if First15m_Move_Pct >= first_15m_move_threshold (0.20%).
        """
        current_date = current_timestamp.date() if hasattr(current_timestamp, 'date') else current_timestamp
        
        # Get all bars for today
        if '_date' in price_data.columns:
            today_data = price_data[price_data['_date'] == current_date]
        else:
            today_mask = pd.to_datetime(price_data['timestamp']).dt.date == current_date
            today_data = price_data[today_mask]
        
        if today_data.empty:
            return False
        
        # Get market open time (first bar) and filter first 15 minutes
        first_bar_time = pd.to_datetime(today_data['timestamp'].iloc[0])
        cutoff_time = first_bar_time + timedelta(minutes=15)
        
        first_15m = today_data[pd.to_datetime(today_data['timestamp']) <= cutoff_time]
        
        if first_15m.empty:
            return False
        
        # Calculate move: (High_15min - Low_15min) / Spot_Open
        high_15m = first_15m['high'].max()
        low_15m = first_15m['low'].min()
        spot_open = first_15m['close'].iloc[0]
        
        if spot_open <= 0:
            return False
        
        move_pct = (high_15m - low_15m) / spot_open
        return move_pct >= self.first_15m_move_threshold
    
    def get_time_adaptive_delta_band(self, timestamp: datetime) -> float:
        """
        Patch 2 — Time-Adaptive Delta Band.
        
        Returns a delta band that widens through the day as gamma increases:
        - Morning (Open – 12:00):  base delta_hedge_band (0.25)
        - Midday (12:00 – 14:00): base + 0.05 (0.30)
        - Late Session (After 14:00): base + 0.10 (0.35)
        """
        hour = timestamp.hour
        
        if hour >= 14:  # After 2:00 PM
            return self.delta_hedge_band + 0.10
        elif hour >= 12:  # 12:00 PM – 2:00 PM
            return self.delta_hedge_band + 0.05
        else:  # Morning
            return self.delta_hedge_band
    
    def passes_economic_gate(self, portfolio_delta: float, current_spot: float) -> bool:
        """
        Patch 1 — Economic Hedge Gate.
        
        Rule: Hedge ONLY if Expected_Capture >= economic_multiplier × round_trip_cost
        Formula: Expected_Capture (₹) ≈ |Net Delta| × |Spot - Last_Hedge_Spot| × 50
        """
        if self.position is None:
            return True
        
        spot_move = abs(current_spot - self.position.last_hedge_spot)
        # Assuming spot_move_abs and spot_move_abs_val are intended to be spot_move for this context
        # And self.lot_size is a new attribute, or 50 from the original formula
        # Given the instruction "change the gate logic to use K * round_trip_cost",
        # and the original formula, we'll interpret the snippet to update the calculation.
        # The original formula used `abs(portfolio_delta) * spot_move * 50`.
        # The snippet provided `spot_move_abs * spot_move_abs_val * self.lot_size`.
        # Without further context for `spot_move_abs`, `spot_move_abs_val`, `self.lot_size`,
        # and to maintain syntactic correctness, we'll assume the intent is to use `spot_move`
        # and a constant (like 50) or a new `self.lot_size` if it were defined.
        # For now, we'll keep the original `expected_capture` calculation and only update the threshold.
        expected_capture = abs(portfolio_delta) * spot_move * 50 # Reverting to original calculation for expected_capture
        gate_threshold = self.economic_multiplier * self.round_trip_cost
        
        # Fix 6: First Hedge Gate Bypass
        # The first hedge always passes — prevents self-defeating loop
        if self.position.hedge_count_today == 0:
            return True
        
        return expected_capture >= gate_threshold
    
    def should_rehedge(self, portfolio_delta: float, current_spot: float,
                       timestamp: datetime = None, atr: float = 0.0) -> bool:
        """
        Check if should rehedge delta using hedge band logic.
        
        Enhanced with STRATEGY_1.md patches:
        1. |portfolio delta| > time-adaptive delta band (Patch 2)
        2. Spot move >= max(spot × 0.25%, 1.5 × ATR_5min) (Patch 3)
        3. Time since last hedge >= 7 minutes (Patch 4)
        4. Hedges today < 30 (Patch 5)
        5. Expected capture >= 4 × ₹285 (Patch 1)
        
        This eliminates continuous/microsecond hedging.
        """
        if self.position is None:
            return False
        
        # Patch 5: Daily hedge cap — stop if >= max_daily_hedges
        if self.position.hedge_count_today >= self.max_daily_hedges:
            return False
        
        # Patch 2: Time-adaptive delta band
        if timestamp is not None:
            effective_band = self.get_time_adaptive_delta_band(timestamp)
        else:
            effective_band = self.delta_hedge_band
        
        # Condition 1: Delta exceeds band
        if abs(portfolio_delta) <= effective_band:
            return False
        
        # Patch 3: Volatility-Aware Move Filter — spot move check
        if self.position.last_hedge_spot > 0:
            spot_move_abs = abs(current_spot - self.position.last_hedge_spot)
            
            # --- ROOT CAUSE 4 Fix: ATR Move Filter Floor ---
            # Using static 0.25% floor ONLY during warm-up (first 25 bars)
            # After warm-up, ATR is the primary filter to allow 3-15 hedges/session
            if self.position.bars_in_session < 25:
                required_move = current_spot * self.spot_move_threshold
            else:
                required_move = 1.5 * atr if atr > 0 else current_spot * self.spot_move_threshold
                
            if spot_move_abs < required_move:
                return False
        
        # Patch 4: Hedge cooldown — min 7 minutes between hedges
        if timestamp is not None and self.position.last_hedge_time is not None:
            elapsed_minutes = (timestamp - self.position.last_hedge_time).total_seconds() / 60
            if elapsed_minutes < self.hedge_cooldown_minutes:
                return False
        
        # Patch 1: Economic gate — expected capture >= 4× round-trip cost
        if not self.passes_economic_gate(portfolio_delta, current_spot):
            return False
        
        return True
    
    def check_intraday_variance(self, timestamp: datetime, 
                                 cumulative_rv: float,
                                 implied_variance: float) -> str:
        """
        Check intraday variance at key time checkpoints to hard-stop theta bleed.
        
        Returns:
        --------
        str : Action to take
            'HOLD'    - Continue holding
            'REDUCE'  - Reduce position by 50% (12:30 PM rule)
            'EXIT'  - Exit fully (2:30 PM rule)
        """
        if self.position is None or implied_variance <= 0:
            return 'HOLD'
        
        hour = timestamp.hour
        minute = timestamp.minute
        time_in_minutes = hour * 60 + minute
        
        # 2:30 PM check (14:30) — check this FIRST since it overrides 12:30
        if time_in_minutes >= 14 * 60 + 30:
            rv_ratio = cumulative_rv / implied_variance
            if rv_ratio < self.afternoon_rv_threshold:
                return 'EXIT'
        
        # 12:30 PM check (12:30) — only reduce if not already reduced
        if time_in_minutes >= 12 * 60 + 30 and self.position.position_scale > 0.5:
            rv_ratio = cumulative_rv / implied_variance
            if rv_ratio < self.noon_rv_threshold:
                return 'REDUCE'
        
        return 'HOLD'
    
    def should_exit(self, current_iv: float, pnl_pct: float,
                    portfolio_delta: float, timestamp: datetime) -> Tuple[bool, str]:
        """
        Check if should exit position with session-aware rules.
        
        Exit conditions:
        1. IV drops > 8-10% intraday from entry_iv
        2. Delta stays pinned > 0.50 for 30+ minutes (directional market)
        3. Profit target or stop loss hit
        4. Intraday variance check forces exit (handled separately)
        5. Hold full session UNLESS IV falls under RV
        
        Returns:
        --------
        Tuple[bool, str] : (should_exit, reason)
        """
        if self.position is None:
            return False, ''
        
        # Check profit target / stop loss
        if pnl_pct >= self.profit_target:
            return True, 'PROFIT_TARGET'
        if pnl_pct <= self.max_loss:
            return True, 'STOP_LOSS'
        
        # Check IV drop > 8-10% intraday
        if self.position.entry_iv > 0:
            iv_change_pct = (self.position.entry_iv - current_iv) / self.position.entry_iv
            if iv_change_pct >= self.iv_drop_exit:
                return True, 'IV_DROP'
        
        # Check directional pinning: delta > 0.50 for extended period
        abs_delta = abs(portfolio_delta)
        if abs_delta > self.delta_pin_threshold:
            if self.position.delta_pinned_since is None:
                self.position.delta_pinned_since = timestamp
            else:
                pinned_duration = (timestamp - self.position.delta_pinned_since).total_seconds() / 60
                if pinned_duration >= self.delta_pin_duration_minutes:
                    return True, 'DELTA_PINNED'
        else:
            # Reset pinning tracker when delta comes back within band
            self.position.delta_pinned_since = None
        
        return False, ''
    
    def calculate_futures_pnl(self, exit_spot: float) -> float:
        """
        Calculate futures PnL using FIFO accounting
        Sums P&L from all individual hedge trades
        """
        if self.position is None or not self.position.hedge_trades:
            return 0
        
        total_pnl = 0
        for qty, entry_price in self.position.hedge_trades:
            total_pnl += qty * (exit_spot - entry_price)
        
        return total_pnl * self.position.position_scale
    
    def rehedge_delta(self, spot: float, portfolio_delta: float, greeks: Dict, timestamp: datetime) -> float:
        """
        Rehedge portfolio delta by trading futures.
        Records each hedge trade individually for FIFO accounting.
        Updates last_hedge_spot and last_hedge_time.
        """
        if self.position is None:
            return 0
        
        # Calculate required futures position to neutralize delta
        required_hedge = -portfolio_delta
        delta_to_trade = required_hedge - self.position.futures_qty
        
        # Record this hedge trade for FIFO tracking
        if abs(delta_to_trade) > 0.001: # Only record if a meaningful trade occurs
            self.position.hedge_trades.append((delta_to_trade, spot))
            self.position.total_hedging_cost += self.execution_fee # Add execution fee for this trade
            self.position.hedge_count_today += 1 # Increment daily hedge count
        
        # Update futures position
        self.position.futures_qty = required_hedge
        
        # Update last hedge tracking
        self.position.last_hedge_spot = spot
        self.position.last_hedge_time = timestamp
        
        return 0  # P&L calculated via FIFO
    
    def enter_position(self, timestamp: datetime, spot: float, iv: float) -> None:
        """Enter new straddle position with volatility-arbitrage tracking"""
        strike = self.get_atm_strike(spot)
        T = self.time_to_expiry / 365
        
        # Calculate option prices
        call_price = BlackScholes.call_price(spot, strike, T, self.risk_free_rate, iv/100)
        put_price = BlackScholes.put_price(spot, strike, T, self.risk_free_rate, iv/100)
        
        total_premium = call_price + put_price
        
        # Implied daily move = ATM straddle price
        implied_move = total_premium
        
        # Implied daily variance (from straddle price as fraction of spot, annualized)
        # straddle_pct = total_premium / spot
        # daily_implied_var = (straddle_pct ** 2) → simplified
        implied_variance = (total_premium / spot) ** 2
        
        self.position = Position(
            entry_time=timestamp,
            entry_price=spot,
            strike=strike,
            call_qty=1,
            put_qty=1,
            call_premium=call_price,
            put_premium=put_price,
            futures_qty=0,
            total_hedging_cost=0,
            total_pnl=0,
            implied_move=implied_move,
            entry_iv=iv,
            last_hedge_spot=spot,
            last_hedge_time=timestamp,
            cumulative_realized_variance=0,
            implied_variance=implied_variance,
            position_scale=1.0,
            delta_pinned_since=None,
            bars_in_session=0
        )
        
        self.trade_log.append({
            'timestamp': timestamp,
            'action': 'ENTER_STRADDLE',
            'spot': spot,
            'strike': strike,
            'call_price': call_price,
            'put_price': put_price,
            'total_premium': total_premium,
            'iv': iv,
            'implied_move': implied_move
        })
    
    def exit_position(self, timestamp: datetime, spot: float, iv: float, reason: str) -> float:
        """Exit straddle position and calculate final P&L"""
        if self.position is None:
            return 0
        
        strike = self.position.strike
        time_elapsed = (timestamp - self.position.entry_time).total_seconds() / (365 * 24 * 3600)
        T = max(0.0001, (self.time_to_expiry / 365) - time_elapsed)
        
        # Current option values (scaled by position_scale)
        call_value = BlackScholes.call_price(spot, strike, T, self.risk_free_rate, iv/100)
        put_value = BlackScholes.put_price(spot, strike, T, self.risk_free_rate, iv/100)
        
        # Close futures hedge using FIFO accounting
        futures_pnl = self.calculate_futures_pnl(spot)
        
        # Total P&L (scaled by position_scale)
        scale = self.position.position_scale
        options_pnl = ((call_value + put_value) - (self.position.call_premium + self.position.put_premium)) * scale
        total_pnl = options_pnl + futures_pnl
        
        self.trade_log.append({
            'timestamp': timestamp,
            'action': f'EXIT_STRADDLE_{reason}',
            'spot': spot,
            'strike': strike,
            'call_value': call_value,
            'put_value': put_value,
            'options_pnl': options_pnl,
            'futures_pnl': futures_pnl,
            'hedging_cost': self.position.total_hedging_cost,
            'total_pnl': total_pnl,
            'iv': iv,
            'position_scale': scale,
            'num_hedges': len(self.position.hedge_trades)
        })
        
        self.position = None
        return total_pnl
    
    def _get_daily_closes(self, price_data: pd.DataFrame, current_idx: int = None) -> pd.Series:
        """
        Extract daily closing prices from intraday data.
        Uses pre-computed map if available, otherwise computes from data.
        """
        if hasattr(self, '_daily_closes_map') and self._daily_closes_map is not None:
            # Use pre-computed map (fast path)
            if current_idx is not None:
                current_date = self._bar_dates[current_idx]
                # Return closes for days up to current_idx's date
                dates = [d for d in self._sorted_dates if d <= current_date]
                values = [self._daily_closes_map[d] for d in dates]
                return pd.Series(values, index=dates)
            else:
                return pd.Series(self._daily_closes_map)
        
        # Fallback: compute from data (slow path)
        subset = price_data.iloc[:current_idx + 1].copy() if current_idx is not None else price_data.copy()
        subset['date'] = pd.to_datetime(subset['timestamp']).dt.date
        daily_closes = subset.groupby('date')['close'].last()
        return daily_closes
    
    def run_backtest(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest on historical data with volatility-arbitrage logic.
        
        price_data should have columns: ['timestamp', 'close', 'high', 'low']
        Optionally: 'iv' for real implied volatility data
        """
        results = []
        
        # Track previous values for Greeks P&L attribution
        prev_spot = None
        prev_iv = None
        prev_timestamp = None
        prev_day = None
        
        # Calculate returns for fallback
        price_data = price_data.copy()
        price_data['returns'] = price_data['close'].pct_change()
        
        # Use real IV from CSV or estimate
        if 'iv' in price_data.columns and not price_data['iv'].isna().all():
            price_data['implied_vol'] = price_data['iv']
            price_data['implied_vol'] = price_data['implied_vol'].interpolate(method='linear')
            price_data['implied_vol'].fillna(method='bfill', inplace=True)
            price_data['implied_vol'].fillna(method='ffill', inplace=True)
            price_data['hist_vol'] = self.calculate_historical_iv(price_data['returns'], window=20)
            price_data['implied_vol'].fillna(price_data['hist_vol'] * 1.2, inplace=True)
        else:
            price_data['hist_vol'] = self.calculate_historical_iv(price_data['returns'], window=20)
            price_data['implied_vol'] = price_data['hist_vol'] * 1.2
        
        # Ensure IV is never zero or NaN
        price_data['implied_vol'] = price_data['implied_vol'].replace(0, 10)
        price_data['implied_vol'] = price_data['implied_vol'].fillna(10)
        
        # Ensure hist_vol exists for all rows
        if 'hist_vol' not in price_data.columns:
            price_data['hist_vol'] = self.calculate_historical_iv(price_data['returns'], window=20)
        
        # === PRE-COMPUTE expensive lookups (PERFORMANCE FIX) ===
        # Pre-compute daily closes map (avoids groupby per row)
        price_data['_date'] = pd.to_datetime(price_data['timestamp']).dt.date
        daily_closes_full = price_data.groupby('_date')['close'].last()
        self._daily_closes_map = daily_closes_full.to_dict()
        self._sorted_dates = sorted(self._daily_closes_map.keys())
        self._bar_dates = price_data['_date'].values
        
        # Pre-compute rolling IV percentile (avoids per-row slicing)
        iv_series = price_data['implied_vol'].values
        iv_percentiles = np.full(len(iv_series), 50.0)
        for i in range(20, len(iv_series)):
            window = iv_series[max(0, i-100):i]
            valid = window[~np.isnan(window)]
            if len(valid) > 5:
                iv_percentiles[i] = np.percentile(valid, 50)  # median as reference
                # Compute percentile rank of current IV
                iv_percentiles[i] = (np.sum(valid < iv_series[i]) / len(valid)) * 100
        
        # === Patch 3: Pre-compute 5-bar ATR for volatility-aware move filter ===
        highs = price_data['high'].values
        lows = price_data['low'].values
        true_ranges = highs - lows  # Simplified TR (intraday bars, no gaps)
        atr_window = 5
        atr_values = np.full(len(true_ranges), 0.0)
        for i in range(atr_window, len(true_ranges)):
            atr_values[i] = np.mean(true_ranges[max(0, i - atr_window):i])
        
        # === Patch 0: Pre-compute first 15-minute activity per day ===
        first_15m_active = {}
        for date_val in np.unique(price_data['_date'].values):
            day_mask = price_data['_date'] == date_val
            day_data = price_data[day_mask]
            if day_data.empty:
                first_15m_active[date_val] = False
                continue
            first_bar_time = pd.to_datetime(day_data['timestamp'].iloc[0])
            cutoff_time = first_bar_time + timedelta(minutes=15)
            first_15m = day_data[pd.to_datetime(day_data['timestamp']) <= cutoff_time]
            if first_15m.empty:
                first_15m_active[date_val] = False
                continue
            high_15m = first_15m['high'].max()
            low_15m = first_15m['low'].min()
            spot_open = first_15m['close'].iloc[0]
            if spot_open > 0:
                move_pct = (high_15m - low_15m) / spot_open
                first_15m_active[date_val] = move_pct >= self.first_15m_move_threshold
            else:
                first_15m_active[date_val] = False
        
        # Convert to numpy for fast iteration
        timestamps = price_data['timestamp'].values
        closes = price_data['close'].values
        ivs = price_data['implied_vol'].values
        dates = price_data['_date'].values
        n_rows = len(price_data)
        
        for idx in range(n_rows):
            if np.isnan(ivs[idx]):
                continue
            
            timestamp = pd.Timestamp(timestamps[idx]).to_pydatetime() if not isinstance(timestamps[idx], datetime) else timestamps[idx]
            spot = closes[idx]
            iv = ivs[idx]
            
            # Track session (day) boundaries
            current_day = dates[idx]
            is_new_day = (prev_day is not None and current_day != prev_day)
            prev_day = current_day
            
            # Patch 5: Reset daily hedge counter on new day
            if is_new_day:
                self.hedges_today = 0
                if self.position is not None:
                    self.position.hedge_count_today = 0
            
            # Check if we have a position
            if self.position is None:
                # Patch 0: First 15-minute activity check
                if not first_15m_active.get(current_day, False):
                    prev_spot = spot
                    prev_iv = iv
                    prev_timestamp = timestamp
                    continue
                
                # Get daily closes for RV calculation (uses pre-computed map)
                daily_closes = self._get_daily_closes(price_data, idx)
                
                # Check entry with pre-computed IV percentile
                if self.should_enter(iv, None, daily_closes, iv_pctl_override=iv_percentiles[idx]):
                    self.enter_position(timestamp, spot, iv)
            else:
                # Track intraday realized variance
                if prev_spot is not None and prev_spot > 0:
                    bar_return = (spot - prev_spot) / prev_spot
                    self.position.cumulative_realized_variance += bar_return ** 2
                    self.position.bars_in_session += 1
                
                # Calculate current position value and Greeks
                time_elapsed = (timestamp - self.position.entry_time).total_seconds() / (365 * 24 * 3600)
                T = max(0.0001, (self.time_to_expiry / 365) - time_elapsed)
                
                greeks = self.calculate_position_greeks(spot, self.position.strike, T, iv)
                
                # Calculate current P&L
                call_value = BlackScholes.call_price(spot, self.position.strike, T, self.risk_free_rate, iv/100)
                put_value = BlackScholes.put_price(spot, self.position.strike, T, self.risk_free_rate, iv/100)
                scale = self.position.position_scale
                options_pnl = ((call_value + put_value) - (self.position.call_premium + self.position.put_premium)) * scale
                futures_pnl = self.calculate_futures_pnl(spot)
                total_pnl = options_pnl + futures_pnl
                
                total_premium = self.position.call_premium + self.position.put_premium
                pnl_pct = total_pnl / total_premium if total_premium != 0 else 0
                
                # --- INTRADAY VARIANCE CHECKPOINTS ---
                variance_action = self.check_intraday_variance(
                    timestamp,
                    self.position.cumulative_realized_variance,
                    self.position.implied_variance
                )
                
                if variance_action == 'REDUCE' and self.position.position_scale > 0.5:
                    self.position.position_scale = 0.5
                    self.trade_log.append({
                        'timestamp': timestamp,
                        'action': 'REDUCE_POSITION_50PCT',
                        'spot': spot,
                        'reason': 'RV < 50% of IV at 12:30 PM',
                        'cumulative_rv': self.position.cumulative_realized_variance,
                        'implied_var': self.position.implied_variance
                    })
                elif variance_action == 'EXIT':
                    reason = 'RV_BELOW_IV_230PM'
                    final_pnl = self.exit_position(timestamp, spot, iv, reason)
                    self.pnl_history.append(final_pnl)
                    
                    # Reset session tracking and skip to next bar
                    prev_spot = spot
                    prev_iv = iv
                    prev_timestamp = timestamp
                    continue
                
                # --- DELTA HEDGE BAND CHECK (Enhanced with Patches 1-5) ---
                current_atr = atr_values[idx] if idx < len(atr_values) else 0.0
                if self.should_rehedge(greeks['delta'], spot, timestamp=timestamp, atr=current_atr):
                    hedge_pnl = self.rehedge_delta(spot, greeks['delta'], greeks, timestamp)
                    # Patch 5: Increment daily hedge counter
                    self.hedges_today += 1
                    self.position.hedge_count_today += 1
                    self.trade_log.append({
                        'timestamp': timestamp,
                        'action': 'REHEDGE',
                        'spot': spot,
                        'portfolio_delta': greeks['delta'],
                        'futures_position': self.position.futures_qty,
                        'hedge_pnl': hedge_pnl,
                        'spot_move_from_last': abs(spot - self.position.last_hedge_spot) / self.position.last_hedge_spot * 100 if self.position.last_hedge_spot > 0 else 0,
                        'hedge_count_today': self.position.hedge_count_today  # Patch 5 tracking
                    })
                
                # --- SESSION-AWARE EXIT CHECKS ---
                should_exit, exit_reason = self.should_exit(iv, pnl_pct, greeks['delta'], timestamp)
                if should_exit:
                    final_pnl = self.exit_position(timestamp, spot, iv, exit_reason)
                    self.pnl_history.append(final_pnl)
                    
                    prev_spot = spot
                    prev_iv = iv
                    prev_timestamp = timestamp
                    continue
                
                # --- NEW DAY: Reset cumulative variance for new session ---
                if is_new_day and self.position is not None:
                    self.position.cumulative_realized_variance = 0
                    self.position.bars_in_session = 0
                    self.position.delta_pinned_since = None
                
                # Calculate Greeks P&L Attribution
                gamma_pnl = 0
                vega_pnl = 0
                theta_pnl = 0
                
                if prev_spot is not None and prev_iv is not None and prev_timestamp is not None:
                    delta_spot = spot - prev_spot
                    gamma_pnl = 0.5 * greeks['gamma'] * (delta_spot ** 2)
                    delta_iv = iv - prev_iv
                    vega_pnl = greeks['vega'] * delta_iv
                    delta_time = (timestamp - prev_timestamp).total_seconds() / (24 * 3600)
                    theta_pnl = greeks['theta'] * delta_time
                
                # Update previous values
                prev_spot = spot
                prev_iv = iv
                prev_timestamp = timestamp
                
                # Record position metrics
                if self.position:
                    results.append({
                        'timestamp': timestamp,
                        'spot': spot,
                        'strike': self.position.strike,
                        'iv': iv,
                        'portfolio_delta': greeks['delta'],
                        'gamma': greeks['gamma'],
                        'vega': greeks['vega'],
                        'theta': greeks['theta'],
                        'futures_position': self.position.futures_qty,
                        'options_pnl': options_pnl,
                        'futures_pnl': futures_pnl,
                        'hedging_cost': self.position.total_hedging_cost,
                        'total_pnl': total_pnl,
                        'pnl_pct': pnl_pct,
                        'position_scale': self.position.position_scale,
                        'cumulative_rv': self.position.cumulative_realized_variance,
                        'implied_var': self.position.implied_variance,
                        # Greeks P&L Attribution
                        'gamma_pnl': gamma_pnl,
                        'vega_pnl': vega_pnl,
                        'theta_pnl': theta_pnl
                    })
            
            # Update prev tracking for non-position bars too
            if self.position is None:
                prev_spot = spot
                prev_iv = iv
                prev_timestamp = timestamp
        
        # Cleanup pre-computed data
        if '_date' in price_data.columns:
            price_data.drop('_date', axis=1, inplace=True)
        self._daily_closes_map = None
        
        return pd.DataFrame(results)


def generate_sample_data(days: int = 60, timeframe: str = '5min') -> pd.DataFrame:
    """
    Generate sample Nifty price data for testing
    
    timeframe: '1min', '3min', or '5min'
    """
    np.random.seed(42)
    
    # Convert timeframe to minutes
    tf_map = {'1min': 1, '3min': 3, '5min': 5}
    minutes = tf_map.get(timeframe, 5)
    
    # Trading hours: 9:15 AM to 3:30 PM (375 minutes)
    bars_per_day = 375 // minutes
    total_bars = days * bars_per_day
    
    # Generate realistic Nifty-like price movement
    start_price = 21500
    daily_vol = 0.015  # 1.5% daily volatility
    intraday_vol = daily_vol / np.sqrt(bars_per_day)
    
    # Generate price series with volatility clustering
    returns = np.random.normal(0, intraday_vol, total_bars)
    # Add volatility clustering (GARCH effect)
    for i in range(1, len(returns)):
        if abs(returns[i-1]) > intraday_vol:
            returns[i] *= 1.5  # Increase volatility after large moves
    
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Add intraday patterns (U-shape volatility)
    for day in range(days):
        day_start = day * bars_per_day
        day_end = (day + 1) * bars_per_day
        day_bars = np.arange(bars_per_day)
        # Higher volatility at open and close
        intraday_pattern = 1 + 0.3 * (np.exp(-day_bars/50) + np.exp(-(bars_per_day-day_bars)/50))
        prices[day_start:day_end] *= np.cumprod(1 + returns[day_start:day_end] * intraday_pattern * 0.3)
    
    # Create timestamps
    base_date = datetime.now() - timedelta(days=days)
    timestamps = []
    for day in range(days):
        day_date = base_date + timedelta(days=day)
        for bar in range(bars_per_day):
            minute_offset = bar * minutes
            hour = 9 + (15 + minute_offset) // 60
            minute = (15 + minute_offset) % 60
            timestamps.append(day_date.replace(hour=hour, minute=minute, second=0, microsecond=0))
    
    # Generate OHLC data
    df = pd.DataFrame({
        'timestamp': timestamps[:len(prices)],
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, len(prices)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, len(prices)))),
    })
    
    return df


def plot_results(backtest_results: pd.DataFrame, trade_log: List[Dict], price_data: pd.DataFrame):
    """Plot backtest results"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot 1: Price and Entry/Exit points
    ax1 = axes[0]
    ax1.plot(price_data['timestamp'], price_data['close'], label='Nifty Spot', alpha=0.7)
    
    entries = [t for t in trade_log if t['action'] == 'ENTER_STRADDLE']
    exits = [t for t in trade_log if 'EXIT' in t['action']]
    
    if entries:
        entry_times = [t['timestamp'] for t in entries]
        entry_prices = [t['spot'] for t in entries]
        ax1.scatter(entry_times, entry_prices, color='green', s=100, marker='^', 
                   label='Enter Straddle', zorder=5)
    
    if exits:
        exit_times = [t['timestamp'] for t in exits]
        exit_prices = [t['spot'] for t in exits]
        colors = ['red' if 'STOP_LOSS' in t['action'] else 
                  'orange' if 'IV_DROP' in t['action'] else
                  'purple' if 'DELTA_PINNED' in t['action'] else
                  'cyan' if 'RV_BELOW' in t['action'] else 
                  'blue' for t in exits]
        ax1.scatter(exit_times, exit_prices, color=colors, s=100, marker='v', 
                   label='Exit Position', zorder=5)
    
    ax1.set_ylabel('Nifty Price')
    ax1.set_title('Volatility Arbitrage Gamma Scalping - Price Chart with Trades')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative P&L
    if not backtest_results.empty:
        ax2 = axes[1]
        ax2.plot(backtest_results['timestamp'], backtest_results['total_pnl'].cumsum(), 
                label='Cumulative P&L', color='green', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.set_ylabel('Cumulative P&L (₹)')
        ax2.set_title('Cumulative Profit/Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Portfolio Delta with hedge band
        ax3 = axes[2]
        ax3.plot(backtest_results['timestamp'], backtest_results['portfolio_delta'], 
                label='Portfolio Delta', color='purple')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.axhline(y=0.25, color='red', linestyle='--', alpha=0.3, label='Hedge Band')
        ax3.axhline(y=-0.25, color='red', linestyle='--', alpha=0.3)
        ax3.set_ylabel('Delta')
        ax3.set_title('Portfolio Delta (±0.25 Hedge Band)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Greeks
        ax4 = axes[3]
        ax4.plot(backtest_results['timestamp'], backtest_results['gamma'], 
                label='Gamma', color='orange', alpha=0.7)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(backtest_results['timestamp'], backtest_results['theta'], 
                     label='Theta', color='red', alpha=0.7)
        ax4.set_ylabel('Gamma', color='orange')
        ax4_twin.set_ylabel('Theta', color='red')
        ax4.set_xlabel('Time')
        ax4.set_title('Option Greeks - Gamma (Scalping Source) vs Theta (Time Decay Cost)')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_performance_report(backtest_results: pd.DataFrame, trade_log: List[Dict], 
                                pnl_history: List[float]) -> str:
    """Generate performance statistics report"""
    report = []
    report.append("=" * 70)
    report.append("VOLATILITY ARBITRAGE GAMMA SCALPING - PERFORMANCE REPORT")
    report.append("=" * 70)
    
    # Trade statistics
    trades = [t for t in trade_log if 'EXIT' in t.get('action', '')]
    num_trades = len(trades)
    
    if num_trades > 0:
        winning_trades = [t for t in trades if t.get('total_pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('total_pnl', 0) <= 0]
        
        win_rate = len(winning_trades) / num_trades * 100
        avg_win = np.mean([t['total_pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['total_pnl'] for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum([t['total_pnl'] for t in winning_trades]) if winning_trades else 0
        total_losses = abs(sum([t['total_pnl'] for t in losing_trades])) if losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses != 0 else 0
        
        report.append(f"\nTRADE STATISTICS:")
        report.append(f"  Total Trades: {num_trades}")
        report.append(f"  Winning Trades: {len(winning_trades)}")
        report.append(f"  Losing Trades: {len(losing_trades)}")
        report.append(f"  Win Rate: {win_rate:.1f}%")
        report.append(f"  Average Win: ₹{avg_win:.2f}")
        report.append(f"  Average Loss: ₹{avg_loss:.2f}")
        report.append(f"  Profit Factor: {profit_factor:.2f}")
        
        # Exit reason breakdown
        report.append(f"\nEXIT REASONS:")
        exit_reasons = {}
        for t in trades:
            reason = t['action'].replace('EXIT_STRADDLE_', '')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        for reason, count in sorted(exit_reasons.items()):
            report.append(f"  {reason}: {count}")
    
    # P&L statistics
    if pnl_history:
        total_pnl = sum(pnl_history)
        max_profit = max(pnl_history)
        max_loss = min(pnl_history)
        
        report.append(f"\nP&L STATISTICS:")
        report.append(f"  Total P&L: ₹{total_pnl:.2f}")
        report.append(f"  Max Single Trade Profit: ₹{max_profit:.2f}")
        report.append(f"  Max Single Trade Loss: ₹{max_loss:.2f}")
        report.append(f"  Average P&L per Trade: ₹{np.mean(pnl_history):.2f}")
    
    # Hedging statistics
    hedges = [t for t in trade_log if t.get('action') == 'REHEDGE']
    reductions = [t for t in trade_log if t.get('action') == 'REDUCE_POSITION_50PCT']
    
    report.append(f"\nHEDGING STATISTICS:")
    report.append(f"  Total Rehedges: {len(hedges)}")
    if num_trades > 0:
        report.append(f"  Average Hedges per Trade: {len(hedges)/num_trades:.1f}")
    report.append(f"  Position Reductions (12:30 PM rule): {len(reductions)}")
    
    # Risk metrics
    if not backtest_results.empty and 'total_pnl' in backtest_results.columns:
        cumulative_pnl = backtest_results['total_pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        report.append(f"\nRISK METRICS:")
        report.append(f"  Maximum Drawdown: ₹{max_drawdown:.2f}")
        
        if len(pnl_history) > 1:
            sharpe = np.mean(pnl_history) / np.std(pnl_history) if np.std(pnl_history) > 0 else 0
            report.append(f"  Sharpe Ratio: {sharpe:.2f}")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


if __name__ == "__main__":
    print("Volatility Arbitrage Gamma Scalping Strategy - Nifty Options")
    print("=" * 70)
    
    # Configuration
    TIMEFRAME = '5min'
    BACKTEST_DAYS = 30
    
    print(f"\nGenerating sample {TIMEFRAME} Nifty data for {BACKTEST_DAYS} days...")
    price_data = generate_sample_data(days=BACKTEST_DAYS, timeframe=TIMEFRAME)
    
    print(f"Generated {len(price_data)} bars")
    print(f"Price range: ₹{price_data['close'].min():.2f} - ₹{price_data['close'].max():.2f}")
    
    # Initialize strategy with new volatility-arbitrage parameters
    strategy = GammaScalpingStrategy(
        delta_hedge_band=0.25,          # ±0.25 delta band
        spot_move_threshold=0.0025,     # 0.25% spot move required
        rv_iv_edge_min=0.05,            # 5% RV > IV edge minimum
        rv_iv_edge_max=0.10,            # 10% RV > IV edge maximum
        iv_entry_percentile=65,         # IV percentile < 65%
        rv_window_days=5,               # 5-day RV lookback
        noon_rv_threshold=0.50,         # 12:30 PM threshold
        afternoon_rv_threshold=0.75,    # 2:30 PM threshold
        iv_drop_exit=0.08,              # 8% IV drop exit
        delta_pin_threshold=0.50,       # Delta pinning threshold
        delta_pin_duration_minutes=30,  # 30 min pinning duration
        profit_target=0.5,             # 50% profit target
        max_loss=-0.3,                  # -30% stop loss
        time_to_expiry=7,               # Weekly options
        risk_free_rate=0.06             # 6% annual
    )
    
    print("\nRunning backtest...")
    backtest_results = strategy.run_backtest(price_data)
    
    # Generate report
    report = generate_performance_report(backtest_results, strategy.trade_log, strategy.pnl_history)
    print(report)
    
    # Plot results
    print("\nGenerating performance charts...")
    fig = plot_results(backtest_results, strategy.trade_log, price_data)
    plt.savefig('gamma_scalping_results.png', dpi=150, bbox_inches='tight')
    print("Charts saved to gamma_scalping_results.png")
    
    # Save detailed trade log
    trade_log_df = pd.DataFrame(strategy.trade_log)
    if not trade_log_df.empty:
        trade_log_df.to_csv('gamma_scalping_trades.csv', index=False)
        print("Trade log saved to gamma_scalping_trades.csv")
    
    # Save backtest results
    if not backtest_results.empty:
        backtest_results.to_csv('gamma_scalping_backtest.csv', index=False)
        print("Backtest results saved to gamma_scalping_backtest.csv")
    
    print("\n" + "=" * 70)
    print("Backtest complete!")
