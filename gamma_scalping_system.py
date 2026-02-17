"""
Gamma Scalping / Volatility Scalping Trading System for Nifty Options

This system implements a delta-neutral straddle strategy that:
1. Buys ATM straddle (call + put)
2. Dynamically hedges delta by trading futures
3. Profits when realized volatility > implied volatility
4. Scalps gamma through continuous delta hedging

Strategy Logic:
- Enter: Buy ATM straddle when IV is relatively low
- Hedge: Rebalance delta when it exceeds threshold
- Exit: Close position at target profit or max loss
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from dataclasses import dataclass

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
    """Track straddle position with realistic costs"""
    entry_time: datetime
    entry_price: float
    strike: float
    call_qty: int
    put_qty: int
    futures_qty: int = 0  # Delta hedge position
    call_premium: float = 0
    put_premium: float = 0
    total_hedging_cost: float = 0
    total_pnl: float = 0
    hedge_trades: list = None  # List of (qty, entry_price) tuples for FIFO tracking
    # Realistic cost tracking
    total_futures_commission: float = 0
    total_options_commission: float = 0
    total_slippage_cost: float = 0
    num_rehedges: int = 0
    
    def __post_init__(self):
        """Initialize mutable defaults"""
        if self.hedge_trades is None:
            self.hedge_trades = []


class GammaScalpingStrategy:
    """
    Gamma Scalping Strategy Implementation
    
    Parameters:
    -----------
    delta_threshold : float
        Rehedge when |portfolio delta| exceeds this (e.g., 0.15 = 15 delta)
    iv_entry_percentile : float
        Enter when IV is below this historical percentile (e.g., 30th percentile)
    iv_exit_percentile : float  
        Exit when IV exceeds this percentile (e.g., 70th percentile)
    profit_target : float
        Target profit as % of premium paid (e.g., 0.5 = 50%)
    max_loss : float
        Maximum loss as % of premium paid (e.g., -0.3 = -30%)
    time_to_expiry : float
        Days to expiry for options (e.g., 7 for weekly)
    risk_free_rate : float
        Annual risk-free rate for Greeks calculation
    futures_commission : float
        Commission per futures trade (₹ per lot)
    options_commission : float
        Commission per options trade (₹ per lot)
    slippage_pct : float
        Slippage as % of trade value (0.05-0.10%)
    """
    
    def __init__(self,
                 delta_threshold: float = 0.15,
                 iv_entry_percentile: float = 30,
                 iv_exit_percentile: float = 70,
                 profit_target: float = 0.5,
                 max_loss: float = -0.3,
                 time_to_expiry: float = 7,
                 risk_free_rate: float = 0.06,
                 futures_commission: float = 30.0,
                 options_commission: float = 75.0,
                 slippage_pct: float = 0.075):
        
        self.delta_threshold = delta_threshold
        self.iv_entry_percentile = iv_entry_percentile
        self.iv_exit_percentile = iv_exit_percentile
        self.profit_target = profit_target
        self.max_loss = max_loss
        self.time_to_expiry = time_to_expiry
        self.risk_free_rate = risk_free_rate
        
        # Realistic trading costs
        self.futures_commission = futures_commission  # ₹30 per lot typical
        self.options_commission = options_commission  # ₹75 per lot typical
        self.slippage_pct = slippage_pct / 100  # Convert to decimal (0.075% = 0.00075)
        
        self.position: Position = None
        self.trade_log = []
        self.pnl_history = []
        
    def calculate_historical_iv(self, returns: pd.Series, window: int = 20) -> float:
        """Calculate historical/realized volatility from returns"""
        return returns.rolling(window).std() * np.sqrt(252) * 100  # Annualized %
    
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
        # Ensure strike is never 0 (minimum is one strike interval)
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
    
    def should_enter(self, current_iv: float, iv_history: pd.Series) -> bool:
        """Check if should enter new position"""
        if self.position is not None:
            return False
        
        if len(iv_history) < 20:
            return False
        
        # Avoid division by zero
        if len(iv_history) == 0:
            return False
        
        iv_percentile = (iv_history < current_iv).sum() / len(iv_history) * 100
        return iv_percentile < self.iv_entry_percentile
    
    def should_exit(self, current_iv: float, iv_history: pd.Series, pnl_pct: float) -> bool:
        """Check if should exit position"""
        if self.position is None:
            return False
        
        # Exit on profit target or max loss
        if pnl_pct >= self.profit_target or pnl_pct <= self.max_loss:
            return True
        
        # Exit if IV spikes (volatility already realized)
        if len(iv_history) >= 20:
            # Avoid division by zero
            if len(iv_history) == 0:
                return False
            
            iv_percentile = (iv_history < current_iv).sum() / len(iv_history) * 100
            if iv_percentile > self.iv_exit_percentile:
                return True
        
        return False
    
    def should_rehedge(self, portfolio_delta: float) -> bool:
        """Check if should rehedge delta"""
        return abs(portfolio_delta) > self.delta_threshold
    
    def calculate_futures_pnl(self, exit_spot: float) -> float:
        """
        Calculate futures PnL using FIFO accounting
        Sums P&L from all individual hedge trades
        """
        if self.position is None or not self.position.hedge_trades:
            return 0
        
        total_pnl = 0
        for qty, entry_price in self.position.hedge_trades:
            # Each trade's PnL: quantity × (exit_price - entry_price)
            total_pnl += qty * (exit_spot - entry_price)
        
        return total_pnl
    
    def rehedge_delta(self, spot: float, portfolio_delta: float, greeks: Dict) -> float:
        """
        Rehedge portfolio delta by trading futures
        Records each hedge trade individually for FIFO accounting
        Returns hedging cost including commission and slippage
        """
        if self.position is None:
            return 0
        
        # Calculate required futures position to neutralize delta
        # Delta of 1 futures contract = 1
        required_hedge = -portfolio_delta  # Opposite sign to neutralize
        
        # Calculate change in hedge position
        delta_hedge = required_hedge - self.position.futures_qty
        
        # Calculate realistic costs for this rehedge
        hedge_cost = 0
        if abs(delta_hedge) > 0.001:  # Only record if meaningful change
            # Commission cost
            commission = self.futures_commission
            
            # Slippage cost (% of notional value)
            notional = abs(delta_hedge) * spot
            slippage = notional * self.slippage_pct
            
            # Total cost for this hedge
            hedge_cost = commission + slippage
            
            # Track costs in position
            self.position.total_futures_commission += commission
            self.position.total_slippage_cost += slippage
            self.position.num_rehedges += 1
            
            # Record this hedge trade for FIFO tracking
            self.position.hedge_trades.append((delta_hedge, spot))
        
        # Update futures position (FIFO accounting in calculate_futures_pnl handles P&L)
        self.position.futures_qty = required_hedge
        
        return -hedge_cost  # Negative because it's a cost
    
    def enter_position(self, timestamp: datetime, spot: float, iv: float) -> None:
        """Enter new straddle position with realistic costs"""
        strike = self.get_atm_strike(spot)
        T = self.time_to_expiry / 365  # Initial time to expiry
        
        # Calculate option prices (mid-price)
        call_price = BlackScholes.call_price(spot, strike, T, self.risk_free_rate, iv/100)
        put_price = BlackScholes.put_price(spot, strike, T, self.risk_free_rate, iv/100)
        
        # Add slippage to entry (buy at ask, which is higher than mid)
        call_slippage = call_price * self.slippage_pct
        put_slippage = put_price * self.slippage_pct
        
        # Actual entry prices including slippage
        call_entry = call_price + call_slippage
        put_entry = put_price + put_slippage
        
        # Options commission (per lot for both call and put)
        options_commission = 2 * self.options_commission  # 2x for call + put
        
        total_premium = call_entry + put_entry
        total_entry_slippage = call_slippage + put_slippage
        
        self.position = Position(
            entry_time=timestamp,
            entry_price=spot,
            strike=strike,
            call_qty=1,
            put_qty=1,
            call_premium=call_entry,  # Includes entry slippage
            put_premium=put_entry,    # Includes entry slippage
            futures_qty=0,
            total_hedging_cost=0,
            total_pnl=0,
            total_options_commission=options_commission,
            total_slippage_cost=total_entry_slippage
        )
        
        self.trade_log.append({
            'timestamp': timestamp,
            'action': 'ENTER_STRADDLE',
            'spot': spot,
            'strike': strike,
            'call_price': call_price,
            'put_price': put_price,
            'total_premium': total_premium,
            'entry_commission': options_commission,
            'entry_slippage': total_entry_slippage,
            'iv': iv
        })
    
    def exit_position(self, timestamp: datetime, spot: float, iv: float, reason: str) -> float:
        """Exit straddle position and calculate final P&L with realistic costs"""
        if self.position is None:
            return 0
        
        strike = self.position.strike
        # Calculate time decay using continuous intraday calculation
        time_elapsed = (timestamp - self.position.entry_time).total_seconds() / (365 * 24 * 3600)
        T = max(0.0001, (self.time_to_expiry / 365) - time_elapsed)
        
        # Current option values (mid-price)
        call_value = BlackScholes.call_price(spot, strike, T, self.risk_free_rate, iv/100)
        put_value = BlackScholes.put_price(spot, strike, T, self.risk_free_rate, iv/100)
        
        # Exit slippage (sell at bid, which is lower than mid)
        call_exit_slippage = call_value * self.slippage_pct
        put_exit_slippage = put_value * self.slippage_pct
        
        # Actual exit values after slippage
        call_exit = call_value - call_exit_slippage
        put_exit = put_value - put_exit_slippage
        
        # Exit commission
        exit_commission = 2 * self.options_commission  # 2x for call + put
        
        # Close futures hedge using FIFO accounting
        futures_pnl = self.calculate_futures_pnl(spot)
        
        # Add exit slippage to tracking
        exit_slippage = call_exit_slippage + put_exit_slippage
        self.position.total_slippage_cost += exit_slippage
        self.position.total_options_commission += exit_commission
        
        # Calculate P&L components
        options_pnl = (call_exit + put_exit) - (self.position.call_premium + self.position.put_premium)
        
        # Total costs
        total_costs = (self.position.total_futures_commission + 
                      self.position.total_options_commission + 
                      self.position.total_slippage_cost)
        
        # Net P&L after all costs
        total_pnl = options_pnl + futures_pnl - total_costs
        
        self.trade_log.append({
            'timestamp': timestamp,
            'action': f'EXIT_STRADDLE_{reason}',
            'spot': spot,
            'strike': strike,
            'call_value': call_value,
            'put_value': put_value,
            'options_pnl': options_pnl,
            'futures_pnl': futures_pnl,
            'futures_commission': self.position.total_futures_commission,
            'options_commission': self.position.total_options_commission,
            'total_slippage': self.position.total_slippage_cost,
            'total_costs': total_costs,
            'num_rehedges': self.position.num_rehedges,
            'total_pnl': total_pnl,
            'iv': iv
        })
        
        self.position = None
        return total_pnl
    
    def run_backtest(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest on historical data
        
        price_data should have columns: ['timestamp', 'close', 'high', 'low']
        """
        results = []
        
        # Track previous values for Greeks P&L attribution
        prev_spot = None
        prev_iv = None
        prev_timestamp = None
        
        # Calculate returns for fallback
        price_data['returns'] = price_data['close'].pct_change()
        
        # **CRITICAL FIX #1: Use real IV from CSV data**
        if 'iv' in price_data.columns and not price_data['iv'].isna().all():
            # Use real IV from CSV (primary source)
            price_data['implied_vol'] = price_data['iv']
            
            # Fill missing values with interpolation
            price_data['implied_vol'] = price_data['implied_vol'].interpolate(method='linear')
            price_data['implied_vol'].fillna(method='bfill', inplace=True)
            price_data['implied_vol'].fillna(method='ffill', inplace=True)
            
            # Fallback to historical vol only if still NaN
            price_data['hist_vol'] = self.calculate_historical_iv(price_data['returns'], window=20)
            price_data['implied_vol'].fillna(price_data['hist_vol'] * 1.2, inplace=True)
        else:
            # Fallback: calculate historical volatility (only if no IV column)
            price_data['hist_vol'] = self.calculate_historical_iv(price_data['returns'], window=20)
            price_data['implied_vol'] = price_data['hist_vol'] * 1.2  # Simplified IV estimate
        
        # Ensure IV is never zero or NaN (which would cause division by zero in Greeks)
        # Replace zero/NaN with a minimum viable IV (10%)
        price_data['implied_vol'] = price_data['implied_vol'].replace(0, 10)
        price_data['implied_vol'] = price_data['implied_vol'].fillna(10)
        
        for idx, row in price_data.iterrows():
            if pd.isna(row['hist_vol']) or pd.isna(row['implied_vol']):
                continue
            
            timestamp = row['timestamp']
            spot = row['close']
            iv = row['implied_vol']
            
            # Check if we have a position
            if self.position is None:
                # Check entry conditions
                iv_history = price_data.loc[:idx, 'implied_vol'].dropna()
                if self.should_enter(iv, iv_history):
                    self.enter_position(timestamp, spot, iv)
            else:
                # Calculate current position value and Greeks with continuous time decay
                time_elapsed = (timestamp - self.position.entry_time).total_seconds() / (365 * 24 * 3600)
                T = max(0.0001, (self.time_to_expiry / 365) - time_elapsed)
                
                greeks = self.calculate_position_greeks(spot, self.position.strike, T, iv)
                
                # Calculate current P&L
                call_value = BlackScholes.call_price(spot, self.position.strike, T, self.risk_free_rate, iv/100)
                put_value = BlackScholes.put_price(spot, self.position.strike, T, self.risk_free_rate, iv/100)
                options_pnl = (call_value + put_value) - (self.position.call_premium + self.position.put_premium)
                futures_pnl = self.calculate_futures_pnl(spot)  # Use FIFO accounting
                total_pnl = options_pnl + futures_pnl  # FIFO already includes all hedge economics
                
                # Avoid division by zero - if premium is 0, treat as 0% P&L
                total_premium = self.position.call_premium + self.position.put_premium
                pnl_pct = total_pnl / total_premium if total_premium != 0 else 0
                
                # Check if need to rehedge
                if self.should_rehedge(greeks['delta']):
                    hedge_pnl = self.rehedge_delta(spot, greeks['delta'], greeks)
                    self.trade_log.append({
                        'timestamp': timestamp,
                        'action': 'REHEDGE',
                        'spot': spot,
                        'portfolio_delta': greeks['delta'],
                        'futures_position': self.position.futures_qty,
                        'hedge_pnl': hedge_pnl
                    })
                
                # Check exit conditions
                iv_history = price_data.loc[:idx, 'implied_vol'].dropna()
                if self.should_exit(iv, iv_history, pnl_pct):
                    reason = 'PROFIT_TARGET' if pnl_pct >= self.profit_target else \
                             'STOP_LOSS' if pnl_pct <= self.max_loss else 'IV_SPIKE'
                    final_pnl = self.exit_position(timestamp, spot, iv, reason)
                    self.pnl_history.append(final_pnl)
                
                # Calculate Greeks P&L Attribution
                gamma_pnl = 0
                vega_pnl = 0
                theta_pnl = 0
                
                if prev_spot is not None and prev_iv is not None and prev_timestamp is not None:
                    # Gamma P&L: 0.5 × gamma × (ΔS)^2
                    delta_spot = spot - prev_spot
                    gamma_pnl = 0.5 * greeks['gamma'] * (delta_spot ** 2)
                    
                    # Vega P&L: vega × ΔIV
                    delta_iv = iv - prev_iv
                    vega_pnl = greeks['vega'] * delta_iv
                    
                    # Theta P&L: theta × Δt (time in days)
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
                        # Greeks P&L Attribution
                        'gamma_pnl': gamma_pnl,
                        'vega_pnl': vega_pnl,
                        'theta_pnl': theta_pnl
                    })
        
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
        colors = ['red' if 'STOP_LOSS' in t['action'] else 'blue' for t in exits]
        ax1.scatter(exit_times, exit_prices, color=colors, s=100, marker='v', 
                   label='Exit Position', zorder=5)
    
    ax1.set_ylabel('Nifty Price')
    ax1.set_title('Gamma Scalping Strategy - Price Chart with Trades')
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
        
        # Plot 3: Portfolio Delta
        ax3 = axes[2]
        ax3.plot(backtest_results['timestamp'], backtest_results['portfolio_delta'], 
                label='Portfolio Delta', color='purple')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax3.axhline(y=0.15, color='red', linestyle='--', alpha=0.3, label='Rehedge Threshold')
        ax3.axhline(y=-0.15, color='red', linestyle='--', alpha=0.3)
        ax3.set_ylabel('Delta')
        ax3.set_title('Portfolio Delta (Delta-Neutral Target)')
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
    """Generate performance statistics report with cost breakdown"""
    report = []
    report.append("=" * 70)
    report.append("GAMMA SCALPING STRATEGY - PERFORMANCE REPORT")
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
        
        # Correct profit factor: sum(profits) / abs(sum(losses))
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
    
    # Cost breakdown statistics
    if trades:
        total_futures_comm = sum([t.get('futures_commission', 0) for t in trades])
        total_options_comm = sum([t.get('options_commission', 0) for t in trades])
        total_slippage = sum([t.get('total_slippage', 0) for t in trades])
        total_costs = total_futures_comm + total_options_comm + total_slippage
        total_rehedges = sum([t.get('num_rehedges', 0) for t in trades])
        
        report.append(f"\nCOST BREAKDOWN (Realistic Trading Costs):")
        report.append(f"  Total Futures Commission: ₹{total_futures_comm:.2f}")
        report.append(f"  Total Options Commission: ₹{total_options_comm:.2f}")
        report.append(f"  Total Slippage Cost: ₹{total_slippage:.2f}")
        report.append(f"  ---")
        report.append(f"  Total Trading Costs: ₹{total_costs:.2f}")
        report.append(f"  Average Cost per Trade: ₹{total_costs/num_trades:.2f}")
        if total_rehedges > 0:
            report.append(f"  Average Cost per Rehedge: ₹{total_futures_comm/total_rehedges:.2f}")
    
    # Hedging statistics
    hedges = [t for t in trade_log if t.get('action') == 'REHEDGE']
    if hedges:
        report.append(f"\nHEDGING STATISTICS:")
        report.append(f"  Total Rehedges: {len(hedges)}")
        report.append(f"  Average Hedges per Trade: {len(hedges)/num_trades:.1f}")
    
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
    print("Gamma Scalping Strategy - Nifty Options")
    print("=" * 70)
    
    # Configuration
    TIMEFRAME = '5min'  # Options: '1min', '3min', '5min'
    BACKTEST_DAYS = 30
    
    print(f"\nGenerating sample {TIMEFRAME} Nifty data for {BACKTEST_DAYS} days...")
    price_data = generate_sample_data(days=BACKTEST_DAYS, timeframe=TIMEFRAME)
    
    print(f"Generated {len(price_data)} bars")
    print(f"Price range: ₹{price_data['close'].min():.2f} - ₹{price_data['close'].max():.2f}")
    
    # Initialize strategy
    strategy = GammaScalpingStrategy(
        delta_threshold=0.15,        # Rehedge when delta > 0.15
        iv_entry_percentile=30,      # Enter when IV in bottom 30%
        iv_exit_percentile=70,       # Exit when IV in top 70%
        profit_target=0.5,           # Exit at 50% profit
        max_loss=-0.3,               # Stop loss at -30%
        time_to_expiry=7,            # Weekly options
        risk_free_rate=0.06          # 6% annual
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
