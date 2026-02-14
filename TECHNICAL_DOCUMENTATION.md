# Gamma Scalping Strategy - Technical Documentation

## Table of Contents
1. [Strategy Overview](#strategy-overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Black-Scholes Model](#black-scholes-model)
4. [Options Greeks](#options-greeks)
5. [Strategy Logic](#strategy-logic)
6. [P&L Calculations](#pl-calculations)
7. [Risk Management](#risk-management)
8. [Implementation Details](#implementation-details)

---

## 🚨 Critical Production-Ready Fixes (February 2026)

This implementation includes three critical fixes that transform the strategy from an academic model to a production-ready trading system:

### ✅ Fix #1: Realistic IV from CSV Data
**Problem:** Previously used synthetic IV (`HV × 1.2`), creating artificial closed-loop dynamics.  
**Solution:** Now uses **real market IV from CSV data** with interpolation for missing values.  
**Impact:** Eliminates fake profits, enables realistic IV spikes/crushes, production-ready for live data.

### ✅ Fix #2: FIFO Futures PnL Tracking  
**Problem:** Incorrect PnL calculation using single entry price despite multiple rehedges.  
**Solution:** **FIFO accounting** tracking each individual hedge trade with its price.  
**Impact:** Accurate PnL calculation across multiple rehedging events.

### ✅ Fix #3: Continuous Intraday Time Decay
**Problem:** Used `.days` causing no theta decay within same day on 5-min bars.  
**Solution:** Changed to **`.total_seconds()`** for continuous decay calculation.  
**Impact:** Proper theta decay on intraday (5-minute) bars, accurate option pricing.

**Modified Files:**
- `gamma_scalping_system.py`: Core strategy logic updated with all three fixes
- Position tracking now includes `hedge_trades` list for FIFO
- Added `calculate_futures_pnl()` method for proper accounting

---

## Strategy Overview

### What is Gamma Scalping?

**Gamma scalping** (also called **volatility scalping**) is a market-neutral options trading strategy that profits from:
- **Realized volatility** being higher than **implied volatility**
- Dynamic delta hedging that "scalps gamma"
- Movement in the underlying asset

### Core Principle

```
Profit = Realized Volatility - Implied Volatility - Transaction Costs - Theta Decay
```

### Strategy Flow

```
Entry → Hold Straddle → Continuous Delta Hedging → Exit
  ↓           ↓                    ↓                  ↓
 Low IV   Monitor      Rehedge when delta     Profit/Loss/IV
          Position      exceeds threshold       threshold
```

---

## Mathematical Foundations

### 1. Historical Volatility (Realized Volatility)

Historical volatility measures the actual price movement over a period:

```
σ_hist = √(252) × std(returns)
```

Where:
- `returns = ln(P_t / P_(t-1))` (log returns)
- `252` = annualization factor (trading days per year)
- `std()` = standard deviation function

**Implementation:**
```python
def calculate_historical_iv(self, returns: pd.Series, window: int = 20):
    rolling_vol = returns.rolling(window=window).std()
    return rolling_vol * np.sqrt(252)  # Annualize
```

### 2. Implied Volatility (IV)

**⚠️ CRITICAL UPDATE:** The system now uses **real market IV from CSV data** instead of synthetic estimation.

Implied volatility represents the market's expectation of future volatility. Our implementation prioritizes real IV data:

**Primary Source:**
```python
# Use real IV from CSV data (column: 'iv')
if 'iv' in price_data.columns:
    price_data['implied_vol'] = price_data['iv']
    # Interpolate missing values
    price_data['implied_vol'] = price_data['implied_vol'].interpolate(method='linear')
```

**Fallback Only:**
```python
# Only used when real IV data is unavailable
IV_fallback = Historical_Volatility × 1.2
```

**Why This Matters:**
- ✅ Real market dynamics (IV spikes, crushes, regime changes)
- ✅ Actual smile/skew effects
- ✅ Eliminates artificial closed-loop profits
- ✅ Production-ready for live market data

---

## Black-Scholes Model

The Black-Scholes model is used to price European options and calculate Greeks.

### Core Parameters

- **S** = Spot price (current price of underlying)
- **K** = Strike price
- **T** = Time to expiry (in years, e.g., 7 days = 7/365)
- **r** = Risk-free rate (annual, e.g., 0.06 for 6%)
- **σ** = Implied volatility (annual)

### Black-Scholes Formula

#### d₁ and d₂ Parameters

```
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)

d₂ = d₁ - σ√T
```

**Implementation:**
```python
def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma*np.sqrt(T)
```

#### Call Option Price

```
C = S × N(d₁) - K × e^(-rT) × N(d₂)
```

Where:
- `N(x)` = Cumulative normal distribution function
- `e^(-rT)` = Present value discount factor

**Implementation:**
```python
def call_price(S, K, T, r, sigma):
    d1 = BlackScholes.d1(S, K, T, r, sigma)
    d2 = BlackScholes.d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
```

#### Put Option Price

```
P = K × e^(-rT) × N(-d₂) - S × N(-d₁)
```

**Implementation:**
```python
def put_price(S, K, T, r, sigma):
    d1 = BlackScholes.d1(S, K, T, r, sigma)
    d2 = BlackScholes.d2(S, K, T, r, sigma)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
```

---

## Options Greeks

Greeks measure the sensitivity of option prices to various factors.

### 1. Delta (Δ)

**Delta** measures the rate of change of option price with respect to the underlying price.

#### Call Delta
```
Δ_call = N(d₁)
```

Range: 0 to 1
- ATM call: ~0.5
- Deep ITM: approaches 1
- Deep OTM: approaches 0

**Implementation:**
```python
def delta_call(S, K, T, r, sigma):
    d1 = BlackScholes.d1(S, K, T, r, sigma)
    return norm.cdf(d1)  # N(d₁)
```

#### Put Delta
```
Δ_put = N(d₁) - 1 = -N(-d₁)
```

Range: -1 to 0
- ATM put: ~-0.5
- Deep ITM: approaches -1
- Deep OTM: approaches 0

**Implementation:**
```python
def delta_put(S, K, T, r, sigma):
    d1 = BlackScholes.d1(S, K, T, r, sigma)
    return -norm.cdf(-d1)  # -(1 - N(d₁))
```

#### Straddle Delta

For an ATM straddle (1 call + 1 put at same strike):
```
Δ_straddle = Δ_call + Δ_put ≈ 0.5 + (-0.5) = 0
```

This is why straddles are **delta-neutral** at entry.

### 2. Gamma (Γ)

**Gamma** measures the rate of change of delta with respect to the underlying price. Same for calls and puts.

```
Γ = φ(d₁) / (S × σ × √T)
```

Where:
- `φ(x)` = Standard normal probability density function
- `φ(d₁) = (1/√(2π)) × e^(-d₁²/2)`

**Key Properties:**
- Always positive for long options
- Highest for ATM options
- Decreases as option moves ITM or OTM
- Decreases as expiry approaches

**Implementation:**
```python
def gamma(S, K, T, r, sigma):
    d1 = BlackScholes.d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))
```

**Why Gamma Matters:**
- Positive gamma means delta increases when price rises, decreases when price falls
- This creates profit opportunity through delta hedging
- "Scalping gamma" = profiting from this dynamic delta change

### 3. Vega (ν)

**Vega** measures sensitivity to volatility changes. Same for calls and puts.

```
ν = S × φ(d₁) × √T / 100
```

Divided by 100 to express per 1% change in IV.

**Key Properties:**
- Always positive for long options
- Highest for ATM options
- Long straddle = positive vega (benefits from IV increase)

**Implementation:**
```python
def vega(S, K, T, r, sigma):
    d1 = BlackScholes.d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100
```

### 4. Theta (Θ)

**Theta** measures time decay (change in option value per day).

#### Call Theta
```
Θ_call = [-S × φ(d₁) × σ / (2√T) - r × K × e^(-rT) × N(d₂)] / 365
```

**Implementation:**
```python
def theta_call(S, K, T, r, sigma):
    d1 = BlackScholes.d1(S, K, T, r, sigma)
    d2 = BlackScholes.d2(S, K, T, r, sigma)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r*T) * norm.cdf(d2))
    return theta / 365  # Convert to daily
```

#### Put Theta
```
Θ_put = [-S × φ(d₁) × σ / (2√T) + r × K × e^(-rT) × N(-d₂)] / 365
```

**Implementation:**
```python
def theta_put(S, K, T, r, sigma):
    d1 = BlackScholes.d1(S, K, T, r, sigma)
    d2 = BlackScholes.d2(S, K, T, r, sigma)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             + r * K * np.exp(-r*T) * norm.cdf(-d2))
    return theta / 365  # Convert to daily
```

**Key Properties:**
- Almost always negative for long options (time decay is a cost)
- Accelerates as expiry approaches
- Must be overcome by gamma scalping profits

### Portfolio Greeks for Straddle

For a straddle (1 ATM call + 1 ATM put):

```python
# Portfolio Greeks
call_delta = BlackScholes.delta_call(S, K, T, r, iv)
put_delta = BlackScholes.delta_put(S, K, T, r, iv)
portfolio_delta = call_delta + put_delta  # ≈ 0 at entry

gamma = BlackScholes.gamma(S, K, T, r, iv)  # Same for both
portfolio_gamma = 2 × gamma  # Double (call + put)

vega = BlackScholes.vega(S, K, T, r, iv)
portfolio_vega = 2 × vega

theta_call = BlackScholes.theta_call(S, K, T, r, iv)
theta_put = BlackScholes.theta_put(S, K, T, r, iv)
portfolio_theta = theta_call + theta_put  # Negative (cost)
```

---

## Strategy Logic

### 1. Entry Conditions

Enter when implied volatility is relatively low:

```python
def should_enter(self, current_iv: float, iv_history: pd.Series):
    if len(iv_history) < 20:  # Need history
        return False
    
    iv_percentile = (iv_history < current_iv).mean() * 100
    
    # Enter when IV is below the entry percentile (e.g., 30th percentile)
    return iv_percentile < self.iv_entry_percentile
```

**Logic:**
- Calculate where current IV ranks in historical distribution
- If IV is below 30th percentile → IV is "cheap" → good time to buy options
- Buying options when IV is low minimizes initial cost

### 2. Position Entry

**⚠️ UPDATED:** Now includes FIFO futures tracking initialization.

```python
def enter_position(self, timestamp, spot, iv):
    # 1. Determine ATM strike
    strike = self.get_atm_strike(spot, strike_interval=50)
    
    # 2. Calculate option premiums with continuous time decay
    T = self.time_to_expiry / 365  # Initial time to expiry
    call_premium = BlackScholes.call_price(spot, strike, T, self.risk_free_rate, iv/100)
    put_premium = BlackScholes.put_price(spot, strike, T, self.risk_free_rate, iv/100)
    
    # 3. Create position with FIFO tracking
    self.position = Position(
        entry_time=timestamp,
        entry_price=spot,
        strike=strike,
        call_qty=1,
        put_qty=1,
        call_premium=call_premium,
        put_premium=put_premium,
        hedge_trades=[]  # FIFO tracking for futures
    )
    
    total_premium = call_premium + put_premium
    # Log entry
```

### 3. Delta Hedging (Rehedging)

This is the core of gamma scalping!

#### When to Rehedge

```python
def should_rehedge(self, portfolio_delta):
    return abs(portfolio_delta) > self.delta_threshold
```

Example: If `delta_threshold = 0.15`, rehedge when `|delta| > 0.15`

#### How to Rehedge

**⚠️ CRITICAL UPDATE:** Now uses FIFO accounting to track each hedge trade individually.

```python
def rehedge_delta(self, spot, portfolio_delta, greeks):
    # Delta of futures = 1 (perfectly tracks spot)
    required_hedge = -portfolio_delta  # Opposite sign to neutralize
    
    # Calculate change in hedge position
    delta_hedge = required_hedge - self.position.futures_qty
    
    # **FIFO TRACKING: Record each trade individually**
    if abs(delta_hedge) > 0.001:  # Only record meaningful changes
        self.position.hedge_trades.append((delta_hedge, spot))
    
    # Hedging cost calculation
    hedging_pnl = -delta_hedge * spot
    
    # Update position
    self.position.futures_qty = required_hedge
    self.position.total_hedging_cost += hedging_pnl
    
    return hedging_pnl
```

**FIFO PnL Calculation:**
```python
def calculate_futures_pnl(self, exit_spot):
    """Calculate futures P&L using FIFO accounting"""
    total_pnl = 0
    for qty, entry_price in self.position.hedge_trades:
        # Each trade's P&L: quantity × (exit_price - entry_price)
        total_pnl += qty * (exit_spot - entry_price)
    return total_pnl
```

**The Magic of Gamma Scalping:**

1. **Price rises** → Delta becomes positive → Sell futures (sell high)
2. **Price falls** → Delta becomes negative → Buy futures (buy low)
3. **Repeat** → Automatically buy low, sell high!

This only works because **gamma changes delta** as price moves.

### 4. Exit Conditions

```python
def should_exit(self, current_iv, iv_history, pnl_pct):
    # 1. Profit target reached
    if pnl_pct >= self.profit_target:
        return True, "PROFIT_TARGET"
    
    # 2. Stop loss hit
    if pnl_pct <= self.max_loss:
        return True, "STOP_LOSS"
    
    # 3. IV spike (volatility already realized)
    if len(iv_history) >= 20:
        iv_percentile = (iv_history < current_iv).mean() * 100
        if iv_percentile > self.iv_exit_percentile:  # e.g., 70th percentile
            return True, "IV_HIGH"
    
    return False, None
```

### 5. Position Exit

**⚠️ CRITICAL UPDATES:** 
1. Continuous intraday time decay using `.total_seconds()`
2. FIFO futures PnL calculation

```python
def exit_position(self, timestamp, spot, iv, reason):
    # 1. Calculate remaining time with CONTINUOUS decay (not daily steps)
    time_elapsed = (timestamp - self.position.entry_time).total_seconds() / (365 * 24 * 3600)
    T = max(0.0001, (self.time_to_expiry / 365) - time_elapsed)
    strike = self.position.strike
    
    # 2. Calculate current option values
    call_value = BlackScholes.call_price(spot, strike, T, self.risk_free_rate, iv/100)
    put_value = BlackScholes.put_price(spot, strike, T, self.risk_free_rate, iv/100)
    
    # 3. Calculate options P&L
    options_pnl = (call_value + put_value) - (self.position.call_premium + self.position.put_premium)
    
    # 4. Calculate futures P&L using FIFO accounting
    futures_pnl = self.calculate_futures_pnl(spot)  # ✅ Correct FIFO
    
    # 5. Total P&L
    total_pnl = options_pnl + futures_pnl + self.position.total_hedging_cost
    
    # 6. Log and clear position
    self.position = None
    
    return total_pnl
```

---

## P&L Calculations

### Components of P&L

```
Total P&L = Options P&L + Futures P&L - Hedging Costs
```

#### 1. Options P&L

```
Options P&L = (Call_Exit_Value - Call_Entry_Premium) + 
              (Put_Exit_Value - Put_Entry_Premium)
```

This captures:
- Intrinsic value changes
- Time decay (theta cost)
- Volatility changes (vega profit/loss)

#### 2. Futures P&L

**⚠️ CRITICAL FIX:** Now uses FIFO accounting for multiple hedge trades.

**OLD (INCORRECT):**
```
❌ Futures P&L = Futures_Quantity × (Current_Price - Single_Entry_Price)
```
This is **WRONG** because it ignores multiple rehedges at different prices.

**NEW (CORRECT - FIFO):**
```
✅ Futures P&L = Σ(qty_i × (exit_price - entry_price_i))
```

Each hedge trade is tracked individually:
```python
# Example:
hedge_trades = [
    (0.5, 21500),   # Buy 0.5 @ 21500
    (0.3, 21600),   # Buy 0.3 @ 21600
]
exit_spot = 21700

# Correct FIFO calculation:
pnl = 0.5*(21700-21500) + 0.3*(21700-21600) = 100 + 30 = 130
# vs OLD (wrong): 0.8*(21700-21500) = 160
```

#### 3. Hedging Costs

```
Hedging Cost = Σ (|Futures_Traded| × Price × Slippage_Rate)
```

Approximates:
- Bid-ask spread
- Market impact
- Commissions

### Break-Even Analysis

For gamma scalping to be profitable:

```
Gamma Scalping Profit > Theta Decay + Hedging Costs
```

More precisely:

```
0.5 × Gamma × (Price_Move)² × Rehedge_Count > |Theta| × Days + Hedging_Costs
```

This is why:
- **High gamma** is good (more scalping profit)
- **Low theta** is good (less decay cost)
- **Volatility** is crucial (creates price moves)

---

## Risk Management

### 1. Position Sizing

In the implementation, we use 1 lot for simplicity. In practice:

```
Position_Size = Account_Risk / Max_Loss_per_Straddle
```

Example:
```
Account_Risk = ₹100,000
Max_Loss = 30% of premium
Straddle_Premium = ₹5,000
Max_Loss_Amount = ₹1,500

Position_Size = ₹100,000 / ₹1,500 = 66 lots
```

### 2. Stop Loss

```
Stop_Loss_Trigger = Entry_Premium × (1 + Max_Loss_Percentage)
```

Example:
```
Entry_Premium = ₹5,000
Max_Loss = -30%
Stop_Loss_Amount = ₹5,000 × 0.30 = ₹1,500
Exit when: Current_Loss ≥ ₹1,500
```

### 3. Profit Target

```
Profit_Target = Entry_Premium × Profit_Target_Percentage
```

Example:
```
Entry_Premium = ₹5,000
Profit_Target = 50%
Take_Profit = ₹5,000 × 0.50 = ₹2,500
Exit when: Current_Profit ≥ ₹2,500
```

### 4. IV Risk Management

Avoid entering when IV is already elevated:

```
IV_Percentile_Rank = (Historical_IV < Current_IV).mean() × 100

Entry Rule: IV_Percentile_Rank < 30
Exit Rule: IV_Percentile_Rank > 70
```

---

## Implementation Details

### Position Dataclass

**⚠️ UPDATED:** Now includes FIFO hedge tracking.

```python
@dataclass
class Position:
    """Track straddle position with FIFO futures accounting"""
    entry_time: datetime
    entry_price: float
    strike: float
    call_qty: int
    put_qty: int
    futures_qty: int = 0  # Current net futures position
    call_premium: float = 0
    put_premium: float = 0
    total_hedging_cost: float = 0
    total_pnl: float = 0
    hedge_trades: list = None  # ✅ NEW: FIFO tracking [(qty, price), ...]
    
    def __post_init__(self):
        """Initialize mutable defaults"""
        if self.hedge_trades is None:
            self.hedge_trades = []
```

**Key Addition:** `hedge_trades` list stores each individual futures trade as `(quantity, entry_price)` tuples for proper FIFO accounting.

### Complete Strategy Workflow

```
1. Initialize Strategy
   ↓
2. Calculate Historical Volatility
   ↓
3. Estimate Implied Volatility
   ↓
4. Check Entry Conditions
   ↓
5. [If True] Enter Straddle Position
   - Buy ATM Call
   - Buy ATM Put
   - Calculate Initial Greeks
   ↓
6. Monitor Position (Every Bar/Tick)
   ↓
7. Calculate Current Greeks
   ↓
8. Check if Rehedge Needed
   ↓
9. [If True] Rehedge Delta
   - Calculate futures needed
   - Execute hedge
   - Record hedging cost
   ↓
10. Calculate Current P&L
    ↓
11. Check Exit Conditions
    ↓
12. [If True] Exit Position
    - Close options
    - Close futures hedge
    - Calculate final P&L
    ↓
13. Return to Step 2 (Look for next entry)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delta_threshold` | 0.15 | Rehedge when \|delta\| > this |
| `iv_entry_percentile` | 30 | Enter when IV < 30th percentile |
| `iv_exit_percentile` | 70 | Exit when IV > 70th percentile |
| `profit_target` | 0.50 | Take profit at 50% of premium |
| `max_loss` | -0.30 | Stop loss at -30% of premium |
| `time_to_expiry` | 7 | Use 7-day options |
| `risk_free_rate` | 0.06 | 6% annual risk-free rate |

### ATM Strike Selection

```python
def get_atm_strike(self, spot_price, strike_interval=50):
    return round(spot_price / strike_interval) * strike_interval
```

Example:
```
Spot = 23,547
Strike_Interval = 50
ATM_Strike = round(23,547 / 50) × 50 = 23,550
```

### Volatility Calculations

**⚠️ UPDATED:** Primary source is now real market IV from CSV.

```python
# 1. Calculate returns (for fallback only)
returns = np.log(prices / prices.shift(1))

# 2. **PRIMARY: Use real IV from CSV data**
if 'iv' in price_data.columns and not price_data['iv'].isna().all():
    # Use real market IV
    price_data['implied_vol'] = price_data['iv']
    
    # Fill missing values with interpolation
    price_data['implied_vol'] = price_data['implied_vol'].interpolate(method='linear')
    price_data['implied_vol'].fillna(method='bfill', inplace=True)
    price_data['implied_vol'].fillna(method='ffill', inplace=True)
    
else:
    # 3. FALLBACK: Rolling historical volatility (only if no IV column)
    rolling_vol = returns.rolling(window=20).std()
    historical_iv = rolling_vol * np.sqrt(252)  # Annualize
    implied_iv = historical_iv * 1.2  # 20% premium to HV
```

**Key Change:** Synthetic `HV × 1.2` is now **fallback only**, not primary source.

---

## Performance Metrics

### Win Rate

```
Win_Rate = (Number_of_Winning_Trades / Total_Trades) × 100%
```

### Profit Factor

```
Profit_Factor = Total_Gross_Profit / Total_Gross_Loss
```

### Average Win/Loss

```
Avg_Win = Total_Profit / Number_of_Wins
Avg_Loss = Total_Loss / Number_of_Losses
```

### Maximum Drawdown

```
Drawdown[t] = Cumulative_PnL[t] - Running_Max_PnL[t]
Max_Drawdown = min(Drawdown)
```

### Sharpe Ratio (Annualized)

```
Sharpe = (Mean_Return - Risk_Free_Rate) / Std_Dev_Returns × √252
```

---

## Summary

The gamma scalping strategy is a sophisticated delta-neutral approach that:

1. **Buys options** when IV is low (cheap volatility)
2. **Maintains delta neutrality** through dynamic hedging
3. **Profits from gamma** by automatically buying low and selling high
4. **Manages risk** through IV monitoring and P&Lstops
5. **Requires high realized volatility** to overcome theta decay

**Success Factors:**
- ✅ Enter when IV is low ( realized volatility
- ✅ Maintain discipline in rehedging
- ✅ Manage theta decay costs
- ✅ Control hedging costs (minimize overtrades)
- ✅ Exit at appropriate profit/loss levels

**Risk Factors:**
- ❌ Low realized volatility (insufficient price movement)
- ❌ High theta decay (especially near expiry)
- ❌ Excessive hedging costs
- ❌ IV crush (buying expensive IV that deflates)
- ❌ Gap moves (delta hedging assumes continuous prices)
