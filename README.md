# Gamma Scalping / Volatility Scalping Trading System for Nifty Options

## 📊 Overview

This is a complete implementation of a **Gamma Scalping** (also called **Volatility Scalping** or **Straddle Scalping**) strategy for Nifty options. The strategy profits when realized volatility exceeds implied volatility through delta-neutral trading.

## 🎯 Strategy Concept

### What is Gamma Scalping?

Gamma scalping is a market-neutral options trading strategy that:
1. **Buys a straddle** (ATM call + ATM put) to gain positive gamma exposure
2. **Delta hedges** by trading Nifty futures to maintain delta neutrality
3. **Profits from volatility** when the underlying moves more than priced in by IV

### The Core Principle

**Profit = Realized Volatility - Implied Volatility - Transaction Costs - Theta Decay**

- **Gamma (Γ)**: Rate of change of delta. Positive gamma means you buy low and sell high automatically through hedging
- **Theta (Θ)**: Time decay. This is your daily cost of holding the position
- **Vega (ν)**: Sensitivity to IV changes. You profit if IV increases after entry

### Why It Works

When you buy a straddle:
- You pay **implied volatility** (IV) priced into options
- You earn **realized volatility** (RV) through delta hedging
- If **RV > IV**, the hedging profits exceed the option premium decay

## 📈 Strategy Mechanics

### Entry Logic
```
Enter when:
1. IV is in bottom 30th percentile (relatively cheap volatility)
2. No existing position
3. Sufficient time to expiry (e.g., 7 days)
```

### Delta Hedging
```
When portfolio delta > threshold (e.g., 0.15):
  - If delta is positive (+0.15): SELL futures to neutralize
  - If delta is negative (-0.15): BUY futures to neutralize
```

**Example:**
- Nifty at 21,500, you buy ATM straddle
- Nifty moves to 21,600 (up 100 points)
- Call delta increases, put delta decreases → net positive delta (+0.20)
- You SELL futures to neutralize → lock in profit from the move
- Nifty moves back to 21,500
- Now net negative delta (-0.20)
- You BUY back futures → lock in another profit

**This is how you "scalp gamma"** - you're continuously buying low and selling high!

### Exit Logic
```
Exit when:
1. Profit target hit (e.g., 50% of premium paid)
2. Stop loss hit (e.g., -30% of premium paid)
3. IV spikes to 70th percentile (volatility already realized)
4. Time decay too high (near expiry)
```

## 💡 Key Concepts

### 1. Implied Volatility (IV)
- The market's expectation of future volatility
- Priced into option premiums via Black-Scholes
- **Low IV** = cheaper options = better entry for gamma scalping
- **High IV** = expensive options = poor entry

### 2. Realized Volatility (RV)
- The actual volatility experienced by the underlying
- Calculated from historical price movements
- **High RV** = more price swings = more hedging opportunities = more profit

### 3. The Greeks

**Delta (Δ)**: Directional exposure
- Long straddle delta ≈ 0 at ATM
- Delta changes as price moves (this is gamma!)

**Gamma (Γ)**: Your edge
- Highest for ATM options
- Long gamma = benefit from price movement
- Gamma decays as you approach expiry

**Theta (Θ)**: Your cost
- Daily time decay
- Long options have negative theta
- This is what you're fighting against

**Vega (ν)**: IV sensitivity
- Long vega = benefit from IV increase
- Can add extra profit if IV rises after entry

## 🔧 Implementation Details

### Files Included

1. **gamma_scalping_system.py** - Main strategy implementation
   - Black-Scholes Greeks calculator
   - Position management
   - Backtesting engine
   - Performance analytics

2. **config.py** - Configuration parameters
   - Strategy parameters
   - Risk management settings
   - Multiple preset configurations

3. **README.md** - This file

### Key Classes

#### `BlackScholes`
Calculates option prices and Greeks using the Black-Scholes model:
- Option prices (call/put)
- Greeks: delta, gamma, vega, theta
- Used for both entry pricing and real-time position management

#### `GammaScalpingStrategy`
Main strategy implementation:
- Entry/exit signal generation
- Delta hedging logic
- Position tracking
- P&L calculation
- Backtesting framework

## 🚀 Usage

### Basic Usage

```python
from gamma_scalping_system import GammaScalpingStrategy, generate_sample_data

# Generate or load Nifty data
price_data = generate_sample_data(days=30, timeframe='5min')

# Initialize strategy
strategy = GammaScalpingStrategy(
    delta_threshold=0.15,       # Rehedge at 15 delta
    iv_entry_percentile=30,     # Enter below 30th IV percentile
    profit_target=0.50,         # 50% profit target
    max_loss=-0.30,             # -30% stop loss
    time_to_expiry=7            # 7-day options
)

# Run backtest
results = strategy.run_backtest(price_data)

# View results
print(generate_performance_report(results, strategy.trade_log, strategy.pnl_history))
```

### With Live Data

```python
# For live trading, replace generate_sample_data() with your data source:

import pandas as pd

# Load from CSV
price_data = pd.read_csv('nifty_5min.csv')
price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])

# Or from API
# price_data = fetch_from_broker_api(symbol='NIFTY', timeframe='5min')

# Run backtest
results = strategy.run_backtest(price_data)
```

### Customizing Parameters

```python
from config import AGGRESSIVE, CONSERVATIVE

# Use aggressive preset
strategy = GammaScalpingStrategy(
    delta_threshold=AGGRESSIVE['DELTA_THRESHOLD'],
    profit_target=AGGRESSIVE['PROFIT_TARGET'],
    max_loss=AGGRESSIVE['MAX_LOSS'],
    iv_entry_percentile=AGGRESSIVE['IV_ENTRY_PERCENTILE']
)

# Or create custom
strategy = GammaScalpingStrategy(
    delta_threshold=0.12,       # More frequent hedging
    iv_entry_percentile=25,     # More selective entry
    profit_target=0.60,         # Higher profit target
    max_loss=-0.25,
    time_to_expiry=5
)
```

## 📊 Understanding the Output

### Performance Metrics

**Win Rate**: % of profitable trades
- Target: >50% (ideally 60-70%)
- Low win rate ok if profit factor is good

**Profit Factor**: Avg Win / |Avg Loss|
- Target: >1.5
- Measures risk-reward efficiency

**Total Rehedges**: Number of delta adjustments
- More rehedges = more opportunities to scalp
- But also = more transaction costs

**Maximum Drawdown**: Largest peak-to-trough decline
- Measures worst-case risk
- Keep below 30-40% of account

### Charts Generated

1. **Price Chart**: Shows entry/exit points
2. **Cumulative P&L**: Running profit/loss
3. **Portfolio Delta**: Shows delta neutrality maintenance
4. **Greeks**: Gamma (your edge) vs Theta (your cost)

## ⚙️ Parameter Tuning Guide

### Delta Threshold
- **Lower (0.08-0.12)**: More frequent hedging, more scalping opportunities, higher costs
- **Higher (0.15-0.20)**: Less hedging, lower costs, more directional risk
- **Recommended**: Start with 0.15, adjust based on transaction costs

### IV Entry Percentile
- **Lower (20-30)**: More selective, enter only when IV is cheap
- **Higher (35-45)**: More trading opportunities, but pay more for volatility
- **Recommended**: 25-35 for Nifty (depends on your IV estimation)

### Profit Target
- **Conservative (30-40%)**: Quick profits, higher win rate
- **Aggressive (50-70%)**: Let winners run, lower win rate
- **Recommended**: 40-50% for weekly options

### Max Loss
- **Tight (-20%)**: Limits losses but may exit too early
- **Wide (-40%)**: Gives trades room but higher risk
- **Recommended**: -25% to -35%

### Time to Expiry
- **Short (2-5 days)**: Higher gamma, higher theta, more active
- **Long (7-14 days)**: Lower gamma, lower theta, more stable
- **Recommended**: 5-7 days for good gamma/theta ratio

## 🎓 Advanced Concepts

### Position Sizing
The script uses 1 lot per straddle. In practice:
```python
# Calculate position size based on portfolio risk
max_risk = portfolio_value * 0.02  # 2% risk
straddle_cost = call_premium + put_premium
lot_size = int(max_risk / (straddle_cost * abs(MAX_LOSS)))
```

### IV Surface Analysis
In production, you should:
1. Get actual IV from option chain (not estimated from HV)
2. Compare IV across strikes and expiries
3. Look for IV skew opportunities
4. Consider VIX India for market-wide volatility

### Transaction Costs
Real trading includes:
```python
# Per trade
futures_commission = 20  # ₹ per contract
options_commission = 50  # ₹ per contract
slippage = spot * 0.0005  # 5 bps

# For active gamma scalping
daily_hedges = 5-10
monthly_cost = daily_hedges * 20 * futures_commission
# Must earn more than this!
```

### Volatility Regimes
Adjust parameters based on market:

**Low Vol (VIX < 15)**:
- More lenient IV entry
- Longer dated options
- Higher profit targets

**High Vol (VIX > 25)**:
- Very selective entry
- Shorter dated options  
- Quick profit taking

## 🔬 Backtesting Best Practices

### Data Quality
- Use clean, validated price data
- Include corporate actions adjustments
- Filter out non-trading hours
- Check for data gaps

### Walk-Forward Analysis
```python
# Instead of one backtest, do multiple:
for start_date in date_ranges:
    window_data = data[start_date:start_date+30days]
    results = strategy.run_backtest(window_data)
    # Evaluate consistency across periods
```

### Parameter Optimization
```python
# Grid search for best parameters
best_sharpe = -999
for delta_thresh in [0.10, 0.12, 0.15, 0.18]:
    for iv_percentile in [20, 25, 30, 35]:
        strategy = GammaScalpingStrategy(
            delta_threshold=delta_thresh,
            iv_entry_percentile=iv_percentile
        )
        results = strategy.run_backtest(data)
        sharpe = calculate_sharpe(results)
        if sharpe > best_sharpe:
            best_params = (delta_thresh, iv_percentile)
```

### Out-of-Sample Testing
- Train on 70% of data
- Test on remaining 30%
- Never optimize on full dataset!

## ⚠️ Risk Warnings

### Key Risks

1. **Gap Risk**: Overnight gaps can blow through hedges
   - Mitigation: Close positions before major events/weekends
   
2. **Theta Decay**: Time works against you daily
   - Mitigation: Need sufficient price movement to overcome
   
3. **Transaction Costs**: Can eat all profits with over-hedging
   - Mitigation: Optimize delta threshold, use wide spreads
   
4. **Model Risk**: Black-Scholes assumptions may not hold
   - Mitigation: Use implied volatility from market when available
   
5. **Execution Risk**: Slippage on fast markets
   - Mitigation: Use limit orders, avoid hedging in volatile moments

### When Strategy Fails

**Does NOT work well when**:
- IV is already high (expensive options)
- Market is trending strongly one direction (constant hedging losses)
- Very low volatility (not enough movement to overcome theta)
- High transaction costs environment

**Works BEST when**:
- IV is low but about to realize
- Choppy, range-bound markets
- Around events (before earnings, policy decisions)
- Good gamma/theta ratio

## 📝 Next Steps

### For Paper Trading
1. Connect to broker API for live data
2. Implement order execution
3. Add position monitoring dashboard
4. Set up alerts for entry/exit signals

### For Live Trading
1. Start with small position sizes
2. Monitor actual vs expected IV
3. Track all transaction costs
4. Keep detailed trading journal
5. Review performance weekly
6. Adjust parameters based on results

### Enhancements
- Add multiple strikes (iron condor scalping)
- Implement volatility forecasting models
- Use machine learning for entry timing
- Add correlation analysis for portfolio hedging
- Implement dynamic position sizing

## 📚 References

### Books
- "Dynamic Hedging" by Nassim Taleb
- "Option Volatility and Pricing" by Sheldon Natenberg
- "Volatility Trading" by Euan Sinclair

### Concepts to Study
- Black-Scholes model and assumptions
- Volatility smile and skew
- Greeks and their relationships
- Delta-neutral trading
- Market microstructure

## 🤝 Support

For questions or improvements:
1. Review the code comments
2. Check the configuration options
3. Run with sample data first
4. Validate results match expectations

---

**Disclaimer**: This is educational software for learning gamma scalping concepts. Use at your own risk. Options trading involves substantial risk and is not suitable for all investors. Test thoroughly with paper trading before risking real capital.

---

Good luck with your gamma scalping! 🎲📈
