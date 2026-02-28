# Nifty50 Gamma Scalping System — Technical Documentation

> **Version**: V1.0 (option-scalping-v1-stable)  
> **Market**: NSE Nifty 50 (India)  
> **Instrument**: ATM Straddle (Weekly Options) + Nifty Futures for delta hedging  
> **Lot Size**: 50

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture & File Structure](#2-architecture--file-structure)
3. [Strategy: Volatility Arbitrage Gamma Scalping](#3-strategy-volatility-arbitrage-gamma-scalping)
   - 3.1 [Core Concept](#31-core-concept)
   - 3.2 [Entry Conditions](#32-entry-conditions)
   - 3.3 [Delta Hedging Logic](#33-delta-hedging-logic)
   - 3.4 [Intraday Variance Checkpoints](#34-intraday-variance-checkpoints)
   - 3.5 [Exit Conditions](#35-exit-conditions)
4. [Mathematical Formulas](#4-mathematical-formulas)
   - 4.1 [Black-Scholes Model](#41-black-scholes-model)
   - 4.2 [Greeks](#42-greeks)
   - 4.3 [Realized Volatility](#43-realized-volatility)
   - 4.4 [P&L Attribution](#44-pl-attribution)
   - 4.5 [Risk Metrics](#45-risk-metrics)
5. [Retail-Hardened Patches (STRATEGY_1.md)](#5-retail-hardened-patches-strategy_1md)
6. [Backtest Engine](#6-backtest-engine)
   - 6.1 [Data Preprocessing](#61-data-preprocessing)
   - 6.2 [Session Backtest Logic](#62-session-backtest-logic)
   - 6.3 [Day Classification](#63-day-classification)
   - 6.4 [Parameter Sweeps](#64-parameter-sweeps)
7. [Streamlit Dashboard (app.py)](#7-streamlit-dashboard-apppy)
8. [Configuration Parameters](#8-configuration-parameters)
9. [Position Sizing & Capital Allocation](#9-position-sizing--capital-allocation)
10. [Data Sources & Format](#10-data-sources--format)
11. [Live Trading Template](#11-live-trading-template)
12. [Performance Metrics Computed](#12-performance-metrics-computed)
13. [Dependencies & Installation](#13-dependencies--installation)

---

## 1. System Overview

This is a **Volatility Arbitrage Gamma Scalping** system designed for Nifty 50 weekly options on NSE. The strategy exploits the persistent spread between **Realized Volatility (RV)** and **Implied Volatility (IV)** by:

1. Buying an **ATM straddle** (long call + long put at the same strike) when RV > IV
2. **Delta-hedging** with Nifty futures to stay delta-neutral and capture gamma P&L
3. **Exiting** when the volatility edge disappears or session-level risk criteria are triggered

**Why it works**: When the market moves more than implied (RV > IV), a long straddle + delta hedge earns money on each hedge cycle. The straddle price rises with realized movement, and each hedge locks in a small profit proportional to `½ × Γ × (ΔS)²`.

---

## 2. Architecture & File Structure

```
files/
│
├── gamma_scalping_system.py   # Core strategy engine (BlackScholes, GammaScalpingStrategy)
├── backtest_engine.py         # Optimized session-by-session backtest pipeline
├── app.py                     # Streamlit dashboard (UI, charts, regime analysis)
├── config.py                  # Global configuration constants & presets
├── api_data_fetcher.py        # Market data fetching (NSEpy / yfinance)
├── live_trading_template.py   # Live trading scaffold (broker API hooks)
├── FINAL_NIFTY_MASTER_ATM.csv # Primary dataset: 1-min ATM options data
└── requirements.txt           # Python dependencies
```

### Module Dependency Flow

```
app.py
  └── backtest_engine.py       ← Primary backtest pipeline (used by dashboard)
       └── [scipy, numpy, pandas]

gamma_scalping_system.py       ← Standalone strategy class (used by live template)
  └── [scipy.stats.norm, numpy, pandas]

live_trading_template.py
  └── gamma_scalping_system.py
```

> **Note**: `backtest_engine.py` is a separate, optimized pipeline distinct from `gamma_scalping_system.py`. The Streamlit app uses `backtest_engine.py`. `gamma_scalping_system.py` is the standalone/research implementation.

---

## 3. Strategy: Volatility Arbitrage Gamma Scalping

### 3.1 Core Concept

The strategy is a **delta-neutral long volatility** position:

| Component | Instrument | Direction |
|-----------|-----------|-----------|
| ATM Call | Weekly Nifty Option | Long |
| ATM Put | Weekly Nifty Option (same strike) | Long |
| Futures | Nifty Futures (current weekly) | Short/Long for hedge |

**P&L drivers**:
- **Gamma P&L (skill)**: Earned from each delta rehedge when market moves significantly
- **Vega P&L (luck)**: Earned if IV expands after entry
- **Theta P&L (cost)**: Paid as time decay, greatest in the last few days to expiry

### 3.2 Entry Conditions

**All three conditions must be true simultaneously**:

#### Condition 1: RV > IV Edge
```
RV_edge = (5-day RV - IV) / IV
Enter if: 5% ≤ RV_edge ≤ 10%
```
- RV edge must be meaningful (≥5%) to justify premium paid
- Capped at 10% to avoid entering in extreme dislocation

#### Condition 2: IV Percentile < 65%
```
IV_percentile = (count of historical IVs < current IV) / total_IVs × 100
Enter if: IV_percentile < 65%
```
- Ensures we are not overpaying for options in a high-IV regime
- Computed over a rolling 100-bar window

#### Condition 3: First 15-Minute Move (Patch 0)
```
First15m_Move_Pct = (High_15min - Low_15min) / Open_Spot
Enter if: First15m_Move_Pct ≥ 0.10%
```
- Avoids "dead" low-volatility days where gamma cannot be harvested
- Entry is triggered after 9:30 AM (bar 15 of 1-min data)

#### High-IV Session Filter (backtest_engine.py)
```
Skip session if: entry_IV > median_IV (across all sessions in dataset)
```
- High-IV sessions consistently underperform due to expensive straddles
- Applied as a hard filter before entry

### 3.3 Delta Hedging Logic

The hedge is triggered only when **all** the following gate conditions pass (in order):

#### Gate 1: Daily Hedge Cap (Patch 5)
```
Skip if: hedge_count_today ≥ 30 (max_daily_hedges)
```

#### Gate 2: Delta Band (Patch 2 — Time-Adaptive)
```
Morning (before 12:00):  |portfolio_delta| > 0.25
Midday (12:00–14:00):    |portfolio_delta| > 0.30
Late Session (>14:00):   |portfolio_delta| > 0.35
```
- Band widens in the afternoon because near-expiry gamma spikes and delta becomes unstable

#### Gate 3: ATR Move Filter (Patch 3)
```
Warm-up phase (first 25 bars after entry):
    required_move = spot × 0.25%

After warm-up:
    required_move = 1.5 × ATR_5min
    (ATR = 5-bar average of |spot_t - spot_{t-1}|)

Fire hedge only if: |spot - last_hedge_spot| ≥ required_move
```

#### Gate 4: Cooldown (Patch 4)
```
Fire hedge only if: elapsed_time_since_last_hedge ≥ 15 minutes
```

#### Gate 5: Economic Gate (Patch 1)
```
Expected_Capture = |net_delta| × |spot - last_hedge_spot| × 50 (lot size)
Gate_Threshold = K × round_trip_cost = 6 × ₹40 = ₹240

Fire hedge only if: Expected_Capture ≥ Gate_Threshold
Exception: First hedge of the session always fires (bypass)
```

#### Hedge Execution (FIFO Accounting)
```
Futures_Delta_To_Trade = -portfolio_delta - current_futures_qty
Record: (qty, spot) for FIFO P&L tracking
Update: last_hedge_spot = spot, last_hedge_time = timestamp
Cost: ₹20 execution fee per hedge leg
```

### 3.4 Intraday Variance Checkpoints

Two hard checkpoints prevent theta bleed on bad-RV days:

#### 12:30 PM Checkpoint
```
RV_ratio = cumulative_realized_variance / implied_variance

If RV_ratio < 0.40 (noon_rv_threshold):
    → REDUCE position to 50%
    → Halve options exposure and hedge P&L
```

#### 2:30 PM Checkpoint
```
If RV_ratio < 0.70 (afternoon_rv_threshold):
    → EXIT entire position immediately
    → Reason: 'rv_exit_1430'
```

**Variance computation**:
```
# Per bar:
log_return = log(spot_t / spot_{t-1})
cumulative_realized_variance += log_return²

# Implied variance from straddle price at entry:
implied_variance = (straddle_price / spot)²

# Intraday scaling:
implied_var_so_far = entry_iv² × (bars_elapsed / 375)
```

### 3.5 Exit Conditions

Checked in priority order. First condition met triggers immediate exit:

| Priority | Condition | Check | Trigger |
|----------|-----------|-------|---------|
| 1 | Profit Target | `pnl_pct ≥ 50%` of premium | `PROFIT_TARGET` |
| 2 | Stop Loss | `pnl_pct ≤ -30%` of premium | `STOP_LOSS` |
| 3 | IV Crush | `(entry_iv - current_iv) / entry_iv ≥ 8%` | `IV_DROP` |
| 4 | Delta Pinning | `|portfolio_delta| > 0.50` for ≥ 30 minutes | `DELTA_PINNED` |
| 5 | 12:30 PM RV Check | `realized_var < 40% of implied_var` | `REDUCE_50PCT` |
| 6 | 2:30 PM RV Check | `realized_var < 70% of implied_var` | `rv_exit_1430` |
| 7 | IV Crush (backtest) | `iv_drop_pct ≥ 12%` | `iv_crush` |
| 8 | Session Close | Default at 3:30 PM | `session_close` |

---

## 4. Mathematical Formulas

### 4.1 Black-Scholes Model

All option pricing uses the standard Black-Scholes model for European options.

**d1 and d2 parameters**:
```
d1 = [ln(S/K) + (r + σ²/2) × T] / (σ × √T)
d2 = d1 - σ × √T

Where:
  S = spot price
  K = strike price (ATM, rounded to nearest ₹50 interval)
  T = time to expiry (in years)
  r = risk-free rate (6% annual)
  σ = implied volatility (decimal form)
```

**Call Price**:
```
C = S × N(d1) - K × e^(-rT) × N(d2)

At expiry (T=0):  C = max(S - K, 0)
```

**Put Price** (via put-call parity):
```
P = K × e^(-rT) × N(-d2) - S × N(-d1)

At expiry (T=0):  P = max(K - S, 0)
```

**ATM Straddle Price**:
```
Straddle = Call + Put
≈ S × σ × √(T/2π) × 2  (ATM approximation)
```

**Implied Daily Move (from straddle)**:
```
implied_move = straddle_price   (absolute ₹ terms)
implied_variance = (straddle_price / spot)²
```

### 4.2 Greeks

All Greeks are computed per-bar for the portfolio:

**Delta (directional sensitivity)**:
```
Call Delta:      Δ_call = N(d1)               [0 to 1]
Put Delta:       Δ_put  = N(d1) - 1           [-1 to 0]
Portfolio Delta: Δ_port = Δ_call + Δ_put + futures_qty
                        ≈ 0 at ATM (straddle is delta-neutral at entry)
```

**Gamma (rate of delta change)**:
```
Γ = N'(d1) / (S × σ × √T)

Portfolio Gamma (straddle): 2 × Γ  [both call and put have same gamma]

where N'(d1) = (1/√(2π)) × e^(-d1²/2)  [standard normal PDF]
```

**Gamma P&L formula** — the core of gamma scalping:
```
Gamma_PnL ≈ ½ × Γ × (ΔS)²

This is the theoretical profit from a spot move ΔS.
Positive for long gamma (long straddle).
```

**Vega (sensitivity to IV change)**:
```
ν = S × N'(d1) × √T / 100   (per 1% change in IV)

Portfolio Vega (straddle): 2 × ν
```

**Theta (time decay)**:
```
θ_call = [-S × N'(d1) × σ / (2√T) - r × K × e^(-rT) × N(d2)] / 365  (daily)
θ_put  = [-S × N'(d1) × σ / (2√T) + r × K × e^(-rT) × N(-d2)] / 365 (daily)

Portfolio Theta (straddle): θ_call + θ_put  [always negative for long options]
```

**Greeks P&L Attribution** (per bar):
```
gamma_pnl  = ½ × portfolio_gamma × (ΔS)²
vega_pnl   = portfolio_vega × Δσ
theta_pnl  = portfolio_theta × Δt  (Δt in days)
```

### 4.3 Realized Volatility

**Historical/Realized Volatility (N-day)**:
```
log_returns = log(P_t / P_{t-1})
RV = std(log_returns over N days) × √252 × 100    (annualized %)

Default N = 5 trading days (recent RV window)
```

**Intraday Cumulative Realized Variance**:
```
bar_return = (spot_t - spot_{t-1}) / spot_{t-1}
cumulative_realized_variance = Σ bar_return²   (sum over session bars)

Annualized per bar: realized_var × 252 × 375  (252 trading days × 375 min/day)
```

**5-Bar ATR (Average True Range)**:
```
TR_i = |spot_i - spot_{i-1}|    (simplified; intraday bars, no overnight gaps)
ATR_5 = mean(TR_{i-5} to TR_{i-1})

Used in Patch 3 move filter: required_move = 1.5 × ATR_5
```

**IV Percentile**:
```
IV_pctl = (count of historical_IV < current_IV) / len(history) × 100

Computed over rolling 100-bar window.
Entry allowed only if IV_pctl < 65.
```

### 4.4 P&L Attribution

**Options P&L** (at position close):
```
options_pnl = (call_value_exit + put_value_exit 
              - call_premium_entry - put_premium_entry) × position_scale
```

**Futures/Hedge P&L (FIFO)**:
```
For each hedge trade (qty_i, entry_spot_i):
    hedge_pnl_i = qty_i × (exit_spot - entry_spot_i)

Total Futures PnL = Σ hedge_pnl_i × position_scale
```

**Session Net P&L** (backtest_engine.py):
```
straddle_pnl = (straddle_price_exit - straddle_price_entry) × 50 × position_scale
hedge_pnl    = Σ [delta_i × (spot_i - last_hedge_spot_i) × 50 × position_scale] - fees_per_hedge

total_fees   = n_hedges × ₹20 + ₹80  (straddle open+close brokerage)
net_pnl      = straddle_pnl + hedge_pnl - ₹80  (straddle fees only; hedge fees already deducted)
gross_pnl    = straddle_pnl + (hedge_pnl before deducting per-hedge fees)
```

**Fee structure**:
```
Per hedge trade:        ₹20 execution fee
Straddle open+close:    ₹80 total (₹20 × 4 legs)
Default round-trip:     ₹40 (₹20 buy + ₹20 sell)
Economic gate basis:    ₹40 (GATE_BASIS_COST)
```

### 4.5 Risk Metrics

**Sharpe Ratio** (annualized):
```
Sharpe = (mean_daily_pnl / std_daily_pnl) × √252
```

**Calmar Ratio**:
```
CAGR_pct = (net_pnl / starting_capital) / n_years × 100
Calmar   = |CAGR_pct| / |max_drawdown_pct|
```

**Maximum Drawdown**:
```
cumulative_pnl = cumsum(daily_pnls)
running_max    = max.accumulate(cumulative_pnl)
drawdown       = cumulative_pnl - running_max
max_dd         = min(drawdown)
```

**Profit Factor**:
```
Profit_Factor = Σ(winning_day_pnls) / |Σ(losing_day_pnls)|
```

**RV/IV Ratio**:
```
rv_iv_ratio = final_realized_var / entry_iv²

RV/IV > 1.0: market moved more than implied (strategy favored)
RV/IV < 1.0: market moved less than implied (theta bled)
```

---

## 5. Retail-Hardened Patches (STRATEGY_1.md)

Six patches were applied to make the strategy robust for retail execution (realistic costs, limited infrastructure):

| Patch | Name | Description |
|-------|------|-------------|
| **Patch 0** | First 15-Min Gate | Skip sessions where opening 15-min move < 0.10% |
| **Patch 1** | Economic Hedge Gate | Hedge only if Expected_Capture ≥ K × ₹40 (K=6) |
| **Patch 2** | Time-Adaptive Delta Band | Widen delta band from 0.25 → 0.30 → 0.35 through the day |
| **Patch 3** | Volatility-Aware Move Filter | Use 1.5× ATR_5min (not static %) after warm-up |
| **Patch 4** | Hedge Cooldown | Minimum 15 minutes between hedges |
| **Patch 5** | Daily Hedge Cap | Maximum 30 hedges per session |
| **Fix 6** | First Hedge Bypass | First hedge always passes economic gate (avoids deadlock) |

**IV-Adjusted Delta Band** (backtest_engine.py Fix 1):
```
iv_adjustment = min(entry_iv / median_iv, 1.0)
delta_band = base_band × iv_adjustment

Interpretation: when IV is high (expensive day), widen band → hedge less
```

**Optimized defaults** (via K × Cooldown heatmap, Sharpe = 3.30):
```
K-factor    = 6    (economic_multiplier)
Cooldown    = 15 minutes
Max hedges  = 30/day
Open filter = 0.10% first-15m move
IV crush    = 12% drop
```

---

## 6. Backtest Engine

### 6.1 Data Preprocessing (`preprocess_raw_data`)

Handles two CSV formats:

**Format A — Long format** (raw NSE options data):
```
Columns: date, time, symbol, option_type, type, strike_offset,
         open, high, low, close, volume, oi, iv, spot

Processing:
1. Filter ATM rows: strike_offset == 0 or 'ATM'
2. Separate CE and PE rows
3. Merge on (date, time) key
4. Create session timestamp
```

**Format B — Wide format** (pre-merged):
```
Columns: date, time, CE_close, PE_close, CE_open, PE_open,
         CE_high, PE_high, CE_low, PE_low, CE_volume, PE_volume, iv, spot
         
Processing:
1. Rename columns to standard names
2. Create straddle_price = close_CE + close_PE
```

**Per-session derived columns**:
```
T (time to expiry per bar):
  rem_min = max(375 - bars_from_start, 1)
  T = rem_min / (375 × 252)

delta_CE = N(d1)     [Black-Scholes call delta]
delta_PE = N(d1) - 1 [Black-Scholes put delta]
net_delta = delta_CE + delta_PE

ATR_5 = 5-bar rolling mean of |spot_t - spot_{t-1}|

realized_var = cumsum(log_returns²) × 252 × 375
iv_drop_pct  = (entry_iv - current_iv) / entry_iv × 100
```

**Data quality filters**:
```
1. Volume guard: skip bars where volume_CE < 20th percentile AND < 100
2. IV sanity: flag iv ≤ 0 or iv > 2.0 as errors
3. Spike filter: flag |spot_pct_change| > 1% as spike
4. Session completeness: only include sessions with ≥ 300 bars
```

### 6.2 Session Backtest Logic (`run_session_backtest`)

Each session (trading day) is backtested independently:

```
Step 1: First 15-min gate check
  first_15m_move = (max - min of first 15 bars) / entry_spot
  If move < open_filter → skip session

Step 2: High-IV skip
  If entry_iv > median_session_iv → skip session

Step 3: Enter at bar 15 (9:30 AM)
  entry_price = straddle_prices[15]
  last_hedge_spot = entry_spot

Step 4: For each subsequent bar:
  a. Skip low-volume bars entirely
  b. Check IV crush exit (iv_drop_pct ≥ 12%)
  c. Check variance checkpoints (12:30 PM, 2:30 PM)
  d. Check daily hedge cap
  e. Evaluate hedge gates (delta, ATR, cooldown, economic)
  f. Execute hedge if all gates pass

Step 5: Exit at triggered bar or session close
  straddle_pnl = (exit_price - entry_price) × 50 × position_scale
  net_pnl = straddle_pnl + Σ(hedge_pnl - fees) - straddle_fees
```

### 6.3 Day Classification

Each session is classified along two axes:

| Classification | Criteria |
|---------------|---------|
| `high_iv_trend` | IV > median AND `|close - open| / (high - low)` > 0.5 |
| `high_iv_chop` | IV > median AND trend ratio ≤ 0.5 |
| `low_iv_trend` | IV ≤ median AND trend ratio > 0.5 |
| `low_iv_chop` | IV ≤ median AND trend ratio ≤ 0.5 |
| `skipped` | Session did not meet entry criteria |

**Trending metric**:
```
trending = |spot_close - spot_open| / max(spot_high - spot_low, 0.01) > 0.5
```

**Day-of-week flags**:
```
days_to_expiry = (3 - dow) % 7   (days until next Thursday)
is_thursday    = (dow == 3)       (Nifty weekly expiry day)
is_expiry_day  = is_thursday
```

### 6.4 Parameter Sweeps

**1D Sweep (`run_parameter_sweep`)**: Varies one parameter while holding others at defaults.  
Returns: `param_value, net_pnl, hedge_count, sharpe, sessions_entered`

**2D Sweep (`run_2d_sweep`)**: Grid search over two parameters (e.g., K × Cooldown).  
Returns: `param1, param2, net_pnl, sharpe`

**Sharpe computed per sweep point**:
```
Sharpe = (mean(daily_pnls) / std(daily_pnls)) × √252
Only entered sessions included in daily_pnls.
```

---

## 7. Streamlit Dashboard (app.py)

The dashboard (`app.py`) is the primary user interface.

### Launch
```bash
streamlit run app.py
```

### Sidebar Controls

| Control | Type | Default | Description |
|---------|------|---------|-------------|
| Upload CSV | File uploader | — | Raw 1-min options data |
| Fee Per Trade (₹) | Number input | ₹20.0 | Execution fee per hedge |
| Economic Gate (K) | Slider | 6.0 | `k_factor` multiplier |
| Hedge Cooldown | Slider | 15 min | Min time between hedges |
| Date Range | Date picker | full range | Filter sessions by date |
| Day Type Filter | Multi-select | all types | Filter by regime |

### Data Loading Priority
1. Uploaded CSV (if provided)
2. `FINAL_NIFTY_MASTER_ATM.csv` (default bundled data)
3. Synthetic data (generated if no file available)

### Dashboard Tabs

**Tab 1 — P&L Analysis**:
- Cumulative equity curve (area chart, `#00E5FF`)
- Drawdown chart (area chart, `#FF1744`)
- Economic gate analysis: fired vs blocked hedges
- Performance by time band (Morning/Midday/Late)
- Greeks scatter: Delta vs Expected Capture, Delta vs Net P&L
- Slippage sensitivity (baseline, 1.5×, 2× fees)
- Best 5 / Worst 5 sessions table

**Tab 2 — Regime Breakdown**:
- Regime performance chart (bar + win rate line)
- Regime summary table
- RV/IV ratio vs P&L scatter with trend line
- Weekday analysis (Mon–Fri P&L bar chart)
- Thursday vs non-Thursday comparison
- Days-to-expiry P&L chart

**Tab 3 — Parameter Robustness**:
- K × Cooldown heatmap (Sharpe Ratio as color)
- K sweep: [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
- Cooldown sweep: [5, 7, 9, 11, 13, 15] minutes

**Tab 4 — Hedge Log**:
- Per-hedge trade table with color coding (green = profit, red = loss)
- Filter by individual date
- Columns: time, spot, delta, expected/actual capture, fee, net P&L, trigger type, blocked status

### Key Metrics Row

| Metric | Formula |
|--------|---------|
| Net P&L | `Σ net_pnl` (entered sessions only) |
| Gross P&L | `Σ gross_pnl` (before fees) |
| Fee Drag % | `total_fees / gross_pnl × 100` |
| Sharpe | `mean(daily) / std(daily) × √252` |
| Calmar | `CAGR% / max_dd%` |
| Profit Factor | `Σ wins / Σ|losses|` |
| Max Drawdown | Peak-to-trough of cumulative P&L |
| Max Consecutive Losers | Longest losing streak |
| Net Win Rate | `% hedge trades with net_pnl > 0` |

---

## 8. Configuration Parameters

All defaults are defined in `config.py` and `backtest_engine.py` constants:

### Entry Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RV_WINDOW_DAYS` | 5 | Days for realized volatility calculation |
| `RV_IV_EDGE_MIN` | 0.05 (5%) | Minimum RV > IV edge to enter |
| `RV_IV_EDGE_MAX` | 0.10 (10%) | Maximum RV > IV edge to enter |
| `IV_ENTRY_PERCENTILE` | 65 | Max IV percentile to enter |
| `FIRST_15M_MOVE_THRESHOLD` | 0.10% | Min first-15-min H-L range |

### Delta Hedging Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DELTA_HEDGE_BAND` | 0.25 | Base delta band |
| `SPOT_MOVE_THRESHOLD` | 0.25% | Min spot move (warm-up phase) |
| `ECONOMIC_MULTIPLIER (K)` | 6 | Gate multiplier (heatmap optimized) |
| `ROUND_TRIP_COST` | ₹40 | Gate basis cost |
| `HEDGE_COOLDOWN_MINUTES` | 15 | Min minutes between hedges |
| `MAX_DAILY_HEDGES` | 30 | Hard daily hedge cap |
| `EXECUTION_FEE` | ₹20 | Per-trade fee |

### Checkpoint Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NOON_RV_THRESHOLD` | 0.40 | 12:30 PM: cut 50% if RV < 40% of IV |
| `AFTERNOON_RV_THRESHOLD` | 0.70 | 2:30 PM: exit if RV < 70% of IV |

### Exit Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PROFIT_TARGET` | 50% | Exit if P&L > 50% of premium paid |
| `MAX_LOSS` | -30% | Exit if P&L < -30% of premium paid |
| `IV_DROP_EXIT` | 8% | Exit if IV drops > 8% intraday |
| `DEFAULT_IV_CRUSH` (backtest) | 12% | IV drop exit for backtest engine |
| `DELTA_PIN_THRESHOLD` | 0.50 | Delta level for pinning detection |
| `DELTA_PIN_DURATION_MINUTES` | 30 | Duration before pinning exit triggers |

### Options Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TIME_TO_EXPIRY` | 7 days | Weekly options |
| `RISK_FREE_RATE` | 6% | India risk-free rate |
| `STRIKE_INTERVAL` | ₹50 | ATM strike rounding |
| `LOT_SIZE` | 50 | Nifty lot size |

### Strategy Presets (config.py)

| Preset | Delta Band | K×Cost | Cooldown | Hedges Cap | Entry Edge |
|--------|-----------|-------|---------|-----------|-----------|
| `CONSERVATIVE` | 0.20 | 5.0× | 10 min | 20/day | 7–12% |
| `AGGRESSIVE` | 0.30 | 3.0× | 5 min | 40/day | 3–15% |
| `LOW_VOL_PARAMS` | 0.20 | — | — | — | 3–?% |
| `HIGH_VOL_PARAMS` | 0.30 | — | — | — | 8–?% |
| `TRENDING_PARAMS` | 0.20 | — | — | — | — |
| `RANGE_BOUND_PARAMS` | 0.35 | — | — | — | — |

---

## 9. Position Sizing & Capital Allocation

All position sizing is fixed at **1 lot** (50 units) per session. The strategy does not scale position size dynamically based on account equity.

### 9.1 Starting Capital

```
Starting Capital:  ₹1,75,000
Source:            app.py (hardcoded)
Rationale:         Approximate margin required to hold a 1-lot ATM Nifty straddle
                   + delta-hedge futures position on NSE
```

This capital figure is used **only** for performance metric calculations (CAGR, Calmar ratio, Total Return %). It is not used to size or scale trades.

### 9.2 Straddle Position (Options)

```
Instrument:    ATM Nifty Weekly Call + Put (same strike)
Lot Size:      50 units per lot
Lots Traded:   1 lot
Quantity:      50 units of each (CE + PE)

Cost at entry: straddle_price × 50
Example:       If straddle = ₹200 (CE ₹100 + PE ₹100)
               Total premium = ₹200 × 50 = ₹10,000

Note:          Actual premium varies with market IV.
               ₹1,75,000 is the margin block, not the premium paid.
```

**Position Scaling** (`position_scale`): Defaults to `1.0`. Drops to `0.5` if the 12:30 PM variance checkpoint triggers a 50% position cut. This halves both the straddle P&L and hedge P&L for the rest of that session.

### 9.3 Delta-Hedge Position (Futures)

Each hedge trade is sized by the current portfolio delta:

```
Hedge Quantity = portfolio_delta × LOT_SIZE × position_scale
               = portfolio_delta × 50 × position_scale

Hedge P&L per trade:
    hedge_pnl = delta × (spot_current - last_hedge_spot) × 50 × position_scale
```

Futures are traded in the current weekly Nifty contract. The hedge quantity is not rounded to whole lots in the backtest — it is treated as a continuous quantity proportional to delta.

### 9.4 Fee Constants (from `backtest_engine.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `LOT_SIZE` | 50 | Nifty lot size (units per lot) |
| `DEFAULT_FEE` | ₹20 | Execution fee per hedge leg |
| `STRADDLE_ENTRY_EXIT_FEE` | ₹80 | Brokerage for straddle open + close (₹20 × 4 legs) |
| `GATE_BASIS_COST` | ₹40 | Economic gate threshold basis (round-trip hedge cost) |
| `DEFAULT_K` | 6 | Gate multiplier (K × ₹40 = ₹240 minimum capture required) |
| `DEFAULT_MAX_HEDGES` | 30 | Max hedge trades per session |

### 9.5 Daily Cost Summary

```
Fixed cost per session:    ₹80   (straddle open + close brokerage)
Variable cost per session: n_hedges × ₹20

Worst-case fees (30 hedges): ₹80 + (30 × ₹20) = ₹680/day
Typical fees (8 hedges):     ₹80 + (8 × ₹20)  = ₹240/day

Break-even per hedge:   ≥ ₹240 expected capture (K=6 gate enforces this)
```

---

## 10. Data Sources & Format

### Primary Dataset: `FINAL_NIFTY_MASTER_ATM.csv`

The bundled 38 MB CSV contains 1-minute bar data for NIFTY ATM options.

**Expected columns (long format)**:
```
date         : date string (format: DD-MM-YY, may be Excel-wrapped as ="DD-MM-YY")
time         : time string (HH:MM:SS)
symbol       : 'NIFTY'
option_type  : 'CE' or 'PE'
type         : 'CE' or 'PE' (alias)
strike_offset: 0 or 'ATM' (filters to ATM only)
open, high, low, close : option OHLC prices
volume       : trade volume
oi           : open interest
iv           : implied volatility (decimal 0.05–2.0, or percentage 5–200)
spot         : Nifty spot price
```

**Auto-detection**: If `CE_close` / `PE_close` columns are present, wide format is assumed.

**IV normalization**:
```python
if iv.median() > 1:   # IV in percentage form
    iv = iv / 100.0   # Convert to decimal
```

### Fetching Live Data (`api_data_fetcher.py`)

**Preferred source**: `yfinance` (Yahoo Finance)
```python
ticker = yf.Ticker("^NSEI")
data = ticker.history(start=start_date, end=end_date, interval='1d')
```

**Fallback**: `NSEpy` (NSE historical API — may have connectivity issues)

**Note**: Both sources provide **daily** data. For intraday simulation, `format_data_for_strategy()` interpolates daily OHLC into synthetic intraday bars. **Real intraday tick data must be obtained via broker APIs** (Zerodha, Upstox, Angel).

---

## 11. Live Trading Template

`live_trading_template.py` provides a scaffold for live deployment.

### `LiveGammaScalper` class

```python
LiveGammaScalper(strategy_params, broker_config)
```

**Methods to implement** (broker-specific):
- `connect_broker_api()` — Zerodha KiteConnect / Upstox / Angel API
- `get_nifty_spot_price()` — Real-time spot via market quote WebSocket
- `get_option_chain()` — ATM call/put prices and IVs
- `place_order(symbol, type, qty, order_type)` — Order execution

**Implemented flow**:
```
monitor_and_trade(interval=60s):
  1. Get spot + IV every 60 seconds
  2. Maintain rolling iv_history (last 100 values)
  3. Check should_enter() → enter straddle if True
  4. While in position:
       → Calculate Greeks
       → Check should_rehedge() → hedge with futures if True
       → Check should_exit() → close position if True
```

**Live symbol construction**:
```python
expiry_date = get_current_weekly_expiry()  # e.g., "25FEB"
call_symbol = f"NIFTY{expiry_date}{strike}CE"
put_symbol  = f"NIFTY{expiry_date}{strike}PE"
futures_sym = f"NFO:NIFTY{expiry_date}FUT"
```

> ⚠️ **Warning**: Paper trade extensively before going live. Start with minimum position sizes. Risk of significant capital loss.

---

## 12. Performance Metrics Computed

| Metric | Where Computed | Description |
|--------|---------------|-------------|
| Net P&L | `backtest_engine`, `app.py` | After all fees |
| Gross P&L | `backtest_engine`, `app.py` | Before straddle fees |
| Fee Drag % | `app.py` | `fees / gross_pnl × 100` |
| Sharpe Ratio | `backtest_engine`, `app.py` | Annualized |
| Calmar Ratio | `app.py` | CAGR / Max Drawdown |
| Profit Factor | `app.py` | Win / Loss gross amounts |
| Max Drawdown | `backtest_engine`, `app.py` | Peak-to-trough |
| Win Rate | `app.py` | Sessions / hedges with +ve P&L |
| Max Consecutive Losers | `app.py` | Risk management signal |
| Avg P&L per Hedge | `app.py` | `net_pnl / total_hedges` |
| RV/IV Ratio | `backtest_engine` | Realized vs implied variance ratio |
| Hedge Count | `backtest_engine` | Per session |
| Day Type | `backtest_engine` | high_iv_trend/chop, low_iv_trend/chop |
| Days to Expiry | `backtest_engine` | DTE from Thursday expiry |
| Gamma P&L | `gamma_scalping_system` | `½ × Γ × ΔS²` attribution |
| Vega P&L | `gamma_scalping_system` | `ν × Δσ` attribution |
| Theta P&L | `gamma_scalping_system` | `θ × Δt` attribution |

---

## 13. Dependencies & Installation

### requirements.txt
```
streamlit
pandas
numpy
scipy
plotly
```

### Install
```bash
pip install -r requirements.txt

# Optional data sources
pip install yfinance          # preferred
pip install nsepy             # fallback (may not work on all OS)
```

### Run Streamlit App
```bash
streamlit run app.py
```

### Run Standalone Backtest
```python
from gamma_scalping_system import GammaScalpingStrategy, generate_sample_data

strategy = GammaScalpingStrategy(delta_hedge_band=0.25, ...)
data = generate_sample_data(days=30, timeframe='5min')
results = strategy.run_backtest(data)
```

### Run Optimized Backtest Engine
```python
from backtest_engine import build_dataframes
import pandas as pd

raw_df = pd.read_csv("FINAL_NIFTY_MASTER_ATM.csv")
df_trades, df_sessions = build_dataframes(raw_df, fee=20, k_factor=6, cooldown_min=15)
print(df_sessions[['date', 'net_pnl', 'hedge_count', 'day_type']])
```

---

## Appendix: Constants Quick Reference

```python
# backtest_engine.py
RISK_FREE_RATE          = 0.065   # 6.5% India risk-free
TRADING_MINUTES_PER_DAY = 375     # 9:15 AM – 3:30 PM
TRADING_DAYS_PER_YEAR   = 252
LOT_SIZE                = 50      # Nifty lot size
DEFAULT_FEE             = 20.0    # ₹ per hedge leg
GATE_BASIS_COST         = 40.0    # ₹ round-trip basis
DEFAULT_K               = 6.0     # Economic gate multiplier
DEFAULT_COOLDOWN        = 15      # Minutes
DEFAULT_MAX_HEDGES      = 30      # Per session
DEFAULT_OPEN_FILTER     = 0.0010  # 0.10% first-15m move
DEFAULT_IV_CRUSH        = 12.0    # IV drop % for exit
STRADDLE_ENTRY_EXIT_FEE = 80.0    # ₹ total for open+close
```
