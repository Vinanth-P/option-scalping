# ⚡ Nifty50 Gamma Scalp 1Min

## Overview
A highly optimized, retail-hardened Gamma Scalping (Volatility Arbitrage) backtesting engine and dashboard for Nifty 50 Options.

Unlike theoretical textbook gamma scalping, this system is explicitly designed for the **Indian retail options market**. It features strict economic gating, intraday volatility checkpoints, and slippage guards to ensure the strategy survives real-world transaction costs.

The core strategy profits when the market moves more than the options market expected it to move (Realized Volatility > Implied Volatility).

## Data Requirements
To run this backtest engine, you need high-quality granular options data:
* **Format**: 1-minute or 3-minute intraday options CSV.
* **Columns Required**: `date`, `time`, `option_type`, `strike_offset` (to identify ATM), `open`, `high`, `low`, `close`, `iv` (Implied Volatility).
* **Current Master File**: `FINAL_NIFTY_MASTER_ATM.csv` (contains 3 years of Nifty ATM Straddle data).

## Strategy Logic 
The engine operates on a strict, logic-gated pipeline. Instead of holding a straddle blindly, it actively trades market "noise" (Gamma) while suffocating time decay (Theta).

### 1. The Entry Pipeline (Finding the Edge)
The engine evaluates the option chain at `09:15 AM` every day. It will only initiate an ATM straddle position if **all** of the following quantitative conditions are met:
* **The Volatility Edge Check**: The engine calculates the trailing 5-day Realized Volatility (RV) of the Nifty 50 spot price and compares it to the Implied Volatility (IV) of the ATM straddle. It demands that historical **RV > IV by a minimum of 5%**.
* **The IV Percentile Guard**: Even if an RV edge exists, the engine refuses to buy if IV is inherently expensive. The engine tracks historical IV percentiles and requires the current IV to sit in the **bottom 65th percentile**.
* **The 15-Minute Morning Filter**: The first 15 minutes of the NSE are notoriously erratic and characterized by massive "theta crush" as overnight premiums evaporate. The engine actively monitors the spot price from 09:15 to 09:30 AM but **will not enter** unless the spot actually travels a minimum geometric distance of `0.10%`. If the market opens dead-flat, the trade is aborted entirely for the session.

### 2. The Hedging Pipeline (Scalping the Gamma)
Once the straddle is purchased, the portfolio Delta is neutral (0.0). As the Nifty moves, the combined Delta drifts. The engine aims to short or buy Nifty Futures to drag the Delta back to neutral and lock in the price difference. However, continuous hedging destroys retail accounts due to slippage, so the engine relies on strict, discrete gating:
* **The Delta Band**: Portfolio Delta must exceed a hard boundary of `±0.25`. Micro-drifts are completely ignored.
* **Spot Distance Requirement**: Even if Delta drifts past `0.25` (due to IV skew, for example), the engine enforces that the underlying spot price must have physically moved at least `0.25%` from the execution price of the *last* hedge.
* **Time-Based Cooldowns**: To combat "delta-chatter" (the engine rapidly flipping buys and sells in choppy noise), it enforces a strict **15-minute minimum time delay** between any consecutive hedges.
* **The Economic K-Factor**: This is the engine's ultimate survival mechanism. The `BlackScholes` module calculates the mathematical expected cash extraction of the hedge. This cash block must be completely guaranteed to exceed the broker's Round-Trip Cost (`₹40`) multiplied by a safety buffer `K` (`Default = 6.0`). The move must lock in at least `₹240`. If it doesn't, the hedge is skipped to save the `₹40` brokerage.
* **Daily Frequency Caps**: The engine will halt all hedging activity completely if it exceeds **30 hedges in a single day**, defending against freak volatility looping.

### 3. The Exit Pipeline (Managing the Risk)
A purchased straddle bleeds cash every minute to Theta decay. The engine actively models Intraday Realized Variance point-by-point to aggressively choke off positions that aren't earning their keep:
* **The 12:30 PM Intraday Guard**: At exactly 12:30 PM, the engine checks if the geometric Intraday Variance is at least `45%` of the straddle's original IV. If it is less, it proves the market is structurally dead for the day. The engine instantly dumps **50% of the position size** to halve the ongoing theta bleed.
* **The 2:30 PM Execution Block**: At 2:30 PM, the requirement tightens. If the Intraday Variance is `< 70%` of IV, the day is declared a total loss, and the engine triggers a **full hard-exit** to dodge the violent 3:00 PM theta cliff.
* **The Directional Pinning Exit**: If the market breaks out into a massive, one-way directional trend, the straddle acts like a pure futures contract—but with a time-decay penalty. If the engine detects Delta holding relentlessly at `> 0.50` for **30 consecutive minutes**, it identifies a trend breakout and exits the entire straddle outright. It locks in the captured directional move without paying further option premiums.
* **Standard Profit/Loss**: Hard stop-loss bounds at `-30%` of premium paid, and a hard take-profit boundary at `+50%`.

## Parameters (current locked values)
* **K-Factor (Economic Multiplier)**: `6.0` (Hedge capture must be > ₹240)
* **Round Trip Cost**: `₹40` (₹20 Buy + ₹20 Sell per hedge)
* **Cooldown Interval**: `15 minutes`
* **First 15m Move Threshold**: `0.10%`
* **Starting Capital**: `₹1,75,000` (1-lot Nifty straddle margin)

*These parameters were locked in after generating 2D heatmaps to maximize Sharpe and Calmar ratios while minimizing drawdown.*

## Performance (current backtest results)
Based on the current locked parameters running on 3-years of 1-minute Nifty options data:
* **Total Return**: ~156%
* **Sharpe Ratio**: ~3.30
* **Calmar Ratio**: ~8.70
* **Profit Factor**: ~1.30
* **Win Rate**: ~45-48%
* **Fee Drag**: Maintained below 5% of Gross P&L using the Economic Gate and Cooldowns.

(*Note: Exact numbers fluctuate slightly depending on the exact date range filtered in the dashboard UI.*)

## Known Issues & Pending Fixes
* **Missing Volume Guards**: Currently, the system trusts the 1-minute printed price even if volume drop-offs occur deep OTM. 
* **IV Spike Exits**: The engine uses a rudimentary IV-drop exit but doesn't instantly scale out on massive IV *spikes* (e.g., unexpected news events mid-session).

## Changelog
* **v1.0.3**: Cleaned up the entire Python codebase (removed dead variables, stripped noisy comments, standardized formatting across all 5 files).
* **v1.0.2**: Fixed the Streamlit Dashboard UI to render completely custom pure-HTML metric cards (added green/red accent backgrounds natively bypassing Streamlit `st.metric` limitations). Restored the transparent header bar.
* **v1.0.1**: Consolidated the "Trade Analytics" and "Risk Summary" tabs into a unified "P&L Analysis" tab. Improved performance heatmaps. Replaced "Gross P&L" with "Total Return %" based on ₹1.75L starting capital.
* **v1.0.0**: Initial structural overhaul. Implemented retail-hardened features (Economic Gates, Cooldowns, Theta Checkpoints). Changed default data file to `FINAL_NIFTY_MASTER_ATM.csv`.