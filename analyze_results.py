"""
Interactive Dashboard for Gamma Scalping Analysis

This creates a detailed analysis dashboard with:
- Trade breakdown by entry/exit conditions
- Greeks evolution over time
- Hedging efficiency analysis
- Risk metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def analyze_trade_log(trade_log_file='gamma_scalping_trades.csv'):
    """Analyze the trade log in detail"""
    
    df = pd.read_csv(trade_log_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Separate different action types
    entries = df[df['action'] == 'ENTER_STRADDLE'].copy()
    exits = df[df['action'].str.contains('EXIT')].copy()
    hedges = df[df['action'] == 'REHEDGE'].copy()
    
    print("=" * 80)
    print("DETAILED TRADE ANALYSIS")
    print("=" * 80)
    
    # Entry Analysis
    print("\n📊 ENTRY ANALYSIS:")
    print(f"  Total Straddles Entered: {len(entries)}")
    if len(entries) > 0:
        print(f"  Average Entry IV: {entries['iv'].mean():.2f}%")
        print(f"  IV Range: {entries['iv'].min():.2f}% - {entries['iv'].max():.2f}%")
        print(f"  Average Entry Price: ₹{entries['spot'].mean():.2f}")
        print(f"  Average Premium Paid: ₹{entries['total_premium'].mean():.2f}")
    
    # Exit Analysis
    print("\n📊 EXIT ANALYSIS:")
    exit_reasons = exits['action'].value_counts()
    print(f"  Total Exits: {len(exits)}")
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/len(exits)*100:.1f}%)")
    
    if len(exits) > 0:
        profitable_exits = exits[exits['total_pnl'] > 0]
        print(f"\n  Profitable Exits: {len(profitable_exits)} ({len(profitable_exits)/len(exits)*100:.1f}%)")
        print(f"  Average Exit IV: {exits['iv'].mean():.2f}%")
    
    # Hedging Analysis
    print("\n📊 HEDGING ANALYSIS:")
    print(f"  Total Rehedges: {len(hedges)}")
    if len(hedges) > 0:
        print(f"  Average |Delta| at Rehedge: {hedges['portfolio_delta'].abs().mean():.3f}")
        print(f"  Max |Delta| Reached: {hedges['portfolio_delta'].abs().max():.3f}")
        
        # Analyze hedging profitability
        hedge_pnl_sum = hedges['hedge_pnl'].sum() if 'hedge_pnl' in hedges else 0
        print(f"  Net Hedging P&L: ₹{hedge_pnl_sum:.2f}")
    
    # P&L Distribution
    print("\n📊 P&L DISTRIBUTION:")
    if len(exits) > 0:
        pnl_values = exits['total_pnl']
        print(f"  Mean P&L: ₹{pnl_values.mean():.2f}")
        print(f"  Median P&L: ₹{pnl_values.median():.2f}")
        print(f"  Std Dev: ₹{pnl_values.std():.2f}")
        print(f"  25th Percentile: ₹{pnl_values.quantile(0.25):.2f}")
        print(f"  75th Percentile: ₹{pnl_values.quantile(0.75):.2f}")
    
    # IV Analysis
    print("\n📊 IMPLIED VOLATILITY ANALYSIS:")
    if len(entries) > 0 and len(exits) > 0:
        # Match entries to exits
        avg_entry_iv = entries['iv'].mean()
        avg_exit_iv = exits['iv'].mean()
        iv_change = avg_exit_iv - avg_entry_iv
        print(f"  Average Entry IV: {avg_entry_iv:.2f}%")
        print(f"  Average Exit IV: {avg_exit_iv:.2f}%")
        print(f"  Average IV Change: {iv_change:+.2f}%")
        
        if iv_change > 0:
            print(f"  → IV expanded on average (favorable for long vega)")
        else:
            print(f"  → IV contracted on average (unfavorable for long vega)")
    
    print("\n" + "=" * 80)
    
    return entries, exits, hedges


def create_detailed_charts(backtest_file='gamma_scalping_backtest.csv',
                          trade_log_file='gamma_scalping_trades.csv'):
    """Create comprehensive analysis charts"""
    
    # Load data
    backtest = pd.read_csv(backtest_file)
    backtest['timestamp'] = pd.to_datetime(backtest['timestamp'])
    
    trades = pd.read_csv(trade_log_file)
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. P&L Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    exits = trades[trades['action'].str.contains('EXIT')]
    if len(exits) > 0:
        pnl_values = exits['total_pnl']
        ax1.hist(pnl_values, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax1.axvline(pnl_values.mean(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean: ₹{pnl_values.mean():.0f}')
        ax1.set_xlabel('P&L per Trade (₹)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('P&L Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Win/Loss Breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    if len(exits) > 0:
        exit_reasons = exits['action'].value_counts()
        colors = ['green' if 'PROFIT' in r else 'red' if 'LOSS' in r else 'blue' 
                 for r in exit_reasons.index]
        ax2.bar(range(len(exit_reasons)), exit_reasons.values, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(exit_reasons)))
        ax2.set_xticklabels([r.replace('EXIT_STRADDLE_', '') for r in exit_reasons.index], 
                           rotation=45, ha='right')
        ax2.set_ylabel('Number of Trades')
        ax2.set_title('Exit Reasons Breakdown')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative P&L with Drawdown
    ax3 = fig.add_subplot(gs[1, :])
    if 'total_pnl' in backtest.columns:
        cumulative_pnl = backtest['total_pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        
        ax3.plot(backtest['timestamp'], cumulative_pnl, label='Cumulative P&L', 
                color='green', linewidth=2)
        ax3.fill_between(backtest['timestamp'], cumulative_pnl, running_max, 
                         alpha=0.3, color='red', label='Drawdown')
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax3.set_ylabel('P&L (₹)')
        ax3.set_title('Cumulative P&L and Drawdown')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Greeks Evolution
    ax4 = fig.add_subplot(gs[2, 0])
    if 'gamma' in backtest.columns and 'vega' in backtest.columns:
        ax4.plot(backtest['timestamp'], backtest['gamma'], label='Gamma', 
                color='orange', alpha=0.8)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(backtest['timestamp'], backtest['vega'], label='Vega', 
                     color='purple', alpha=0.8)
        ax4.set_ylabel('Gamma', color='orange')
        ax4_twin.set_ylabel('Vega', color='purple')
        ax4.set_title('Positive Gamma & Vega Exposure')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
    
    # 5. Delta Control
    ax5 = fig.add_subplot(gs[2, 1])
    if 'portfolio_delta' in backtest.columns:
        ax5.plot(backtest['timestamp'], backtest['portfolio_delta'], 
                label='Portfolio Delta', color='blue', linewidth=1.5)
        ax5.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)
        ax5.axhline(0.15, color='red', linestyle='--', alpha=0.5, label='Rehedge Threshold')
        ax5.axhline(-0.15, color='red', linestyle='--', alpha=0.5)
        ax5.fill_between(backtest['timestamp'], -0.15, 0.15, alpha=0.1, color='green')
        ax5.set_ylabel('Delta')
        ax5.set_title('Delta Neutrality Control (Target: 0 ± 0.15)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Theta vs Gamma P&L
    ax6 = fig.add_subplot(gs[3, 0])
    if 'theta' in backtest.columns:
        cumulative_theta = backtest['theta'].cumsum()
        ax6.plot(backtest['timestamp'], cumulative_theta, label='Cumulative Theta (Cost)', 
                color='red', linewidth=2)
        if 'total_pnl' in backtest.columns:
            ax6.plot(backtest['timestamp'], backtest['total_pnl'].cumsum(), 
                    label='Total P&L', color='green', linewidth=2, alpha=0.7)
        ax6.set_ylabel('Cumulative (₹)')
        ax6.set_title('Theta Decay Cost vs Total P&L')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # 7. IV at Entry vs Exit
    ax7 = fig.add_subplot(gs[3, 1])
    entries = trades[trades['action'] == 'ENTER_STRADDLE']
    exits_iv = trades[trades['action'].str.contains('EXIT')]
    
    if len(entries) > 0 and len(exits_iv) > 0:
        # Pair entries with their corresponding exits
        entry_ivs = []
        exit_ivs = []
        
        for i in range(min(len(entries), len(exits_iv))):
            entry_ivs.append(entries.iloc[i]['iv'])
            exit_ivs.append(exits_iv.iloc[i]['iv'])
        
        x = np.arange(len(entry_ivs))
        width = 0.35
        
        ax7.bar(x - width/2, entry_ivs, width, label='Entry IV', alpha=0.7, color='blue')
        ax7.bar(x + width/2, exit_ivs, width, label='Exit IV', alpha=0.7, color='orange')
        ax7.set_xlabel('Trade Number')
        ax7.set_ylabel('Implied Volatility (%)')
        ax7.set_title('IV at Entry vs Exit (Want IV to Rise)')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Gamma Scalping Strategy - Detailed Analysis', fontsize=16, fontweight='bold')
    plt.savefig('gamma_scalping_detailed_analysis.png', dpi=150, bbox_inches='tight')
    print("\nDetailed analysis chart saved to: gamma_scalping_detailed_analysis.png")
    
    return fig


def analyze_hedging_efficiency(trade_log_file='gamma_scalping_trades.csv'):
    """Analyze how effective the hedging was"""
    
    trades = pd.read_csv(trade_log_file)
    
    entries = trades[trades['action'] == 'ENTER_STRADDLE']
    exits = trades[trades['action'].str.contains('EXIT')]
    hedges = trades[trades['action'] == 'REHEDGE']
    
    print("\n" + "=" * 80)
    print("HEDGING EFFICIENCY ANALYSIS")
    print("=" * 80)
    
    if len(exits) > 0:
        total_options_pnl = exits['options_pnl'].sum() if 'options_pnl' in exits else 0
        total_futures_pnl = exits['futures_pnl'].sum() if 'futures_pnl' in exits else 0
        total_hedging_cost = exits['hedging_cost'].sum() if 'hedging_cost' in exits else 0
        
        print(f"\nP&L Breakdown:")
        print(f"  Options P&L: ₹{total_options_pnl:,.2f}")
        print(f"  Futures P&L: ₹{total_futures_pnl:,.2f}")
        print(f"  Hedging Costs: ₹{total_hedging_cost:,.2f}")
        print(f"  Total P&L: ₹{(total_options_pnl + total_futures_pnl + total_hedging_cost):,.2f}")
        
        print(f"\nHedging Contribution:")
        total_pnl = total_options_pnl + total_futures_pnl + total_hedging_cost
        if total_pnl != 0:
            hedge_contrib = (total_futures_pnl + total_hedging_cost) / total_pnl * 100
            print(f"  Hedging contributed {hedge_contrib:.1f}% of total P&L")
        
        if len(hedges) > 0:
            trades_count = len(exits)
            avg_hedges = len(hedges) / trades_count
            print(f"\nHedging Frequency:")
            print(f"  Average hedges per trade: {avg_hedges:.1f}")
            print(f"  Total hedge events: {len(hedges)}")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("\n🔍 GAMMA SCALPING - DETAILED ANALYSIS\n")
    
    # Analyze trade log
    entries, exits, hedges = analyze_trade_log()
    
    # Analyze hedging efficiency
    analyze_hedging_efficiency()
    
    # Create detailed charts
    fig = create_detailed_charts()
    
    print("\n✅ Analysis complete! Check the generated charts and CSV files.")
    print("\nFiles generated:")
    print("  1. gamma_scalping_results.png - Main performance chart")
    print("  2. gamma_scalping_detailed_analysis.png - Detailed analysis")
    print("  3. gamma_scalping_trades.csv - Trade log")
    print("  4. gamma_scalping_backtest.csv - Tick-by-tick results")
