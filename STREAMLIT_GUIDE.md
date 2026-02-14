# Gamma Scalping Streamlit App - Quick Start Guide

## 🚀 Getting Started

### Installation

1. **Install Dependencies**
   ```powershell
   cd c:\Users\asus\OneDrive\Desktop\round2\files
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```powershell
   streamlit run app.py
   ```

The app will automatically open in your default browser at `http://localhost:8501`

## 📋 Features

### Interactive Dashboard
- **Real-time backtesting** with configurable parameters
- **Multiple strategy presets**: Conservative, Aggressive, High Frequency
- **Interactive charts** using Plotly (zoom, pan, hover)
- **Performance metrics** dashboard
- **Trade log** with detailed transaction history
- **Greeks analysis** visualization
- **CSV export** functionality

### Tabs Overview

1. **📊 Dashboard**
   - Performance summary metrics (P&L, win rate, profit factor, max drawdown)
   - Exit reasons breakdown
   - P&L distribution histogram
   - Hedging statistics

2. **📈 Charts**
   - Price chart with entry/exit markers
   - Cumulative P&L with drawdown
   - Portfolio delta control visualization

3. **📋 Trade Log**
   - Complete trade history table
   - Export to CSV functionality

4. **🎯 Greeks Analysis**
   - Gamma & Vega evolution
   - Theta decay analysis
   - P&L components breakdown

5. **📖 Documentation**
   - Strategy explanation
   - Parameter tuning guide
   - Risk warnings
   - Tips for success

## ⚙️ Configuration

### Strategy Parameters

- **Delta Threshold** (0.05-0.30): Rehedge trigger level
- **IV Entry Percentile** (10-50): Entry signal threshold
- **IV Exit Percentile** (50-90): Exit signal threshold
- **Profit Target** (10%-100%): Take profit level
- **Max Loss** (-50% to -10%): Stop loss level
- **Days to Expiry** (2-21): Option time to expiry
- **Risk Free Rate** (2%-10%): For Greeks calculation

### Data Configuration

- **Timeframe**: 1min, 3min, or 5min candles
- **Backtest Days**: 10-90 days
- **Data Source**: Generate sample data or upload CSV

### CSV Upload Format

If uploading custom data, CSV should have these columns:
- `timestamp`: Date/time (parseable format)
- `close`: Closing price
- `high`: High price
- `low`: Low price

Example:
```csv
timestamp,close,high,low
2024-01-01 09:15:00,21500,21520,21490
2024-01-01 09:20:00,21510,21530,21500
...
```

## 🎯 Using the App

### Basic Workflow

1. **Select a Preset** (optional)
   - Click "Conservative", "Aggressive", or "High Freq" in sidebar
   - Parameters will auto-populate

2. **Adjust Parameters**
   - Fine-tune using sliders in sidebar
   - See real-time parameter updates

3. **Configure Data**
   - Choose timeframe and backtest period
   - Generate sample data or upload CSV

4. **Run Backtest**
   - Click "🚀 Run Backtest" button
   - Wait for completion (usually 5-15 seconds)

5. **Analyze Results**
   - Explore different tabs
   - Download CSVs for further analysis

### Strategy Presets

**Conservative**
- Lower delta threshold (0.10)
- Lower profit target (30%)
- Tighter stop loss (-20%)
- More selective entry (25th percentile)
- Best for: Stable markets, risk-averse traders

**Aggressive**
- Higher delta threshold (0.20)
- Higher profit target (75%)
- Wider stop loss (-40%)
- Less selective entry (35th percentile)
- Best for: Volatile markets, experienced traders

**High Frequency**
- Very low delta threshold (0.08)
- Quick profit target (25%)
- Tight stop loss (-15%)
- Best for: Active scalping, lower timeframes

## 📊 Interpreting Results

### Key Metrics

- **Total P&L**: Overall profit/loss
- **Win Rate**: Percentage of profitable trades (aim for >50%)
- **Profit Factor**: Avg Win / |Avg Loss| (aim for >1.5)
- **Max Drawdown**: Worst peak-to-trough decline

### Good Performance Indicators

✅ Win rate above 50%  
✅ Profit factor above 1.5  
✅ Positive total P&L  
✅ Max drawdown <30% of total P&L  
✅ Consistent hedging keeping delta near zero  

### Warning Signs

⚠️ Win rate below 40%  
⚠️ Profit factor below 1.0  
⚠️ Large drawdowns (>40% of capital)  
⚠️ Too many rehedges (high transaction costs)  
⚠️ Delta frequently breaching threshold  

## 🔧 Troubleshooting

### App Won't Start

```powershell
# Reinstall dependencies
pip install --upgrade streamlit pandas numpy scipy matplotlib plotly

# Try running with specific Python version
python -m streamlit run app.py
```

### Import Errors

Make sure all files are in the same directory:
- `app.py`
- `gamma_scalping_system.py`
- `config.py`
- `analyze_results.py`
- `requirements.txt`

### No Trades Executed

Try adjusting:
- Increase IV Entry Percentile (make entry less selective)
- Increase backtest days (more opportunities)
- Check that sample data is generating correctly

### Charts Not Displaying

- Ensure Plotly is installed: `pip install plotly`
- Check browser console for JavaScript errors
- Try refreshing the page

## 💡 Tips

1. **Start Conservative**: Use conservative preset for first runs
2. **Compare Strategies**: Run multiple configurations and compare
3. **Check Greeks**: Ensure gamma is positive and theta is manageable
4. **Monitor Delta**: Delta should stay within threshold most of the time
5. **Export Data**: Download CSVs for deeper analysis in Excel/Python

## 🎓 Learning Resources

- Review the **Documentation** tab in the app
- Read the original `README.md` for strategy details
- Study the trade log to understand entry/exit logic
- Experiment with different parameters

## ⚠️ Important Reminders

- **This is for educational purposes only**
- **Options trading involves substantial risk**
- **Always paper trade before live trading**
- **Past performance ≠ future results**
- **Test thoroughly across different market conditions**

## 📞 Support

For issues:
1. Check this guide
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify file paths are correct

---

**Happy Gamma Scalping! 🎲📈**
