"""
Configuration file for Gamma Scalping Strategy

Modify these parameters to customize the strategy behavior
"""

# ============================================================================
# STRATEGY PARAMETERS
# ============================================================================

# Delta Hedging Parameters
DELTA_THRESHOLD = 0.15          # Rehedge when |portfolio delta| > 0.15
                                # Lower = more frequent hedging = higher transaction costs
                                # Higher = less frequent hedging = more directional risk

# Entry/Exit Conditions
IV_ENTRY_PERCENTILE = 30        # Enter when IV below 30th percentile (low IV)
IV_EXIT_PERCENTILE = 70         # Exit when IV above 70th percentile (high IV)
PROFIT_TARGET = 0.50            # Exit at 50% profit on premium paid
MAX_LOSS = -0.30                # Stop loss at -30% of premium paid

# Option Parameters
TIME_TO_EXPIRY = 7              # Days to expiry (7 = weekly options)
RISK_FREE_RATE = 0.06           # Annual risk-free rate (6%)
STRIKE_INTERVAL = 50            # Nifty strike interval (typically 50)

# ============================================================================
# DATA PARAMETERS
# ============================================================================

TIMEFRAME = '5min'              # Options: '1min', '3min', '5min'
BACKTEST_DAYS = 30              # Number of days to backtest

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

MAX_POSITIONS = 1               # Maximum concurrent straddles
POSITION_SIZE = 1               # Lot size multiplier (1 = 1 lot)

# ============================================================================
# ADVANCED PARAMETERS
# ============================================================================

# Volatility Calculation
HV_WINDOW = 20                  # Historical volatility lookback (bars)
IV_HV_MULTIPLIER = 1.2          # IV typically trades at premium to HV

# Transaction Costs (for live trading)
FUTURES_COMMISSION = 20         # Per futures contract (₹)
OPTIONS_COMMISSION = 50         # Per options contract (₹)
SLIPPAGE_BPS = 5                # Slippage in basis points

# Greeks Calculation
MIN_GAMMA_THRESHOLD = 0.001     # Minimum gamma to maintain position
MIN_TIME_VALUE = 10             # Minimum option time value (₹)

# ============================================================================
# STRATEGY VARIATIONS
# ============================================================================

# You can test different variations:

# Conservative (Lower Risk, Lower Returns)
CONSERVATIVE = {
    'DELTA_THRESHOLD': 0.10,
    'PROFIT_TARGET': 0.30,
    'MAX_LOSS': -0.20,
    'IV_ENTRY_PERCENTILE': 25
}

# Aggressive (Higher Risk, Higher Returns)
AGGRESSIVE = {
    'DELTA_THRESHOLD': 0.20,
    'PROFIT_TARGET': 0.75,
    'MAX_LOSS': -0.40,
    'IV_ENTRY_PERCENTILE': 35
}

# High Frequency (More Trades, More Hedging)
HIGH_FREQUENCY = {
    'DELTA_THRESHOLD': 0.08,
    'PROFIT_TARGET': 0.25,
    'MAX_LOSS': -0.15,
    'TIMEFRAME': '1min'
}

# ============================================================================
# MARKET CONDITIONS
# ============================================================================

# Different parameters work better in different market conditions:

# Low Volatility Environment
LOW_VOL_PARAMS = {
    'IV_ENTRY_PERCENTILE': 40,  # More lenient entry
    'TIME_TO_EXPIRY': 14,        # Longer dated options
    'DELTA_THRESHOLD': 0.12
}

# High Volatility Environment  
HIGH_VOL_PARAMS = {
    'IV_ENTRY_PERCENTILE': 20,  # Very selective entry
    'TIME_TO_EXPIRY': 3,         # Shorter dated options
    'DELTA_THRESHOLD': 0.18      # Less frequent hedging
}

# Trending Market
TRENDING_PARAMS = {
    'DELTA_THRESHOLD': 0.12,    # More frequent hedging
    'PROFIT_TARGET': 0.40,       # Take profits quicker
    'MAX_LOSS': -0.25
}

# Range-Bound Market
RANGE_BOUND_PARAMS = {
    'DELTA_THRESHOLD': 0.18,    # Less hedging in low movement
    'PROFIT_TARGET': 0.60,       # Let winners run
    'MAX_LOSS': -0.35
}
