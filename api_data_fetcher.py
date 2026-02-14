"""
API Data Fetcher for Indian Market Data

This module fetches Nifty spot price data from NSE using the NSEpy library.
Provides historical data for backtesting the gamma scalping strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional, Tuple
import warnings

try:
    from nsepy import get_history
    from nsepy.symbols import get_symbol_list
    NSEPY_AVAILABLE = True
except ImportError:
    NSEPY_AVAILABLE = False
    warnings.warn("NSEpy not installed. Install with: pip install nsepy")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not installed. Install with: pip install yfinance")


class APIDataFetcher:
    """Fetches market data from NSE API"""
    
    def __init__(self, use_yfinance=True):
        """
        Initialize the API data fetcher
        
        Args:
            use_yfinance: If True, use yfinance (more reliable). Otherwise use NSEpy.
        """
        self.use_yfinance = use_yfinance and YFINANCE_AVAILABLE
        
        if self.use_yfinance:
            if not YFINANCE_AVAILABLE:
                warnings.warn("yfinance not available, falling back to NSEpy")
                self.use_yfinance = False
        
        if not self.use_yfinance and not NSEPY_AVAILABLE:
            raise ImportError("Neither yfinance nor NSEpy are installed. Please install at least one: pip install yfinance")
    
    
    def fetch_nifty_data_yfinance(
        self,
        start_date: date,
        end_date: date,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch Nifty data using yfinance (Yahoo Finance)
        
        Args:
            start_date: Start date
            end_date: End date
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
        
        Returns:
            DataFrame with OHLC data
        """
        try:
            print(f"Fetching Nifty data from Yahoo Finance ({start_date} to {end_date}, interval={interval})...")
            
            # Nifty 50 symbol on Yahoo Finance
            ticker = yf.Ticker("^NSEI")
            
            # Fetch data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if data is None or data.empty:
                raise ValueError(f"No data returned for Nifty 50 between {start_date} and {end_date}")
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns to match our format
            if 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'Date'})
            
            print(f"✓ Successfully fetched {len(data)} records from Yahoo Finance")
            return data
            
        except Exception as e:
            error_msg = f"Failed to fetch data from Yahoo Finance: {str(e)}"
            print(f"✗ {error_msg}")
            raise ConnectionError(error_msg)
    
    def fetch_nifty_data(
        self, 
        start_date: date, 
        end_date: date,
        symbol: str = "NIFTY 50"
    ) -> pd.DataFrame:
        """
        Fetch Nifty spot price data from NSE
        
        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            symbol: Index symbol (default: "NIFTY 50")
        
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        
        Raises:
            ValueError: If date range is invalid
            ConnectionError: If API call fails
        """
        # Validate dates
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        
        if end_date > date.today():
            raise ValueError("End date cannot be in the future")
        
        # Check if requesting too much data
        days_diff = (end_date - start_date).days
        if days_diff > 365:
            warnings.warn(f"Requesting {days_diff} days of data. This might take a while.")
        
        # Use yfinance if available (more reliable)
        if self.use_yfinance:
            return self.fetch_nifty_data_yfinance(start_date, end_date, interval='1d')
        
        # Fallback to NSEpy
        
        try:
            # Fetch data from NSE
            print(f"Fetching Nifty data from {start_date} to {end_date}...")
            data = get_history(
                symbol=symbol,
                start=start_date,
                end=end_date,
                index=True  # This is an index, not a stock
            )
            
            if data is None or data.empty:
                raise ValueError(f"No data returned for {symbol} between {start_date} and {end_date}")
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            print(f"✓ Successfully fetched {len(data)} records")
            return data
            
        except Exception as e:
            error_msg = f"Failed to fetch data from NSE: {str(e)}"
            print(f"✗ {error_msg}")
            raise ConnectionError(error_msg)
    
    def format_data_for_strategy(
        self, 
        raw_data: pd.DataFrame, 
        timeframe: str = '5min'
    ) -> pd.DataFrame:
        """
        Format API data to match the strategy's expected format
        
        Args:
            raw_data: Raw data from NSE API (daily OHLC)
            timeframe: Target timeframe (1min, 5min, 15min, 1D)
        
        Returns:
            DataFrame with columns: timestamp, close, high, low
        
        Note:
            NSEpy provides daily data. For intraday timeframes, we'll simulate
            intraday data by interpolating between daily bars. For real intraday
            data, a premium API would be needed.
        """
        # Create a copy to avoid modifying original
        data = raw_data.copy()
        
        # Ensure Date column exists and is datetime
        if 'Date' in data.columns:
            data['timestamp'] = pd.to_datetime(data['Date'])
        elif 'date' in data.columns:
            data['timestamp'] = pd.to_datetime(data['date'])
        else:
            # If Date is the index
            data['timestamp'] = pd.to_datetime(data.index)
        
        # Standardize column names
        column_mapping = {
            'Close': 'close',
            'High': 'high',
            'Low': 'low',
            'Open': 'open'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                data[new_col] = data[old_col]
        
        # If timeframe is daily, just return the daily data
        if timeframe.lower() in ['1d', 'day', 'daily']:
            result = data[['timestamp', 'close', 'high', 'low']].copy()
            result = result.sort_values('timestamp').reset_index(drop=True)
            return result
        
        # For intraday timeframes, simulate by creating multiple bars per day
        print(f"⚠ NSEpy provides daily data. Simulating {timeframe} bars by interpolation...")
        
        # Determine bars per day based on timeframe
        timeframe_mapping = {
            '1min': 375,   # 9:15 AM to 3:30 PM = 375 minutes
            '3min': 125,
            '5min': 75,
            '15min': 25,
            '30min': 13,
            '1h': 7
        }
        
        bars_per_day = timeframe_mapping.get(timeframe, 75)  # Default to 5min
        
        # Generate intraday data
        intraday_data = []
        
        for idx, row in data.iterrows():
            base_date = pd.to_datetime(row['timestamp']).date()
            open_price = row.get('open', row['close'])
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # Create price path for the day
            # Start at open, random walk to close, respecting high/low
            for bar_idx in range(bars_per_day):
                # Calculate timestamp for this bar
                start_hour = 9
                start_minute = 15
                
                if timeframe == '1min':
                    bar_time = start_hour * 60 + start_minute + bar_idx
                elif timeframe == '5min':
                    bar_time = start_hour * 60 + start_minute + (bar_idx * 5)
                elif timeframe == '15min':
                    bar_time = start_hour * 60 + start_minute + (bar_idx * 15)
                else:
                    bar_time = start_hour * 60 + start_minute + (bar_idx * 5)
                
                hours = bar_time // 60
                minutes = bar_time % 60
                
                timestamp = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=hours, minutes=minutes)
                
                # Interpolate price
                progress = bar_idx / bars_per_day
                base_close = open_price + (close_price - open_price) * progress
                
                # Add some randomness within high/low bounds
                volatility = (high_price - low_price) * 0.3
                noise = np.random.normal(0, volatility)
                bar_close = base_close + noise
                
                # Ensure within daily high/low
                bar_close = max(min(bar_close, high_price), low_price)
                
                # Create high/low for this bar (small variation)
                bar_high = bar_close * (1 + np.random.uniform(0, 0.002))
                bar_low = bar_close * (1 - np.random.uniform(0, 0.002))
                
                intraday_data.append({
                    'timestamp': timestamp,
                    'close': bar_close,
                    'high': bar_high,
                    'low': bar_low
                })
        
        result = pd.DataFrame(intraday_data)
        result = result.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Generated {len(result)} intraday bars from {len(data)} daily bars")
        return result
    
    def fetch_and_format(
        self,
        start_date: date,
        end_date: date,
        timeframe: str = '5min',
        symbol: str = "NIFTY 50"
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Fetch and format data in one call
        
        Args:
            start_date: Start date
            end_date: End date
            timeframe: Timeframe for strategy
            symbol: Index symbol
        
        Returns:
            Tuple of (formatted_data, metadata_dict)
        """
        try:
            # Fetch raw data
            raw_data = self.fetch_nifty_data(start_date, end_date, symbol)
            
            # Format for strategy
            formatted_data = self.format_data_for_strategy(raw_data, timeframe)
            
            # Create metadata
            metadata = {
                'source': 'NSE (via NSEpy)',
                'symbol': symbol,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'timeframe': timeframe,
                'total_records': len(formatted_data),
                'daily_records': len(raw_data),
                'success': True
            }
            
            return formatted_data, metadata
            
        except Exception as e:
            # Return error metadata
            metadata = {
                'source': 'NSE (via NSEpy)',
                'symbol': symbol,
                'error': str(e),
                'success': False
            }
            raise


# Convenience function for easy import
def fetch_nifty_data(start_date: date, end_date: date, timeframe: str = '5min') -> pd.DataFrame:
    """
    Convenience function to fetch and format Nifty data
    
    Args:
        start_date: Start date
        end_date: End date
        timeframe: Timeframe (1min, 5min, 15min, 1D)
    
    Returns:
        Formatted DataFrame ready for strategy
    """
    fetcher = APIDataFetcher()
    data, metadata = fetcher.fetch_and_format(start_date, end_date, timeframe)
    return data
