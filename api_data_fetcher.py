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
    """Fetches Nifty market data from NSE/Yahoo Finance for backtesting."""

    def __init__(self, use_yfinance=True):
        self.use_yfinance = use_yfinance and YFINANCE_AVAILABLE
        if self.use_yfinance and not YFINANCE_AVAILABLE:
            warnings.warn("yfinance not available, falling back to NSEpy")
            self.use_yfinance = False
        if not self.use_yfinance and not NSEPY_AVAILABLE:
            raise ImportError("Neither yfinance nor NSEpy installed. pip install yfinance")

    def fetch_nifty_data_yfinance(self, start_date: date, end_date: date, interval: str = '1d') -> pd.DataFrame:
        try:
            print(f"Fetching Nifty data from Yahoo Finance ({start_date} to {end_date}, interval={interval})...")
            ticker = yf.Ticker("^NSEI")
            data = ticker.history(start=start_date, end=end_date, interval=interval)

            if data is None or data.empty:
                raise ValueError(f"No data returned for Nifty 50 between {start_date} and {end_date}")

            data = data.reset_index()
            if 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'Date'})

            print(f"✓ Fetched {len(data)} records from Yahoo Finance")
            return data
        except Exception as e:
            raise ConnectionError(f"Failed to fetch from Yahoo Finance: {e}")

    def fetch_nifty_data(self, start_date: date, end_date: date, symbol: str = "NIFTY 50") -> pd.DataFrame:
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        if end_date > date.today():
            raise ValueError("End date cannot be in the future")

        days_diff = (end_date - start_date).days
        if days_diff > 365:
            warnings.warn(f"Requesting {days_diff} days of data. This might take a while.")

        if self.use_yfinance:
            return self.fetch_nifty_data_yfinance(start_date, end_date, interval='1d')

        try:
            print(f"Fetching Nifty data from {start_date} to {end_date}...")
            data = get_history(symbol=symbol, start=start_date, end=end_date, index=True)

            if data is None or data.empty:
                raise ValueError(f"No data returned for {symbol} between {start_date} and {end_date}")

            data = data.reset_index()
            print(f"✓ Fetched {len(data)} records")
            return data
        except Exception as e:
            raise ConnectionError(f"Failed to fetch from NSE: {e}")

    def format_data_for_strategy(self, raw_data: pd.DataFrame, timeframe: str = '5min') -> pd.DataFrame:
        """Format API data to match strategy's expected format (timestamp, close, high, low)."""
        data = raw_data.copy()

        if 'Date' in data.columns:
            data['timestamp'] = pd.to_datetime(data['Date'])
        elif 'date' in data.columns:
            data['timestamp'] = pd.to_datetime(data['date'])
        else:
            data['timestamp'] = pd.to_datetime(data.index)

        for old_col, new_col in {'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open'}.items():
            if old_col in data.columns:
                data[new_col] = data[old_col]

        if timeframe.lower() in ['1d', 'day', 'daily']:
            return data[['timestamp', 'close', 'high', 'low']].sort_values('timestamp').reset_index(drop=True)

        # Simulate intraday bars from daily OHLC
        print(f"⚠ NSEpy provides daily data. Simulating {timeframe} bars by interpolation...")

        bars_per_day = {'1min': 375, '3min': 125, '5min': 75, '15min': 25, '30min': 13, '1h': 7}.get(timeframe, 75)
        intraday_data = []

        for _, row in data.iterrows():
            base_date = pd.to_datetime(row['timestamp']).date()
            open_price = row.get('open', row['close'])
            high_price, low_price, close_price = row['high'], row['low'], row['close']

            for bar_idx in range(bars_per_day):
                start_hour, start_minute = 9, 15
                if timeframe == '1min':
                    bar_time = start_hour * 60 + start_minute + bar_idx
                elif timeframe == '5min':
                    bar_time = start_hour * 60 + start_minute + (bar_idx * 5)
                elif timeframe == '15min':
                    bar_time = start_hour * 60 + start_minute + (bar_idx * 15)
                else:
                    bar_time = start_hour * 60 + start_minute + (bar_idx * 5)

                hours, minutes = bar_time // 60, bar_time % 60
                timestamp = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=hours, minutes=minutes)

                progress = bar_idx / bars_per_day
                base_close = open_price + (close_price - open_price) * progress
                volatility = (high_price - low_price) * 0.3
                bar_close = np.clip(base_close + np.random.normal(0, volatility), low_price, high_price)

                intraday_data.append({
                    'timestamp': timestamp,
                    'close': bar_close,
                    'high': bar_close * (1 + np.random.uniform(0, 0.002)),
                    'low': bar_close * (1 - np.random.uniform(0, 0.002))
                })

        result = pd.DataFrame(intraday_data).sort_values('timestamp').reset_index(drop=True)
        print(f"✓ Generated {len(result)} intraday bars from {len(data)} daily bars")
        return result

    def fetch_and_format(self, start_date: date, end_date: date, timeframe: str = '5min',
                         symbol: str = "NIFTY 50") -> Tuple[pd.DataFrame, dict]:
        try:
            raw_data = self.fetch_nifty_data(start_date, end_date, symbol)
            formatted_data = self.format_data_for_strategy(raw_data, timeframe)
            metadata = {
                'source': 'NSE', 'symbol': symbol,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'timeframe': timeframe, 'total_records': len(formatted_data),
                'daily_records': len(raw_data), 'success': True
            }
            return formatted_data, metadata
        except Exception as e:
            raise


def fetch_nifty_data(start_date: date, end_date: date, timeframe: str = '5min') -> pd.DataFrame:
    """Convenience function to fetch and format Nifty data."""
    fetcher = APIDataFetcher()
    data, _ = fetcher.fetch_and_format(start_date, end_date, timeframe)
    return data
