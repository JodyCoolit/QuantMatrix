from datetime import datetime
import pandas as pd
import yfinance as yf

from .data_source import DataSource


class YahooFinanceSource(DataSource):
    """Yahoo Finance implementation of DataSource."""
    
    def fetch_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.
        
        Args:
            symbol: The ticker symbol
            timeframe: The data interval (1d, 1h, etc.)
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data
        """
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d',
            '1wk': '1wk',
            '1mo': '1mo'
        }
        
        yf_interval = interval_map.get(timeframe, '1d')
        
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=yf_interval,
                progress=False
            )
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            data = data.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'adj close': 'adjusted_close',
                'volume': 'volume'
            })
            
            return data
        except Exception as e:
            print(f"Error fetching data from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str) -> pd.DataFrame:
        """
        Get the latest data for a symbol from Yahoo Finance.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            DataFrame with the latest market data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            
            if data.empty:
                return pd.DataFrame()
            
            # Get the latest data point
            latest = data.iloc[-1:].copy()
            
            # Standardize column names
            latest.columns = [col.lower() for col in latest.columns]
            latest = latest.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'adj close': 'adjusted_close',
                'volume': 'volume'
            })
            
            return latest
        except Exception as e:
            print(f"Error fetching realtime data from Yahoo Finance: {e}")
            return pd.DataFrame() 