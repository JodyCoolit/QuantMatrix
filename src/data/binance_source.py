from datetime import datetime
import pandas as pd
from binance.client import Client

from .data_source import DataSource


class BinanceSource(DataSource):
    """Binance implementation of DataSource."""
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        """
        Initialize the Binance data source.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
        """
        self.client = Client(api_key, api_secret)
    
    def fetch_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical data from Binance.
        
        Args:
            symbol: The trading pair symbol (e.g., 'BTCUSDT')
            timeframe: The data interval (1m, 5m, 15m, etc.)
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data
        """
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '3m': Client.KLINE_INTERVAL_3MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '2h': Client.KLINE_INTERVAL_2HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '6h': Client.KLINE_INTERVAL_6HOUR,
            '8h': Client.KLINE_INTERVAL_8HOUR,
            '12h': Client.KLINE_INTERVAL_12HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
            '3d': Client.KLINE_INTERVAL_3DAY,
            '1w': Client.KLINE_INTERVAL_1WEEK,
            '1mo': Client.KLINE_INTERVAL_1MONTH
        }
        
        binance_interval = interval_map.get(timeframe, Client.KLINE_INTERVAL_1DAY)
        
        try:
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)
            
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=binance_interval,
                start_str=start_ts,
                end_str=end_ts
            )
            
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            
            df = pd.DataFrame(klines, columns=columns)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Select only the OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
        except Exception as e:
            print(f"Error fetching data from Binance: {e}")
            return pd.DataFrame()
    
    def get_realtime_data(self, symbol: str) -> pd.DataFrame:
        """
        Get the latest data for a symbol from Binance.
        
        Args:
            symbol: The trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            DataFrame with the latest market data
        """
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            
            latest_data = {
                'timestamp': [datetime.now()],
                'symbol': [symbol],
                'open': [float(ticker['openPrice'])],
                'high': [float(ticker['highPrice'])],
                'low': [float(ticker['lowPrice'])],
                'close': [float(ticker['lastPrice'])],
                'volume': [float(ticker['volume'])]
            }
            
            df = pd.DataFrame(latest_data)
            df = df.set_index('timestamp')
            
            return df
        except Exception as e:
            print(f"Error fetching realtime data from Binance: {e}")
            return pd.DataFrame() 