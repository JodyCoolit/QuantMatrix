from datetime import datetime, timedelta
import time
import logging
import random
import os
import json
import pickle
import pandas as pd
import yfinance as yf

from .data_source import DataSource


class YahooFinanceSource(DataSource):
    """Yahoo Finance implementation of DataSource with local caching."""
    
    def __init__(self, max_retries=3, initial_backoff=2, max_backoff=60, cache_dir=None, cache_expiry_days=1, offline_mode=False):
        """
        Initialize Yahoo Finance data source with retry parameters and caching.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            cache_dir: Directory for caching data, defaults to ./.yahoofinance_cache
            cache_expiry_days: Number of days after which cache expires
            offline_mode: If True, will only use cached data and not make API calls
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.offline_mode = offline_mode
        self.cache_expiry_days = cache_expiry_days
        
        # 设置缓存目录
        if cache_dir is None:
            # 默认在当前工作目录下创建缓存文件夹
            self.cache_dir = os.path.join(os.getcwd(), '.yahoofinance_cache')
        else:
            self.cache_dir = cache_dir
            
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Yahoo Finance data source initialized with cache at {self.cache_dir}")
        if self.offline_mode:
            self.logger.warning("Running in OFFLINE MODE - only cached data will be used")
    
    def _get_cache_key(self, symbol, timeframe, start_date, end_date=None):
        """
        Generate a cache key for the given parameters.
        
        Args:
            symbol: The ticker symbol
            timeframe: The data interval
            start_date: Start date for data
            end_date: End date for data (optional)
            
        Returns:
            A string representing the cache key
        """
        if end_date is None:
            end_str = "current"
        else:
            end_str = end_date.strftime('%Y%m%d')
            
        return f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_str}"
    
    def _get_cache_path(self, cache_key):
        """Get the file path for a cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _save_to_cache(self, data, cache_key):
        """
        Save data to cache.
        
        Args:
            data: Data to cache
            cache_key: Cache key
        """
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'timestamp': datetime.now(),
                    'data': data
                }, f)
            self.logger.debug(f"Data saved to cache: {cache_key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save data to cache: {e}")
            return False
    
    def _load_from_cache(self, cache_key):
        """
        Load data from cache if available and not expired.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data or None if not available or expired
        """
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                
            # Check if cache is expired
            cache_age = datetime.now() - cached['timestamp']
            if cache_age.days > self.cache_expiry_days:
                self.logger.debug(f"Cache expired for {cache_key}")
                return None
                
            self.logger.info(f"Using cached data for {cache_key}")
            return cached['data']
            
        except Exception as e:
            self.logger.error(f"Failed to load data from cache: {e}")
            return None
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Retry a function call with exponential backoff.
        
        Args:
            func: Function to call
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            The result of the function call, or None if all retries failed
        """
        # 如果处于离线模式，不进行 API 调用
        if self.offline_mode:
            self.logger.warning("Skipping API call in offline mode")
            return None
            
        retry_count = 0
        backoff = self.initial_backoff
        
        while retry_count <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                
                # 检查是否是API限制相关错误
                if retry_count < self.max_retries and ("rate limit" in error_msg or "too many requests" in error_msg):
                    retry_count += 1
                    
                    # 添加随机抖动以避免雷鸣效应
                    jitter = random.uniform(0, 0.5 * backoff)
                    wait_time = backoff + jitter
                    
                    self.logger.warning(f"Yahoo Finance rate limit hit. Retrying in {wait_time:.2f} seconds. (Attempt {retry_count}/{self.max_retries})")
                    time.sleep(wait_time)
                    
                    # 指数退避
                    backoff = min(backoff * 2, self.max_backoff)
                else:
                    # 非API限制错误或达到最大重试次数
                    self.logger.error(f"Error fetching data from Yahoo Finance: {e}")
                    return None
        
        self.logger.error(f"Failed to fetch data from Yahoo Finance after {self.max_retries} retries")
        return None
    
    def fetch_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance with caching.
        
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
        
        # 生成缓存键并尝试从缓存加载
        cache_key = self._get_cache_key(symbol, yf_interval, start_date, end_date)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        # 处于离线模式但没有缓存数据，生成模拟数据
        if self.offline_mode:
            self.logger.warning(f"No cached data available for {symbol} in offline mode. Generating mock data.")
            return self._generate_mock_data(symbol, start_date, end_date)
        
        # 尝试从 API 获取数据
        def _download_data():
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=yf_interval,
                progress=False
            )
            
            if data.empty:
                self.logger.warning(f"No data returned for {symbol} from {start_date} to {end_date}")
                return pd.DataFrame()
            
            # 标准化列名
            data.columns = [col.lower() for col in data.columns]
            data = data.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'adj close': 'adjusted_close',
                'volume': 'volume'
            })
            
            # 保存到缓存
            self._save_to_cache(data, cache_key)
            
            return data
        
        result = self._retry_with_backoff(_download_data)
        return result if result is not None else pd.DataFrame()
    
    def _generate_mock_data(self, symbol, start_date, end_date):
        """
        生成测试用的模拟股票数据。
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        # 创建日期范围
        days = (end_date - start_date).days + 1
        date_range = pd.date_range(start=start_date, periods=days, freq='D')
        
        # 使用股票代码作为随机种子，确保相同股票生成相同数据
        random.seed(hash(symbol) % 100000)
        
        # 生成起始价格，不同股票不同范围
        if symbol.upper() in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
            base_price = random.uniform(100, 200)
        else:
            base_price = random.uniform(50, 500)
        
        # 生成价格序列
        prices = []
        current_price = base_price
        volatility = base_price * 0.01  # 1% 波动率
        
        for _ in range(len(date_range)):
            # 生成当日价格
            open_price = current_price
            high_price = open_price * (1 + random.uniform(0, 0.02))
            low_price = open_price * (1 - random.uniform(0, 0.02))
            close_price = random.uniform(low_price, high_price)
            
            # 成交量
            volume = int(random.uniform(1000000, 10000000))
            
            prices.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'adjusted_close': close_price,
                'volume': volume
            })
            
            # 更新下一天的开盘价
            current_price = close_price * (1 + random.uniform(-0.03, 0.03))
        
        # 创建 DataFrame
        df = pd.DataFrame(prices, index=date_range)
        
        self.logger.info(f"Generated mock data for {symbol} from {start_date} to {end_date}")
        return df
    
    def get_realtime_data(self, symbol: str) -> pd.DataFrame:
        """
        Get the latest data for a symbol from Yahoo Finance.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            DataFrame with the latest market data
        """
        # 尝试从缓存加载，使用简单键名
        cache_key = f"{symbol}_realtime"
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        # 处于离线模式但没有缓存数据，生成模拟数据
        if self.offline_mode:
            self.logger.warning(f"No cached realtime data available for {symbol} in offline mode. Generating mock data.")
            mock_data = self._generate_mock_data(symbol, datetime.now() - timedelta(days=1), datetime.now())
            if not mock_data.empty:
                return mock_data.iloc[-1:].copy()
            return pd.DataFrame()
        
        def _get_latest_data():
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            
            if data.empty:
                self.logger.warning(f"No realtime data returned for {symbol}")
                return pd.DataFrame()
            
            # 获取最新数据点
            latest = data.iloc[-1:].copy()
            
            # 标准化列名
            latest.columns = [col.lower() for col in latest.columns]
            latest = latest.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'adj close': 'adjusted_close',
                'volume': 'volume'
            })
            
            # 保存到缓存
            self._save_to_cache(latest, cache_key)
            
            return latest
        
        result = self._retry_with_backoff(_get_latest_data)
        return result if result is not None else pd.DataFrame() 