"""
Test module for data sources in the QuantMatrix project.
This module provides functions to test various data sources including Yahoo Finance, Binance, etc.
"""

from datetime import datetime, timedelta
import logging
import time
import pandas as pd
from src.data.data_manager import DataManager
from src.data.data_processor import DataProcessor
from src.data.yahoo_finance_source import YahooFinanceSource
from src.data.binance_source import BinanceSource


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("data_source_tester")


class DataSourceTester:
    """Class for testing different data sources."""
    
    def __init__(self):
        """Initialize the tester with a data manager."""
        self.data_processor = DataProcessor()
        self.data_manager = DataManager(self.data_processor)
        
        # Initialize date range for testing (last 30 days)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30)
        
        # Track which sources have been initialized
        self.initialized_sources = set()
    
    def init_sources(self, sources=None, binance_api_key="", binance_api_secret="", offline_mode=False):
        """
        Initialize all data sources.
        
        Args:
            sources (list): List of sources to initialize. If None, all sources are initialized.
            binance_api_key (str): Binance API key.
            binance_api_secret (str): Binance API secret.
            offline_mode (bool): If True, use offline mode to prevent API calls.
        """
        # Default to all sources if none specified
        if sources is None:
            sources = ["yahoo", "binance"]
        
        # Add Yahoo Finance data source
        if "yahoo" in sources and "yahoo" not in self.initialized_sources:
            try:
                # Create Yahoo Finance source with retry capabilities
                yahoo_source = YahooFinanceSource(
                    max_retries=3, 
                    initial_backoff=5, 
                    max_backoff=60,
                    offline_mode=offline_mode,
                    cache_expiry_days=7  # Cache for a week
                )
                self.data_manager.add_data_source("yahoo", yahoo_source)
                self.initialized_sources.add("yahoo")
                logger.info(f"Initialized Yahoo Finance data source (offline_mode={offline_mode})")
            except Exception as e:
                logger.error(f"Error initializing Yahoo Finance: {e}")
        
        # Add Binance data source
        if "binance" in sources and "binance" not in self.initialized_sources:
            try:
                self.data_manager.add_data_source("binance", BinanceSource(
                    api_key=binance_api_key, 
                    api_secret=binance_api_secret
                ))
                self.initialized_sources.add("binance")
                logger.info("Initialized Binance data source")
            except Exception as e:
                logger.error(f"Error initializing Binance: {e}")
    
    def test_yahoo_finance(self, symbol="AAPL", timeframe="1d"):
        """
        Test Yahoo Finance data source.
        
        Args:
            symbol (str): Stock symbol to test
            timeframe (str): Timeframe for data retrieval
            
        Returns:
            dict: Test results containing historical and realtime data
        """
        # Initialize the Yahoo Finance source if not already done
        if "yahoo" not in self.initialized_sources:
            self.init_sources(sources=["yahoo"])
        
        # Check if initialization was successful
        if "yahoo" not in self.initialized_sources:
            logger.error("Yahoo Finance source is not initialized, skipping test")
            return None
        
        logger.info(f"=== Testing Yahoo Finance Data Source for {symbol} ===")
        results = {"historical": None, "realtime": None}
        
        try:
            # Fetch historical data
            logger.info(f"Fetching historical data for {symbol} from {self.start_date} to {self.end_date}")
            historical_data = self.data_manager.get_historical_data(
                source="yahoo",
                symbol=symbol,
                timeframe=timeframe,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if historical_data is not None and not historical_data.empty:
                results["historical"] = historical_data
                logger.info(f"Retrieved {len(historical_data)} rows of {symbol} data from Yahoo Finance")
                logger.info(f"Sample data:\n{historical_data.head()}")
            else:
                logger.warning(f"No historical data retrieved for {symbol}")
            
            # Get real-time data (only if historical data was successful)
            if results["historical"] is not None and not results["historical"].empty:
                logger.info(f"Fetching real-time data for {symbol}")
                try:
                    self.data_manager.fetch_realtime_update("yahoo", symbol)
                    realtime_data = self.data_manager.sources["yahoo"].get_realtime_data(symbol)
                    
                    if realtime_data is not None and not realtime_data.empty:
                        results["realtime"] = realtime_data
                        logger.info(f"Real-time {symbol} data:\n{realtime_data}")
                    else:
                        logger.warning(f"No real-time data retrieved for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error fetching real-time data: {e}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error with Yahoo Finance: {e}")
            return None
    
    def test_binance(self, symbol="BTCUSDT", timeframe="1d"):
        """
        Test Binance data source.
        
        Args:
            symbol (str): Crypto symbol to test
            timeframe (str): Timeframe for data retrieval
            
        Returns:
            dict: Test results containing historical and realtime data
        """
        # Initialize the Binance source if not already done
        if "binance" not in self.initialized_sources:
            self.init_sources(sources=["binance"])
        
        # Check if initialization was successful
        if "binance" not in self.initialized_sources:
            logger.error("Binance source is not initialized, skipping test")
            return None
        
        logger.info(f"=== Testing Binance Data Source for {symbol} ===")
        results = {"historical": None, "realtime": None}
        
        try:
            # Fetch historical data
            logger.info(f"Fetching historical data for {symbol} from {self.start_date} to {self.end_date}")
            historical_data = self.data_manager.get_historical_data(
                source="binance",
                symbol=symbol,
                timeframe=timeframe,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            if historical_data is not None and not historical_data.empty:
                results["historical"] = historical_data
                logger.info(f"Retrieved {len(historical_data)} rows of {symbol} data from Binance")
                logger.info(f"Sample data:\n{historical_data.head()}")
            else:
                logger.warning(f"No historical data retrieved for {symbol}")
            
            # Get real-time data
            logger.info(f"Fetching real-time data for {symbol}")
            try:
                self.data_manager.fetch_realtime_update("binance", symbol)
                realtime_data = self.data_manager.sources["binance"].get_realtime_data(symbol)
                
                if realtime_data is not None and not realtime_data.empty:
                    results["realtime"] = realtime_data
                    logger.info(f"Real-time {symbol} data:\n{realtime_data}")
                else:
                    logger.warning(f"No real-time data retrieved for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching real-time data: {e}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error with Binance: {e}")
            return None
    
    def test_with_indicators(self, source="yahoo", symbol="AAPL", timeframe="1d"):
        """
        Test data retrieval with technical indicators.
        
        Args:
            source (str): Data source to use
            symbol (str): Symbol to test
            timeframe (str): Timeframe for data retrieval
            
        Returns:
            pd.DataFrame: Data with indicators
        """
        # Initialize the requested source if not already done
        if source not in self.initialized_sources:
            self.init_sources(sources=[source])
        
        # Check if initialization was successful
        if source not in self.initialized_sources:
            logger.error(f"{source.capitalize()} source is not initialized, skipping test")
            return None
        
        logger.info(f"=== Testing {source.capitalize()} with Technical Indicators for {symbol} ===")
        
        try:
            indicators = [
                {"type": "SMA", "params": {"window": 20}},
                {"type": "RSI", "params": {"window": 14}}
            ]
            
            logger.info(f"Fetching data with indicators for {symbol}")
            data_with_indicators = self.data_manager.get_historical_data(
                source=source,
                symbol=symbol,
                timeframe=timeframe,
                start_date=self.start_date,
                end_date=self.end_date,
                apply_indicators=True,
                indicators=indicators
            )
            
            if data_with_indicators is not None and not data_with_indicators.empty:
                logger.info(f"{symbol} data with indicators:\n{data_with_indicators.tail()}")
                return data_with_indicators
            else:
                logger.warning(f"No data with indicators retrieved for {symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error with indicators: {e}")
            return None


def test_with_retry(test_func, *args, max_attempts=3, retry_interval=60, switch_to_offline=True, **kwargs):
    """
    Run a test function with retry if it fails. Switches to offline mode after first API rate limit failure.
    
    Args:
        test_func: Function to run
        *args: Arguments to pass to function
        max_attempts: Maximum number of attempts
        retry_interval: Seconds to wait between attempts
        switch_to_offline: If True, switches to offline mode after a rate limit error
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Result of the function
    """
    attempt = 1
    offline_activated = False
    
    while attempt <= max_attempts:
        logger.info(f"Attempt {attempt}/{max_attempts}")
        
        # 如果之前的尝试失败且开启了自动切换离线模式
        if attempt > 1 and switch_to_offline and not offline_activated:
            logger.warning("Switching to OFFLINE MODE to avoid API rate limits")
            
            # 重新初始化 DataSourceTester，设置为离线模式
            if isinstance(test_func.__self__, DataSourceTester):
                tester = test_func.__self__
                # 把原有的数据源清理掉
                for source_name in list(tester.initialized_sources):
                    if source_name in tester.data_manager.sources:
                        tester.data_manager.sources.pop(source_name)
                tester.initialized_sources.clear()
                # 用离线模式重新初始化
                tester.init_sources(offline_mode=True)
                offline_activated = True
        
        result = test_func(*args, **kwargs)
        
        # 检查结果是否有效
        empty_result = (
            result is None or 
            (isinstance(result, pd.DataFrame) and result.empty) or
            (isinstance(result, dict) and (result.get("historical") is None or result.get("historical").empty))
        )
        
        # 检查是否是因为 API 限制而失败
        api_limited = False
        if isinstance(test_func.__self__, DataSourceTester) and any(
            "rate limit" in log.lower() or "too many requests" in log.lower() 
            for log in logging.getLogger("src.data.yahoo_finance_source").handlers
        ):
            api_limited = True
        
        if empty_result:
            if attempt < max_attempts:
                if api_limited and switch_to_offline and not offline_activated:
                    logger.warning("API rate limit detected. Switching to offline mode for next attempt.")
                    continue  # 直接进入下一次尝试，会切换到离线模式
                else:
                    logger.warning(f"Test failed, retrying in {retry_interval} seconds...")
                    time.sleep(retry_interval)
            else:
                logger.error(f"Test failed after {max_attempts} attempts")
        else:
            logger.info(f"Test succeeded on attempt {attempt} {'(offline mode)' if offline_activated else ''}")
            return result
            
        attempt += 1
            
    return result  # Return the last result even if it failed


def main():
    """Main function to run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data source tests")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode (use cached data)")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol to test")
    args = parser.parse_args()
    
    tester = DataSourceTester()
    
    # Initialize data sources with offline mode if specified
    tester.init_sources(offline_mode=args.offline)
    
    # Test Yahoo Finance with retry
    yahoo_results = test_with_retry(
        tester.test_yahoo_finance,
        symbol=args.symbol, 
        timeframe="1d",
        max_attempts=3,
        retry_interval=120,
        switch_to_offline=True
    )
    
    # Test Binance
    binance_results = tester.test_binance(symbol="BTCUSDT", timeframe="1d")
    
    # Test with indicators (using whichever source worked)
    if yahoo_results and yahoo_results.get("historical") is not None and not yahoo_results.get("historical").empty:
        source_for_indicators = "yahoo"
        symbol_for_indicators = args.symbol
    elif binance_results and binance_results.get("historical") is not None and not binance_results.get("historical").empty:
        source_for_indicators = "binance"
        symbol_for_indicators = "BTCUSDT"
    else:
        logger.warning("Skipping indicator tests as no data sources provided successful results")
        source_for_indicators = None
        symbol_for_indicators = None
    
    if source_for_indicators:
        logger.info(f"Testing indicators with {source_for_indicators} data for {symbol_for_indicators}")
        indicators_result = test_with_retry(
            tester.test_with_indicators,
            source=source_for_indicators,
            symbol=symbol_for_indicators,
            timeframe="1d",
            max_attempts=2,
            retry_interval=60,
            switch_to_offline=True
        )
    
    logger.info("=== All tests completed ===")


if __name__ == "__main__":
    main()