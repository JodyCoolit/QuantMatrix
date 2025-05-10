from datetime import datetime
from typing import Dict, Callable, List, Any

import pandas as pd

from .data_source import DataSource
from .data_processor import DataProcessor


class DataManager:
    """
    DataManager is responsible for managing different data sources
    and coordinating data fetching, processing and storage.
    """
    
    def __init__(self, data_processor: DataProcessor = None):
        """
        Initialize DataManager with an optional data processor.
        
        Args:
            data_processor: An optional DataProcessor instance for data processing
        """
        self.sources: Dict[str, DataSource] = {}
        self.data_processor = data_processor or DataProcessor()
        self.realtime_callbacks: Dict[str, List[Callable]] = {}
    
    def add_data_source(self, name: str, source: DataSource) -> None:
        """
        Add a data source to the manager.
        
        Args:
            name: A unique identifier for the data source
            source: A DataSource implementation
        """
        self.sources[name] = source
    
    def get_historical_data(self, source: str, symbol: str, timeframe: str, 
                          start_date: datetime, end_date: datetime,
                          apply_indicators: bool = False,
                          indicators: List[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Get historical data from a specific source.
        
        Args:
            source: The name of the data source to use
            symbol: The ticker symbol or asset identifier
            timeframe: The data interval (e.g., '1d', '1h', '5m')
            start_date: The beginning date for the data
            end_date: The ending date for the data
            apply_indicators: Whether to apply technical indicators
            indicators: List of indicators to apply
            
        Returns:
            DataFrame with historical market data
        """
        if source not in self.sources:
            raise ValueError(f"Data source '{source}' not found")
        
        # Fetch raw data from the source
        data = self.sources[source].fetch_data(symbol, timeframe, start_date, end_date)
        
        # Clean the data
        data = self.data_processor.clean_data(data)
        
        # Calculate indicators if requested
        if apply_indicators and indicators:
            data = self.data_processor.calculate_indicators(data, indicators)
        
        return data
    
    def subscribe_to_realtime(self, source: str, symbol: str, callback: Callable[[pd.DataFrame], None]) -> None:
        """
        Subscribe to real-time data updates for a symbol.
        
        Args:
            source: The name of the data source to use
            symbol: The ticker symbol or asset identifier
            callback: A function to call with new data
        """
        if source not in self.sources:
            raise ValueError(f"Data source '{source}' not found")
        
        # Create a key for this subscription
        key = f"{source}_{symbol}"
        
        # Add callback to the list
        if key not in self.realtime_callbacks:
            self.realtime_callbacks[key] = []
        
        self.realtime_callbacks[key].append(callback)
    
    def fetch_realtime_update(self, source: str, symbol: str) -> None:
        """
        Manually fetch a real-time update and trigger callbacks.
        
        Args:
            source: The name of the data source to use
            symbol: The ticker symbol or asset identifier
        """
        if source not in self.sources:
            raise ValueError(f"Data source '{source}' not found")
        
        # Create a key for this subscription
        key = f"{source}_{symbol}"
        
        # If there are no callbacks, there's no need to fetch data
        if key not in self.realtime_callbacks or not self.realtime_callbacks[key]:
            return
        
        # Fetch real-time data
        data = self.sources[source].get_realtime_data(symbol)
        
        # Skip if no data was received
        if data.empty:
            return
        
        # Process the data
        data = self.data_processor.clean_data(data)
        
        # Call all registered callbacks
        for callback in self.realtime_callbacks[key]:
            callback(data) 