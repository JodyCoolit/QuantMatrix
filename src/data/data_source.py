from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

import pandas as pd


class DataSource(ABC):
    """Abstract base class for all data sources."""

    @abstractmethod
    def fetch_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical data for a specific symbol.
        
        Args:
            symbol: The ticker symbol or asset identifier
            timeframe: The data interval (e.g., '1d', '1h', '5m')
            start_date: The beginning date for the data
            end_date: The ending date for the data
            
        Returns:
            DataFrame with historical market data
        """
        pass
    
    @abstractmethod
    def get_realtime_data(self, symbol: str) -> pd.DataFrame:
        """
        Get real-time data for a specific symbol.
        
        Args:
            symbol: The ticker symbol or asset identifier
            
        Returns:
            DataFrame with the latest market data
        """
        pass 