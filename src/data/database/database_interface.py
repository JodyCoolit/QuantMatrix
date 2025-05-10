"""
Database interface for time series market data storage.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Union, Optional, Any

import pandas as pd


class TimeSeriesDatabaseInterface(ABC):
    """
    Abstract base class for time series database implementations.
    All concrete database implementations should inherit from this class.
    """
    
    @abstractmethod
    def connect(self) -> None:
        """
        Connect to the database.
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """
        Disconnect from the database.
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the database connection is active.
        
        Returns:
            bool: True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    def store_data(self, symbol: str, data: pd.DataFrame, source: str = 'unknown') -> bool:
        """
        Store market data for a symbol.
        
        Args:
            symbol: The ticker symbol or asset identifier
            data: DataFrame with market data (indexed by timestamp)
            source: The source of the data
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_data(self, symbol: str, start_date: Optional[datetime] = None, 
                 end_date: Optional[datetime] = None, fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get market data for a symbol within a date range.
        
        Args:
            symbol: The ticker symbol or asset identifier
            start_date: Start date for data retrieval (None for earliest available)
            end_date: End date for data retrieval (None for latest available)
            fields: List of fields to retrieve (None for all available)
            
        Returns:
            DataFrame with market data
        """
        pass
    
    @abstractmethod
    def update_data(self, symbol: str, new_data: pd.DataFrame, source: str = 'unknown') -> bool:
        """
        Update market data for a symbol with new data.
        
        Args:
            symbol: The ticker symbol or asset identifier
            new_data: DataFrame with new market data
            source: The source of the data
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_data(self, symbol: str, start_date: Optional[datetime] = None, 
                    end_date: Optional[datetime] = None) -> bool:
        """
        Delete market data for a symbol within a date range.
        
        Args:
            symbol: The ticker symbol or asset identifier
            start_date: Start date for data deletion (None for earliest available)
            end_date: End date for data deletion (None for latest available)
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[Dict[str, Union[str, datetime]]]:
        """
        Get a list of all available symbols in the database.
        
        Returns:
            List of dictionaries containing symbol information
        """
        pass
    
    @abstractmethod
    def get_first_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        Get the timestamp of the first available data point for a symbol.
        
        Args:
            symbol: The ticker symbol or asset identifier
            
        Returns:
            The earliest timestamp, or None if no data is available
        """
        pass
    
    @abstractmethod
    def get_last_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        Get the timestamp of the last available data point for a symbol.
        
        Args:
            symbol: The ticker symbol or asset identifier
            
        Returns:
            The latest timestamp, or None if no data is available
        """
        pass 