from typing import List, Dict, Any
import pandas as pd
import numpy as np


class DataProcessor:
    """
    DataProcessor handles data cleaning and indicator calculations.
    """
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess market data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            return data
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Handle missing values
        df = df.fillna(method='ffill')  # Forward fill
        df = df.fillna(method='bfill')  # Backward fill remaining NaNs
        
        # Remove duplicate indices
        df = df[~df.index.duplicated(keep='last')]
        
        # Sort by index (timestamp)
        df = df.sort_index()
        
        return df
    
    def calculate_indicators(self, data: pd.DataFrame, indicators: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Calculate technical indicators on market data.
        
        Args:
            data: DataFrame with market data
            indicators: List of indicator configurations
            
        Returns:
            DataFrame with added indicator columns
        """
        if data.empty:
            return data
        
        df = data.copy()
        
        for indicator in indicators:
            indicator_type = indicator.get('type', '').lower()
            params = indicator.get('params', {})
            
            if indicator_type == 'sma':
                self._add_simple_moving_average(df, **params)
            elif indicator_type == 'ema':
                self._add_exponential_moving_average(df, **params)
            elif indicator_type == 'rsi':
                self._add_rsi(df, **params)
            elif indicator_type == 'macd':
                self._add_macd(df, **params)
            elif indicator_type == 'bollinger_bands':
                self._add_bollinger_bands(df, **params)
        
        return df
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data for machine learning.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Normalized DataFrame
        """
        if data.empty:
            return data
        
        df = data.copy()
        
        # Min-max normalization for each column
        for col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:  # Avoid division by zero
                df[col] = (df[col] - min_val) / (max_val - min_val)
        
        return df
    
    def _add_simple_moving_average(self, df: pd.DataFrame, window: int = 20, column: str = 'close') -> None:
        """Add Simple Moving Average to the DataFrame."""
        df[f'sma_{window}'] = df[column].rolling(window=window).mean()
    
    def _add_exponential_moving_average(self, df: pd.DataFrame, window: int = 20, column: str = 'close') -> None:
        """Add Exponential Moving Average to the DataFrame."""
        df[f'ema_{window}'] = df[column].ewm(span=window, adjust=False).mean()
    
    def _add_rsi(self, df: pd.DataFrame, window: int = 14, column: str = 'close') -> None:
        """Add Relative Strength Index to the DataFrame."""
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    def _add_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
                signal_period: int = 9, column: str = 'close') -> None:
        """Add MACD to the DataFrame."""
        fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()
        
        df['macd_line'] = fast_ema - slow_ema
        df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
    
    def _add_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: float = 2.0, 
                           column: str = 'close') -> None:
        """Add Bollinger Bands to the DataFrame."""
        df[f'bb_middle_{window}'] = df[column].rolling(window=window).mean()
        df[f'bb_std_{window}'] = df[column].rolling(window=window).std()
        
        df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + (df[f'bb_std_{window}'] * num_std)
        df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - (df[f'bb_std_{window}'] * num_std) 