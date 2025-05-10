from datetime import datetime
from typing import Dict, List, Union, Optional
import os
import json

import pandas as pd
import sqlite3
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, DateTime, MetaData


class TimeSeriesDatabase:
    """
    Database for storing and retrieving time series market data.
    """
    
    def __init__(self, db_path: str = 'market_data.db'):
        """
        Initialize the time series database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.metadata = MetaData()
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
            # Check if the symbols table exists
            with self.engine.connect() as conn:
                result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='symbols'")
                if not result.fetchone():
                    # Create the symbols table
                    symbols_table = Table(
                        'symbols', self.metadata,
                        Column('id', Integer, primary_key=True),
                        Column('symbol', String, unique=True),
                        Column('source', String),
                        Column('first_date', DateTime),
                        Column('last_date', DateTime)
                    )
                    symbols_table.create(self.engine)
            
            # Check if the market_data table exists
            with self.engine.connect() as conn:
                result = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data'")
                if not result.fetchone():
                    # Create the market_data table
                    market_data_table = Table(
                        'market_data', self.metadata,
                        Column('id', Integer, primary_key=True),
                        Column('symbol', String),
                        Column('timestamp', DateTime),
                        Column('open', Float),
                        Column('high', Float),
                        Column('low', Float),
                        Column('close', Float),
                        Column('volume', Float),
                        Column('adjusted_close', Float, nullable=True)
                    )
                    market_data_table.create(self.engine)
                    
                    # Create index for faster queries
                    conn.execute("CREATE INDEX idx_symbol_timestamp ON market_data (symbol, timestamp)")
        
        except Exception as e:
            print(f"Error creating database tables: {e}")
    
    def store_data(self, symbol: str, data: pd.DataFrame, source: str = 'unknown') -> None:
        """
        Store market data for a symbol.
        
        Args:
            symbol: The ticker symbol or asset identifier
            data: DataFrame with market data (indexed by timestamp)
            source: The source of the data
        """
        if data.empty:
            return
        
        try:
            # Make sure the DataFrame has the right columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Prepare data for storage
            df = data.copy()
            
            # Reset index to get timestamp as a column
            if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            
            # Ensure the DataFrame has a timestamp column
            if 'timestamp' not in df.columns:
                raise ValueError("DataFrame must have a timestamp column or index")
            
            # Add symbol column if not present
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            
            # Make sure adjusted_close column exists (can be None)
            if 'adjusted_close' not in df.columns:
                df['adjusted_close'] = None
            
            # Store in the database
            df = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close']]
            df.to_sql('market_data', self.engine, if_exists='append', index=False)
            
            # Update symbol information
            first_date = df['timestamp'].min()
            last_date = df['timestamp'].max()
            
            # Check if symbol exists
            with self.engine.connect() as conn:
                result = conn.execute(f"SELECT id FROM symbols WHERE symbol = '{symbol}'")
                row = result.fetchone()
                
                if row:
                    # Update existing symbol
                    conn.execute(
                        f"UPDATE symbols SET last_date = '{last_date}', source = '{source}' WHERE symbol = '{symbol}'"
                    )
                else:
                    # Insert new symbol
                    conn.execute(
                        f"INSERT INTO symbols (symbol, source, first_date, last_date) VALUES ('{symbol}', '{source}', '{first_date}', '{last_date}')"
                    )
        
        except Exception as e:
            print(f"Error storing data for {symbol}: {e}")
    
    def get_data(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get market data for a symbol within a date range.
        
        Args:
            symbol: The ticker symbol or asset identifier
            start_date: Start date for data retrieval (None for earliest available)
            end_date: End date for data retrieval (None for latest available)
            
        Returns:
            DataFrame with market data
        """
        try:
            query = f"SELECT * FROM market_data WHERE symbol = '{symbol}'"
            
            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            
            if end_date:
                query += f" AND timestamp <= '{end_date}'"
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql(query, self.engine)
            
            if df.empty:
                return pd.DataFrame()
            
            # Set timestamp as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Drop the id column
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            
            return df
        
        except Exception as e:
            print(f"Error retrieving data for {symbol}: {e}")
            return pd.DataFrame()
    
    def update_data(self, symbol: str, new_data: pd.DataFrame, source: str = 'unknown') -> None:
        """
        Update market data for a symbol with new data.
        
        Args:
            symbol: The ticker symbol or asset identifier
            new_data: DataFrame with new market data
            source: The source of the data
        """
        if new_data.empty:
            return
        
        try:
            # Get existing data to avoid duplicates
            existing_data = self.get_data(symbol)
            
            if existing_data.empty:
                # No existing data, just store the new data
                self.store_data(symbol, new_data, source)
                return
            
            # Find the latest timestamp in the existing data
            latest_timestamp = existing_data.index.max()
            
            # Filter new data to only include rows after the latest timestamp
            if isinstance(new_data.index, pd.DatetimeIndex):
                new_rows = new_data[new_data.index > latest_timestamp]
            else:
                # If the timestamp is a column
                if 'timestamp' in new_data.columns:
                    new_rows = new_data[new_data['timestamp'] > latest_timestamp]
                else:
                    # Can't determine what data is new, store all
                    new_rows = new_data
            
            if not new_rows.empty:
                self.store_data(symbol, new_rows, source)
        
        except Exception as e:
            print(f"Error updating data for {symbol}: {e}")
    
    def get_available_symbols(self) -> List[Dict[str, Union[str, datetime]]]:
        """
        Get a list of all available symbols in the database.
        
        Returns:
            List of dictionaries containing symbol information
        """
        try:
            query = "SELECT symbol, source, first_date, last_date FROM symbols ORDER BY symbol"
            df = pd.read_sql(query, self.engine)
            
            if df.empty:
                return []
            
            # Convert to list of dictionaries
            symbols = []
            for _, row in df.iterrows():
                symbols.append({
                    'symbol': row['symbol'],
                    'source': row['source'],
                    'first_date': row['first_date'],
                    'last_date': row['last_date']
                })
            
            return symbols
        
        except Exception as e:
            print(f"Error retrieving available symbols: {e}")
            return [] 