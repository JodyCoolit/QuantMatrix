"""
InfluxDB implementation of the time series database interface.
Only supports InfluxDB 3.x.
"""
from datetime import datetime
import os
import platform
from typing import Dict, List, Union, Optional
import logging
import pandas as pd
from influxdb_client_3 import InfluxDBClient3, flight_client_options

from .database_interface import TimeSeriesDatabaseInterface


class InfluxDBRepository(TimeSeriesDatabaseInterface):
    """
    InfluxDB implementation of the time series database interface.
    """
    
    def __init__(self, url: str, token: str, bucket: str, 
                 org: str = '', measurement: str = 'market_data'):
        """
        Initialize the InfluxDB repository.
        
        Args:
            url: InfluxDB server URL
            token: Authentication token
            bucket: Database name
            org: Not used in InfluxDB 3.x, kept for compatibility
            measurement: Measurement name for market data
        """
        self.url = url
        self.token = token
        self.bucket = bucket
        self.measurement = measurement
        self.client = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> None:
        """
        Connect to the InfluxDB server.
        """
        try:
            # Parse URL to get host, protocol, and port
            url = self.url.lower()
            protocol = "https://" if url.startswith("https://") else "http://"
            
            # Remove protocol
            host_part = url.replace("https://", "").replace("http://", "")
            
            # Split host and port
            if ":" in host_part:
                host, port_str = host_part.split(":", 1)
                # Handle path in URL
                if "/" in port_str:
                    port_str = port_str.split("/", 1)[0]
                try:
                    port = int(port_str)
                except ValueError:
                    self.logger.warning(f"Invalid port in URL {self.url}, using default port")
                    port = 8086
            else:
                host = host_part
                port = 8086  # Default port
                
            if "/" in host:
                host = host.split("/", 1)[0]
                
            self.logger.debug(f"Connecting to InfluxDB at host={host}")
            
            # If localhost is specified, try using IP address directly
            if host.lower() in ['localhost', '127.0.0.1']:
                alt_host = '127.0.0.1'
                self.logger.debug(f"Using alternative host: {alt_host}")
            else:
                alt_host = host
            
            # 处理Windows上的证书问题
            fco = None
            if platform.system() == 'Windows':
                try:
                    import certifi
                    self.logger.debug("Using certifi for Windows certificate handling")
                    with open(certifi.where(), "r") as f:
                        cert = f.read()
                    fco = flight_client_options(tls_root_certs=cert)
                except ImportError:
                    self.logger.warning("certifi package not found. Windows users may experience certificate issues.")
            
            # 创建客户端
            try:
                if fco:
                    self.client = InfluxDBClient3(
                        host=alt_host,
                        token=self.token,
                        database=self.bucket,
                        flight_client_options=fco
                    )
                else:
                    self.client = InfluxDBClient3(
                        host=alt_host,
                        token=self.token,
                        database=self.bucket
                    )
                self.logger.info(f"Connected to InfluxDB at {self.url}")
            except Exception as conn_error:
                self.logger.error(f"Could not connect to {alt_host}: {conn_error}")
                # 如果连接失败且使用的是替代主机，尝试原始主机
                if alt_host != host:
                    self.logger.debug(f"Trying original host: {host}")
                    try:
                        if fco:
                            self.client = InfluxDBClient3(
                                host=host,
                                token=self.token,
                                database=self.bucket,
                                flight_client_options=fco
                            )
                        else:
                            self.client = InfluxDBClient3(
                                host=host,
                                token=self.token,
                                database=self.bucket
                            )
                        self.logger.info(f"Connected to InfluxDB at {self.url}")
                    except Exception as orig_host_error:
                        self.logger.error(f"Could not connect to {host}: {orig_host_error}")
                        raise
                else:
                    raise
        except Exception as e:
            self.logger.error(f"Failed to connect to InfluxDB: {e}")
            raise
    
    def disconnect(self) -> None:
        """
        Disconnect from the InfluxDB server.
        """
        if self.client:
            self.client.close()
        self.client = None
        self.logger.info("Disconnected from InfluxDB")
    
    def is_connected(self) -> bool:
        """
        Check if the database connection is active.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.client is not None
    
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
        if not self.is_connected():
            self.logger.error("Not connected to InfluxDB")
            return False
        
        if data.empty:
            self.logger.warning(f"Empty data for {symbol}, nothing to store")
            return True
        
        try:
            # 准备数据
            # 检查数据格式，确保有时间列
            df = data.copy()
            
            # 如果数据帧有DatetimeIndex，重置索引并获取时间戳列
            if isinstance(df.index, pd.DatetimeIndex):
                self.logger.debug(f"DataFrame has DatetimeIndex: {df.index.name}")
                # 如果索引没有名称，将其设置为'timestamp'
                if df.index.name is None:
                    df.index.name = 'timestamp'
                df = df.reset_index()
                timestamp_col = df.index.name or 'timestamp'
            else:
                # 查找名为'timestamp'的列
                if 'timestamp' in df.columns:
                    timestamp_col = 'timestamp'
                # 如果没有timestamp列，尝试查找日期类型的列
                else:
                    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
                    if date_cols:
                        timestamp_col = date_cols[0]
                        self.logger.debug(f"Using {timestamp_col} as timestamp column")
                    else:
                        self.logger.error(f"No timestamp column found in data")
                        self.logger.debug(f"Available columns: {df.columns.tolist()}")
                        return False
            
            # 确保timestamp_col在列中
            if timestamp_col not in df.columns:
                self.logger.error(f"Timestamp column '{timestamp_col}' not found in data")
                self.logger.debug(f"Available columns: {df.columns.tolist()}")
                return False
            
            # 确保时间戳列是datetime类型
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # 添加标签列
            df['symbol'] = symbol
            df['source'] = source
            
            # 写入数据
            self.client.write(
                record=df,
                data_frame_measurement_name=self.measurement,
                data_frame_tag_columns=['symbol', 'source'],
                data_frame_timestamp_column=timestamp_col
            )
            
            self.logger.info(f"Stored {len(df)} data points for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing data for {symbol}: {e}")
            return False
    
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
        if not self.is_connected():
            self.logger.error("Not connected to InfluxDB")
            return pd.DataFrame()
        
        try:
            # 构建查询
            query = f"SELECT "
            
            if fields and len(fields) > 0:
                query += ", ".join(fields)
            else:
                query += "*"
                
            query += f" FROM {self.measurement} WHERE symbol = '{symbol}'"
            
            if start_date:
                query += f" AND time >= '{start_date.isoformat()}'"
            if end_date:
                query += f" AND time <= '{end_date.isoformat()}'"
                
            query += " ORDER BY time"
            
            # 执行查询
            self.logger.debug(f"Executing query: {query}")
            result = self.client.query(
                query=query,
                language="influxql",
                mode="pandas"
            )
            
            if result.empty:
                return pd.DataFrame()
            
            # 设置索引
            if 'time' in result.columns:
                result = result.set_index('time')
                result.index.name = 'timestamp'
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving data for {symbol}: {e}")
            return pd.DataFrame()
    
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
        # InfluxDB 3.x会自动处理重复数据，我们可以直接存储
        return self.store_data(symbol, new_data, source)
    
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
        if not self.is_connected():
            self.logger.error("Not connected to InfluxDB")
            return False
        
        try:
            # 对于InfluxDB 3.x，需要使用SQL执行删除操作
            predicate = f"symbol = '{symbol}'"
            
            if start_date and end_date:
                delete_query = f"DELETE FROM {self.measurement} WHERE {predicate} AND time >= '{start_date.isoformat()}' AND time <= '{end_date.isoformat()}'"
            elif start_date:
                delete_query = f"DELETE FROM {self.measurement} WHERE {predicate} AND time >= '{start_date.isoformat()}'"
            elif end_date:
                delete_query = f"DELETE FROM {self.measurement} WHERE {predicate} AND time <= '{end_date.isoformat()}'"
            else:
                delete_query = f"DELETE FROM {self.measurement} WHERE {predicate}"
                
            # 执行删除
            self.logger.debug(f"Executing delete query: {delete_query}")
            self.client.query(query=delete_query, language="sql")
            
            self.logger.info(f"Deleted data for {symbol} from {start_date} to {end_date}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting data for {symbol}: {e}")
            return False
    
    def get_available_symbols(self) -> List[Dict[str, Union[str, datetime]]]:
        """
        Get a list of all available symbols in the database.
        
        Returns:
            List of dictionaries containing symbol information
        """
        if not self.is_connected():
            self.logger.error("Not connected to InfluxDB")
            return []
        
        try:
            # 查询所有唯一的symbol标签
            query = f"SELECT DISTINCT(symbol) FROM {self.measurement}"
            self.logger.debug(f"Executing query: {query}")
            symbols_query = self.client.query(query=query, language="influxql", mode="pandas")
            
            if symbols_query.empty:
                return []
            
            # 获取唯一的symbol值
            result = []
            
            for symbol in symbols_query['distinct']:
                if not symbol or pd.isna(symbol):
                    continue
                    
                first_ts = self.get_first_timestamp(symbol)
                last_ts = self.get_last_timestamp(symbol)
                
                result.append({
                    'symbol': symbol,
                    'first_date': first_ts,
                    'last_date': last_ts
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving available symbols: {e}")
            return []
    
    def get_first_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        Get the timestamp of the first available data point for a symbol.
        
        Args:
            symbol: The ticker symbol or asset identifier
            
        Returns:
            The earliest timestamp, or None if no data is available
        """
        if not self.is_connected():
            self.logger.error("Not connected to InfluxDB")
            return None
        
        try:
            query = f"SELECT time FROM {self.measurement} WHERE symbol = '{symbol}' ORDER BY time ASC LIMIT 1"
            self.logger.debug(f"Executing query: {query}")
            result = self.client.query(query=query, language="influxql", mode="pandas")
            
            if result.empty:
                return None
            
            return result['time'].iloc[0]
            
        except Exception as e:
            self.logger.error(f"Error retrieving first timestamp for {symbol}: {e}")
            return None
    
    def get_last_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        Get the timestamp of the last available data point for a symbol.
        
        Args:
            symbol: The ticker symbol or asset identifier
            
        Returns:
            The latest timestamp, or None if no data is available
        """
        if not self.is_connected():
            self.logger.error("Not connected to InfluxDB")
            return None
        
        try:
            query = f"SELECT time FROM {self.measurement} WHERE symbol = '{symbol}' ORDER BY time DESC LIMIT 1"
            self.logger.debug(f"Executing query: {query}")
            result = self.client.query(query=query, language="influxql", mode="pandas")
            
            if result.empty:
                return None
            
            return result['time'].iloc[0]
            
        except Exception as e:
            self.logger.error(f"Error retrieving last timestamp for {symbol}: {e}")
            return None 