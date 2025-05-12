"""
TimescaleDB implementation of the time series database interface.
"""
import logging
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from .database_interface import TimeSeriesDatabaseInterface


class TimescaleDBRepository(TimeSeriesDatabaseInterface):
    """
    TimescaleDB implementation of the time series database interface.
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        schema: str = "public",
        sslmode: str = "prefer",
        measurement: str = "market_data"
    ):
        """
        Initialize the TimescaleDB repository.
        
        Args:
            host: 数据库主机地址
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
            schema: 模式名称，默认为public
            sslmode: SSL模式，默认为prefer
            measurement: 默认表名，默认为market_data
        """
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.schema = schema
        self.sslmode = sslmode
        self.measurement = measurement
        self.conn = None
        self.cursor = None
        self.is_connected_flag = False
        
    def connect(self) -> bool:
        """
        Connect to the TimescaleDB server.
        
        Returns:
            连接是否成功
        """
        if self.is_connected_flag:
            return True
            
        try:
            self.logger.debug(f"正在连接TimescaleDB: host={self.host}, port={self.port}, database={self.database}")
            
            # 连接到数据库
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                sslmode=self.sslmode
            )
            
            # 创建游标
            self.cursor = self.conn.cursor()
            
            # 检查TimescaleDB扩展是否已安装
            self.cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb';")
            if not self.cursor.fetchone():
                self.logger.warning("TimescaleDB扩展尚未安装在此数据库上")
            
            self.is_connected_flag = True
            self.logger.info(f"成功连接到TimescaleDB: {self.host}:{self.port}/{self.database}")
            return True
            
        except Exception as e:
            self.logger.error(f"连接到TimescaleDB失败: {e}")
            self.is_connected_flag = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the TimescaleDB server.
        
        Returns:
            断开连接是否成功
        """
        if not self.is_connected_flag:
            return True
            
        try:
            if self.cursor:
                self.cursor.close()
                
            if self.conn:
                self.conn.close()
                
            self.is_connected_flag = False
            self.logger.info("已断开与TimescaleDB的连接")
            return True
            
        except Exception as e:
            self.logger.error(f"断开与TimescaleDB的连接失败: {e}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if the database connection is active.
        
        Returns:
            True if connected, False otherwise
        """
        if not self.is_connected_flag or not self.conn:
            return False
            
        try:
            # 尝试执行一个简单的查询来验证连接
            self.cursor.execute("SELECT 1")
            result = self.cursor.fetchone()
            return result is not None and result[0] == 1
        except Exception:
            self.is_connected_flag = False
            return False
    
    def create_hypertable(self, table_name: str, time_column: str = "time") -> bool:
        """
        创建超表或将现有表转换为超表。
        
        Args:
            table_name: 表名
            time_column: 时间列名，默认为"time"
            
        Returns:
            操作是否成功
        """
        if not self.is_connected_flag:
            self.logger.error("未连接到TimescaleDB")
            return False
            
        try:
            # 检查表是否存在
            self.cursor.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = %s AND table_name = %s);",
                (self.schema, table_name)
            )
            
            table_exists = self.cursor.fetchone()[0]
            
            if not table_exists:
                self.logger.error(f"表 {table_name} 不存在")
                return False
                
            # 检查表是否已经是超表
            self.cursor.execute(
                "SELECT EXISTS (SELECT FROM timescaledb_information.hypertables WHERE hypertable_name = %s);",
                (table_name,)
            )
            
            is_hypertable = self.cursor.fetchone()[0]
            
            if is_hypertable:
                self.logger.info(f"表 {table_name} 已经是超表")
                return True
                
            # 将表转换为超表 - 修复SQL语句中的标识符引用
            query = sql.SQL("SELECT create_hypertable({}, {});").format(
                sql.Identifier(table_name), 
                sql.Literal(time_column)
            )
            self.cursor.execute(query)
            
            self.conn.commit()
            self.logger.info(f"成功将表 {table_name} 转换为超表")
            return True
            
        except Exception as e:
            self.logger.error(f"创建超表失败: {e}")
            self.conn.rollback()
            return False
    
    def ensure_table_exists(self, table_name: Optional[str] = None, schema: Optional[Dict] = None) -> bool:
        """
        确保表存在，如果不存在则创建。
        
        Args:
            table_name: 表名，默认使用self.measurement
            schema: 表结构定义，键为列名，值为数据类型
            
        Returns:
            操作是否成功
        """
        if not self.is_connected_flag:
            self.logger.error("未连接到TimescaleDB")
            return False
            
        if table_name is None:
            table_name = self.measurement
            
        if schema is None:
            # 默认schema，适用于市场数据
            schema = {
                "time": "TIMESTAMP NOT NULL",
                "symbol": "TEXT NOT NULL",
                "open": "DOUBLE PRECISION",
                "high": "DOUBLE PRECISION",
                "low": "DOUBLE PRECISION",
                "close": "DOUBLE PRECISION",
                "volume": "DOUBLE PRECISION",
                "source": "TEXT"
            }
            
        try:
            # 检查表是否存在
            self.cursor.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = %s AND table_name = %s);",
                (self.schema, table_name)
            )
            
            table_exists = self.cursor.fetchone()[0]
            
            if table_exists:
                self.logger.info(f"表 {table_name} 已存在")
                return True
                
            # 创建表
            columns_sql = []
            for col_name, col_type in schema.items():
                columns_sql.append(sql.SQL("{} {}").format(
                    sql.Identifier(col_name), 
                    sql.SQL(col_type)
                ))
                
            query = sql.SQL("CREATE TABLE {} ({})").format(
                sql.Identifier(table_name),
                sql.SQL(", ").join(columns_sql)
            )
            
            self.cursor.execute(query)
            
            # 添加主键
            if "time" in schema and "symbol" in schema:
                self.cursor.execute(
                    sql.SQL("ALTER TABLE {} ADD PRIMARY KEY (time, symbol);").format(
                        sql.Identifier(table_name)
                    )
                )
            
            self.conn.commit()
            self.logger.info(f"成功创建表 {table_name}")
            
            # 将表转换为超表
            return self.create_hypertable(table_name)
            
        except Exception as e:
            self.logger.error(f"确保表存在失败: {e}")
            self.conn.rollback()
            return False
    
    def store(self, data: pd.DataFrame, table_name: Optional[str] = None) -> bool:
        """
        Store data into TimescaleDB.
        
        Args:
            data: 要存储的数据，DataFrame格式
            table_name: 表名，默认使用self.measurement
            
        Returns:
            存储是否成功
        """
        if not self.is_connected_flag:
            self.logger.error("未连接到TimescaleDB")
            return False
            
        if table_name is None:
            table_name = self.measurement
            
        try:
            # 确保表存在
            if not self.ensure_table_exists(table_name):
                return False
                
            # 准备列名
            columns = list(data.columns)
            
            # 准备数据
            values = [tuple(row) for row in data.values]
            
            # 使用execute_values批量插入数据
            execute_values(
                self.cursor,
                sql.SQL("INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING").format(
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(map(sql.Identifier, columns))
                ).as_string(self.conn),
                values
            )
            
            self.conn.commit()
            self.logger.info(f"成功存储 {len(values)} 条数据到表 {table_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"存储数据失败: {e}")
            self.conn.rollback()
            return False
    
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
        if not self.is_connected_flag:
            if not self.connect():
                return False
                
        try:
            # 确保数据帧有正确的列
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in data.columns:
                    self.logger.error(f"数据缺少必要的列: {col}")
                    return False
            
            # 确保有索引（时间戳）
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.error("数据必须以时间戳作为索引")
                return False
                
            # 准备数据
            df = data.reset_index().copy()
            df.rename(columns={df.columns[0]: 'time'}, inplace=True)
            df['symbol'] = symbol
            df['source'] = source
            
            # 存储数据
            return self.store(df, self.measurement)
            
        except Exception as e:
            self.logger.error(f"存储市场数据失败: {e}")
            return False
    
    def delete(self, table_name: Optional[str] = None, conditions: Optional[Dict] = None) -> bool:
        """
        从TimescaleDB中删除数据。
        
        Args:
            table_name: 表名，默认使用self.measurement
            conditions: 删除条件，键为列名，值为要匹配的值
            
        Returns:
            删除是否成功
        """
        if not self.is_connected_flag:
            self.logger.error("未连接到TimescaleDB")
            return False
            
        if table_name is None:
            table_name = self.measurement
            
        try:
            # 检查表是否存在
            self.cursor.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = %s AND table_name = %s);",
                (self.schema, table_name)
            )
            
            table_exists = self.cursor.fetchone()[0]
            
            if not table_exists:
                self.logger.error(f"表 {table_name} 不存在")
                return False
                
            # 构建WHERE条件
            where_clauses = []
            params = []
            
            if conditions:
                for col, value in conditions.items():
                    if isinstance(value, (list, tuple)):
                        placeholders = ", ".join(["%s"] * len(value))
                        where_clauses.append(f"{col} IN ({placeholders})")
                        params.extend(value)
                    else:
                        where_clauses.append(f"{col} = %s")
                        params.append(value)
            
            if where_clauses:
                query = f"DELETE FROM {table_name} WHERE {' AND '.join(where_clauses)}"
            else:
                query = f"DELETE FROM {table_name}"
                
            self.cursor.execute(query, params)
            
            rows_deleted = self.cursor.rowcount
            self.conn.commit()
            self.logger.info(f"从表 {table_name} 删除了 {rows_deleted} 条数据")
            return True
            
        except Exception as e:
            self.logger.error(f"删除数据失败: {e}")
            self.conn.rollback()
            return False
    
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
        if not self.is_connected_flag:
            if not self.connect():
                return False
                
        try:
            conditions = {'symbol': symbol}
            
            if start_date and end_date:
                conditions['time'] = f">= '{start_date}' AND time <= '{end_date}'"
            elif start_date:
                conditions['time'] = f">= '{start_date}'"
            elif end_date:
                conditions['time'] = f"<= '{end_date}'"
                
            # 使用通用的delete方法
            return self.delete(self.measurement, conditions)
            
        except Exception as e:
            self.logger.error(f"删除市场数据失败: {e}")
            return False
    
    def query(
        self, 
        query: str, 
        params: Optional[Union[Dict[str, Any], List[Any], tuple]] = None
    ) -> Optional[pd.DataFrame]:
        """
        执行自定义查询。
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果DataFrame，如果失败则返回None
        """
        if not self.is_connected_flag:
            self.logger.error("未连接到TimescaleDB")
            return None
            
        try:
            self.cursor.execute(query, params)
            
            # 获取列名
            column_names = [desc[0] for desc in self.cursor.description]
            
            # 获取结果
            results = self.cursor.fetchall()
            
            # 将结果转换为DataFrame
            df = pd.DataFrame(results, columns=column_names)
            
            self.logger.info(f"查询成功，返回 {len(df)} 条结果")
            return df
            
        except Exception as e:
            self.logger.error(f"查询失败: {e}")
            return None
    
    def query_data(
        self,
        table_name: Optional[str] = None,
        symbol: Optional[Union[str, List[str]]] = None,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        order_by: str = "time ASC"
    ) -> Optional[pd.DataFrame]:
        """
        查询时间序列数据。
        
        Args:
            table_name: 表名，默认使用self.measurement
            symbol: 股票代码或代码列表
            start_time: 开始时间
            end_time: 结束时间
            columns: 要查询的列，默认为全部
            limit: 限制返回的记录数
            order_by: 排序规则，默认为按时间升序
            
        Returns:
            查询结果DataFrame，如果失败则返回None
        """
        if not self.is_connected_flag:
            self.logger.error("未连接到TimescaleDB")
            return None
            
        if table_name is None:
            table_name = self.measurement
            
        try:
            # 构建查询
            select_clause = "*"
            if columns:
                select_clause = ", ".join(columns)
                
            query = f"SELECT {select_clause} FROM {table_name}"
            
            # 添加条件
            conditions = []
            params = []
            
            if symbol:
                if isinstance(symbol, list):
                    placeholders = ", ".join(["%s"] * len(symbol))
                    conditions.append(f"symbol IN ({placeholders})")
                    params.extend(symbol)
                else:
                    conditions.append("symbol = %s")
                    params.append(symbol)
            
            if start_time:
                if isinstance(start_time, str):
                    start_time = pd.to_datetime(start_time)
                conditions.append("time >= %s")
                params.append(start_time)
                
            if end_time:
                if isinstance(end_time, str):
                    end_time = pd.to_datetime(end_time)
                conditions.append("time <= %s")
                params.append(end_time)
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            # 添加排序
            query += f" ORDER BY {order_by}"
            
            # 添加限制
            if limit:
                query += f" LIMIT {limit}"
                
            # 执行查询
            self.cursor.execute(query, params)
            
            # 获取列名
            column_names = [desc[0] for desc in self.cursor.description]
            
            # 获取结果
            results = self.cursor.fetchall()
            
            # 将结果转换为DataFrame
            df = pd.DataFrame(results, columns=column_names)
            
            self.logger.info(f"查询成功，返回 {len(df)} 条结果")
            return df
            
        except Exception as e:
            self.logger.error(f"查询数据失败: {e}")
            return None
    
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
        if not self.is_connected_flag:
            if not self.connect():
                return pd.DataFrame()
                
        try:
            # 使用通用的查询方法
            result = self.query_data(
                table_name=self.measurement,
                symbol=symbol,
                start_time=start_date,
                end_time=end_date,
                columns=fields,
                order_by="time ASC"
            )
            
            if result is None or result.empty:
                return pd.DataFrame()
                
            # 将time列设置为索引
            if 'time' in result.columns:
                result.set_index('time', inplace=True)
                
            return result
            
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {e}")
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
        # 在TimescaleDB中，我们可以使用INSERT ... ON CONFLICT 语句
        # 这样就可以直接使用store_data方法
        return self.store_data(symbol, new_data, source)
    
    def get_available_symbols(self) -> List[Dict[str, Union[str, datetime]]]:
        """
        Get a list of all available symbols in the database.
        
        Returns:
            List of dictionaries containing symbol information
        """
        if not self.is_connected_flag:
            if not self.connect():
                return []
                
        try:
            query = f"""
            SELECT 
                symbol,
                MIN(time) as first_timestamp,
                MAX(time) as last_timestamp,
                COUNT(*) as data_points
            FROM 
                {self.measurement}
            GROUP BY 
                symbol
            ORDER BY 
                symbol
            """
            
            result = self.query(query)
            
            if result is None or result.empty:
                return []
                
            # 转换为所需的格式
            symbols_info = []
            for _, row in result.iterrows():
                symbols_info.append({
                    'symbol': row['symbol'],
                    'first_timestamp': row['first_timestamp'],
                    'last_timestamp': row['last_timestamp'],
                    'data_points': int(row['data_points'])
                })
                
            return symbols_info
            
        except Exception as e:
            self.logger.error(f"获取可用交易对失败: {e}")
            return []
    
    def get_first_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        Get the timestamp of the first available data point for a symbol.
        
        Args:
            symbol: The ticker symbol or asset identifier
            
        Returns:
            The earliest timestamp, or None if no data is available
        """
        if not self.is_connected_flag:
            if not self.connect():
                return None
                
        try:
            query = f"""
            SELECT 
                MIN(time) as first_timestamp
            FROM 
                {self.measurement}
            WHERE 
                symbol = %s
            """
            
            result = self.query(query, [symbol])
            
            if result is None or result.empty or pd.isna(result.iloc[0, 0]):
                return None
                
            return result.iloc[0, 0]
            
        except Exception as e:
            self.logger.error(f"获取第一个时间戳失败: {e}")
            return None
    
    def get_last_timestamp(self, symbol: str) -> Optional[datetime]:
        """
        Get the timestamp of the last available data point for a symbol.
        
        Args:
            symbol: The ticker symbol or asset identifier
            
        Returns:
            The latest timestamp, or None if no data is available
        """
        if not self.is_connected_flag:
            if not self.connect():
                return None
                
        try:
            query = f"""
            SELECT 
                MAX(time) as last_timestamp
            FROM 
                {self.measurement}
            WHERE 
                symbol = %s
            """
            
            result = self.query(query, [symbol])
            
            if result is None or result.empty or pd.isna(result.iloc[0, 0]):
                return None
                
            return result.iloc[0, 0]
            
        except Exception as e:
            self.logger.error(f"获取最后一个时间戳失败: {e}")
            return None 