"""
Example script demonstrating the use of the database abstraction layer.
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.database.database_factory import DatabaseFactory
from src.data.database.db_config_loader import DBConfigLoader


def generate_sample_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Generate sample market data for demonstration.
    
    Args:
        symbol: The ticker symbol
        start_date: Start date for data generation
        end_date: End date for data generation
        
    Returns:
        DataFrame with sample market data
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate random price data with a slight upward trend
    np.random.seed(hash(symbol) % 10000)  # Use symbol name as seed for reproducibility
    
    n_days = len(date_range)
    
    # Generate more stable daily returns
    daily_returns = np.random.normal(0.0005, 0.01, n_days)  # Slight positive bias
    
    # Calculate prices
    prices = 100 * (1 + daily_returns).cumprod()
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_days))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_days)
    }, index=date_range)
    
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def main():
    """Main function demonstrating database usage."""
    # 加载数据库配置
    config_loader = DBConfigLoader()
    db_args = config_loader.get_database_factory_args()
    
    # 创建数据库实例
    print(f"Creating database connection with config: {db_args}")
    db = DatabaseFactory.create('influxdb', **db_args)
    
    try:
        # 连接到数据库
        print("Connecting to database...")
        db.connect()
        print("Connected successfully!")
        
        # 生成样本数据
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        for symbol in symbols:
            print(f"Generating data for {symbol}...")
            data = generate_sample_data(symbol, start_date, end_date)
            
            # 存储数据
            print(f"Storing data for {symbol}...")
            success = db.store_data(symbol, data, source='sample')
            
            if success:
                print(f"Successfully stored data for {symbol}")
            else:
                print(f"Failed to store data for {symbol}")
        
        # 获取可用的股票代码
        print("\nAvailable symbols:")
        symbols = db.get_available_symbols()
        for symbol_info in symbols:
            print(f"  {symbol_info['symbol']}: {symbol_info['first_date']} to {symbol_info['last_date']}")
        
        # 检索股票数据
        symbol = 'AAPL'
        print(f"\nRetrieving data for {symbol}...")
        data = db.get_data(symbol, start_date, end_date)
        
        if not data.empty:
            print(f"Retrieved {len(data)} data points for {symbol}")
            print("\nFirst few rows:")
            print(data.head())
        else:
            print(f"No data found for {symbol}")
        
        # 更新数据
        print(f"\nUpdating data for {symbol}...")
        new_data = generate_sample_data(symbol, end_date, end_date + timedelta(days=5))
        success = db.update_data(symbol, new_data, source='sample')
        
        if success:
            print(f"Successfully updated data for {symbol}")
        else:
            print(f"Failed to update data for {symbol}")
        
        # 获取第一个和最后一个时间戳
        first_ts = db.get_first_timestamp(symbol)
        last_ts = db.get_last_timestamp(symbol)
        
        print(f"\nData range for {symbol}:")
        print(f"  First timestamp: {first_ts}")
        print(f"  Last timestamp: {last_ts}")
        
        # 删除数据
        print(f"\nDeleting data for {symbol}...")
        success = db.delete_data(symbol, end_date, end_date + timedelta(days=5))
        
        if success:
            print(f"Successfully deleted data for {symbol}")
        else:
            print(f"Failed to delete data for {symbol}")
    
    finally:
        # 断开与数据库的连接
        db.disconnect()
        print("\nDisconnected from database")


if __name__ == "__main__":
    main() 