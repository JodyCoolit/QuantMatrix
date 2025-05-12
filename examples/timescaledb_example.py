"""
TimescaleDB示例脚本。
展示如何使用TimescaleDB存储和查询时间序列数据。
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 导入数据库模块
from src.data.database import create_timescaledb_repository

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('timescaledb_example')

def generate_sample_data(symbols, days=10):
    """
    生成样本数据。
    
    Args:
        symbols: 股票代码列表
        days: 天数
        
    Returns:
        样本数据DataFrame
    """
    data = []
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    
    # 为每个股票生成数据
    for symbol in symbols:
        current_date = start_date
        base_price = np.random.uniform(50, 200)
        
        while current_date <= end_date:
            # 生成当天的OHLCV数据
            price_volatility = np.random.uniform(0.01, 0.03)
            open_price = base_price * (1 + np.random.uniform(-price_volatility, price_volatility))
            high_price = open_price * (1 + np.random.uniform(0, price_volatility * 2))
            low_price = open_price * (1 - np.random.uniform(0, price_volatility * 2))
            close_price = np.random.uniform(low_price, high_price)
            volume = np.random.randint(1000, 1000000)
            
            # 添加到数据列表
            data.append({
                'time': current_date,
                'symbol': symbol,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
            # 更新基础价格和日期
            base_price = close_price
            current_date += timedelta(days=1)
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    return df

def main():
    """主函数"""
    try:
        # 连接到TimescaleDB
        logger.info("连接到TimescaleDB...")
        repo = create_timescaledb_repository()
        
        # 生成示例数据
        symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
        logger.info(f"生成样本数据，股票代码: {symbols}")
        sample_data = generate_sample_data(symbols)
        
        # 存储数据
        table_name = 'market_data'
        logger.info(f"将数据存储到表 {table_name}...")
        success = repo.store(sample_data, table_name)
        
        if success:
            logger.info(f"成功存储 {len(sample_data)} 条数据")
            
            # 查询数据示例
            logger.info("查询数据示例:")
            
            # 示例1：查询特定股票
            logger.info("\n1. 查询特定股票 (AAPL):")
            apple_data = repo.query_data(table_name=table_name, symbol='AAPL')
            if apple_data is not None:
                logger.info(f"找到 {len(apple_data)} 条记录")
                print(apple_data.head())
            
            # 示例2：查询特定时间范围
            one_week_ago = datetime.now() - timedelta(days=7)
            logger.info(f"\n2. 查询最近一周的数据:")
            recent_data = repo.query_data(
                table_name=table_name, 
                start_time=one_week_ago
            )
            if recent_data is not None:
                logger.info(f"找到 {len(recent_data)} 条记录")
                print(recent_data.head())
            
            # 示例3：同时按股票和时间范围查询
            three_days_ago = datetime.now() - timedelta(days=3)
            logger.info(f"\n3. 查询特定股票 (MSFT) 最近3天的数据:")
            msft_recent = repo.query_data(
                table_name=table_name,
                symbol='MSFT',
                start_time=three_days_ago
            )
            if msft_recent is not None:
                logger.info(f"找到 {len(msft_recent)} 条记录")
                print(msft_recent)
            
            # 示例4：使用自定义SQL
            logger.info("\n4. 使用自定义SQL查询 (计算每只股票的平均收盘价):")
            avg_query = """
            SELECT 
                symbol, 
                AVG(close) as avg_close 
            FROM 
                market_data 
            GROUP BY 
                symbol
            """
            avg_results = repo.query(avg_query)
            if avg_results is not None:
                logger.info(f"计算结果:")
                print(avg_results)
            
            # 示例5：时间聚合查询 (使用TimescaleDB的时间桶函数)
            logger.info("\n5. 使用TimescaleDB的时间聚合函数:")
            time_bucket_query = """
            SELECT 
                time_bucket('1 day', time) AS day, 
                symbol,
                FIRST(open, time) as open,
                MAX(high) as high,
                MIN(low) as low,
                LAST(close, time) as close,
                SUM(volume) as volume
            FROM 
                market_data 
            GROUP BY 
                day, symbol
            ORDER BY 
                day DESC, symbol
            LIMIT 10
            """
            time_bucket_results = repo.query(time_bucket_query)
            if time_bucket_results is not None:
                logger.info(f"时间聚合结果:")
                print(time_bucket_results)
                
        else:
            logger.error("存储数据失败")
            
        # 断开连接
        repo.disconnect()
        logger.info("示例完成")
        
    except Exception as e:
        logger.error(f"运行示例时出错: {e}")
        raise

if __name__ == "__main__":
    main() 