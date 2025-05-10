#!/usr/bin/env python
"""
InfluxDB使用示例。
演示如何使用InfluxDB存储库保存和检索数据。
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入数据库模块
from src.data.database import create_influxdb_repository


def create_sample_data(symbol: str, days: int = 10) -> pd.DataFrame:
    """
    创建示例市场数据。
    
    Args:
        symbol: 股票代码
        days: 天数
        
    Returns:
        包含OHLCV数据的DataFrame
    """
    # 创建日期范围
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = [end_date - timedelta(days=i) for i in range(days)]
    dates.reverse()  # 从早到晚排序
    
    # 生成随机价格
    last_close = 100.0 + np.random.rand() * 50  # 初始价格在100-150之间
    data = []
    
    for date in dates:
        # 生成当天价格
        change_pct = (np.random.rand() - 0.5) * 0.05  # -2.5%到+2.5%的变化
        open_price = last_close * (1 + (np.random.rand() - 0.5) * 0.01)  # 相对昨收小幅波动
        high_price = max(open_price, last_close) * (1 + np.random.rand() * 0.02)  # 确保high大于open和last_close
        low_price = min(open_price, last_close) * (1 - np.random.rand() * 0.02)  # 确保low小于open和last_close
        close_price = last_close * (1 + change_pct)  # 今收基于昨收计算变化
        volume = int(np.random.rand() * 1000000) + 500000  # 50万到150万的成交量
        
        # 记录行情
        data.append({
            'timestamp': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
        # 更新last_close用于下一个交易日
        last_close = close_price
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


def main():
    """
    主函数。
    """
    try:
        # 创建InfluxDB存储库
        repo = create_influxdb_repository()
        logger.info("创建InfluxDB存储库成功")
        
        # 创建样本数据
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        for symbol in symbols:
            # 生成示例数据
            data = create_sample_data(symbol, days=30)
            logger.info(f"为 {symbol} 创建了 {len(data)} 条示例数据")
            
            # 存储数据
            success = repo.store_data(symbol, data, source='example')
            if success:
                logger.info(f"成功存储 {symbol} 的数据")
            else:
                logger.error(f"存储 {symbol} 的数据失败")
        
        # 检索数据
        for symbol in symbols:
            # 获取最近7天的数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            data = repo.get_data(symbol, start_date=start_date, end_date=end_date)
            if not data.empty:
                logger.info(f"检索到 {symbol} 的 {len(data)} 条数据")
                logger.info(f"数据示例:\n{data.head()}")
            else:
                logger.warning(f"未检索到 {symbol} 的数据")
        
        # 获取可用的股票列表
        symbols_info = repo.get_available_symbols()
        logger.info(f"数据库中有 {len(symbols_info)} 个股票")
        for info in symbols_info:
            logger.info(f"股票: {info['symbol']}, 起始日期: {info['first_date']}, 结束日期: {info['last_date']}")
        
        # 断开连接
        repo.disconnect()
        logger.info("已断开与InfluxDB的连接")
        
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)


if __name__ == "__main__":
    main() 