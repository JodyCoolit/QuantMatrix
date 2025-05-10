#!/usr/bin/env python
"""
InfluxDB写入和查询测试。
演示如何使用InfluxDB存储库写入和查询实际数据。
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
import json

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 也设置其他相关模块的日志级别
logging.getLogger('src.data.database').setLevel(logging.DEBUG)

# 首先检查InfluxDB是否运行
def check_influxdb(url):
    """检查InfluxDB是否在指定URL运行"""
    try:
        # 从配置文件中获取URL和token
        config_path = os.path.join(project_root, 'config', 'influxdb_config.json')
        token = None
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                url = config.get('url', url)
                token = config.get('token')
                
        # 尝试连接到InfluxDB的健康检查端点
        health_url = f"{url}/health"
        logger.info(f"尝试连接到InfluxDB: {health_url}")
        
        headers = {}
        if token:
            auth_header = config.get('auth_header', 'Authorization')
            headers[auth_header] = f"Token {token}"
            logger.debug(f"使用认证头: {auth_header}")
            
        response = requests.get(health_url, headers=headers, timeout=5)
        if response.status_code == 200:
            logger.info("InfluxDB正在运行")
            return True
        else:
            logger.warning(f"InfluxDB可能未正常运行，状态码: {response.status_code}")
            logger.warning(f"响应内容: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"无法连接到InfluxDB: {e}")
        logger.warning("请确保InfluxDB服务正在运行，并且能够在配置的URL上访问")
        return False

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
    
    # 创建DataFrame - 不设置索引，保持timestamp作为列
    df = pd.DataFrame(data)
    
    return df


def main():
    """
    主函数。
    """
    # 检查InfluxDB是否运行
    if not check_influxdb("http://localhost:8181"):
        logger.error("InfluxDB服务未运行，请先启动服务")
        logger.info("如果使用不同的URL，请更新配置文件或修改此脚本中的URL")
        return
        
    try:
        # 创建InfluxDB存储库
        repo = create_influxdb_repository()
        logger.info("创建InfluxDB存储库成功")
        
        # 检查连接状态
        if repo.is_connected():
            logger.info("InfluxDB连接状态: 已连接")
        else:
            logger.error("InfluxDB连接状态: 未连接")
            return
        
        # 创建样本数据
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        for symbol in symbols:
            # 生成示例数据
            data = create_sample_data(symbol, days=30)
            logger.info(f"为 {symbol} 创建了 {len(data)} 条示例数据")
            logger.info(f"数据样例:\n{data.head(2)}")
            
            # 存储数据
            try:
                success = repo.store_data(symbol, data, source='example')
                if success:
                    logger.info(f"成功存储 {symbol} 的数据")
                else:
                    logger.error(f"存储 {symbol} 的数据失败")
            except Exception as e:
                logger.error(f"存储 {symbol} 数据时发生异常: {e}", exc_info=True)
        
        # 检索数据
        for symbol in symbols:
            # 获取最近7天的数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            try:
                data = repo.get_data(symbol, start_date=start_date, end_date=end_date)
                if not data.empty:
                    logger.info(f"检索到 {symbol} 的 {len(data)} 条数据")
                    logger.info(f"数据示例:\n{data.head(2)}")
                else:
                    logger.warning(f"未检索到 {symbol} 的数据")
            except Exception as e:
                logger.error(f"检索 {symbol} 数据时发生异常: {e}", exc_info=True)
        
        # 获取可用的股票列表
        try:
            symbols_info = repo.get_available_symbols()
            logger.info(f"数据库中有 {len(symbols_info)} 个股票")
            for info in symbols_info:
                logger.info(f"股票: {info['symbol']}, 起始日期: {info['first_date']}, 结束日期: {info['last_date']}")
        except Exception as e:
            logger.error(f"获取可用股票列表时发生异常: {e}", exc_info=True)
        
        # 断开连接
        repo.disconnect()
        logger.info("已断开与InfluxDB的连接")
        
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)


if __name__ == "__main__":
    main() 