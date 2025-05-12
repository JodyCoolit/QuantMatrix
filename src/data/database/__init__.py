"""
Database abstraction layer for QuantMatrix.
Provides interfaces and implementations for various database backends.
"""
from .database_interface import TimeSeriesDatabaseInterface
from .timescaledb_repository import TimescaleDBRepository
from .database_factory import DatabaseFactory
from .db_config_loader import DBConfigLoader

# 导出主要类
__all__ = [
    'TimeSeriesDatabaseInterface',
    'TimescaleDBRepository',
    'DatabaseFactory', 
    'DBConfigLoader'
]

# 创建便捷函数
def create_timescaledb_repository(config_file=None):
    """
    便捷函数：创建TimescaleDB存储库实例。
    
    Args:
        config_file: 配置文件路径，默认使用标准路径
        
    Returns:
        已连接的TimescaleDB存储库实例
    """
    return DatabaseFactory.create_timescaledb_repository(config_file)

def get_db_config():
    """
    便捷函数：获取数据库配置。
    
    Returns:
        数据库配置字典
    """
    loader = DBConfigLoader()
    return loader.load_config() 