"""
数据库工厂模块。
用于创建数据库接口实例。
"""
import logging
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from .database_interface import TimeSeriesDatabaseInterface
from .influxdb_repository import InfluxDBRepository


class DatabaseFactory:
    """
    数据库工厂类，用于创建数据库接口实例。
    """
    
    @staticmethod
    def create_influxdb_repository(config_file: Optional[str] = None) -> InfluxDBRepository:
        """
        创建InfluxDB存储库实例。
        
        Args:
            config_file: InfluxDB配置文件路径，默认为项目根目录下的config/influxdb_config.json
            
        Returns:
            InfluxDBRepository实例
        """
        logger = logging.getLogger(__name__)
        
        # 如果未指定配置文件，使用默认路径
        if config_file is None:
            # 假设当前文件在src/data/database目录下
            # 项目根目录在三级目录之上
            project_root = Path(__file__).parent.parent.parent.parent
            config_file = os.path.join(project_root, 'config', 'influxdb_config.json')
        
        try:
            # 加载配置文件
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            logger.info(f"成功加载InfluxDB配置: {config_file}")
            
            # 创建InfluxDB存储库实例
            repo = InfluxDBRepository(
                url=config['url'],
                token=config['token'],
                bucket=config['bucket'],
                measurement=config.get('measurement', 'market_data')
            )
            
            # 尝试连接
            repo.connect()
            
            logger.info("InfluxDB存储库创建成功并已连接")
            return repo
            
        except FileNotFoundError:
            logger.error(f"找不到InfluxDB配置文件: {config_file}")
            raise
        except json.JSONDecodeError:
            logger.error(f"InfluxDB配置文件格式错误: {config_file}")
            raise
        except Exception as e:
            logger.error(f"创建InfluxDB存储库失败: {e}")
            raise
    
    @staticmethod
    def create_database(db_type: str, config: Optional[Dict[str, Any]] = None) -> TimeSeriesDatabaseInterface:
        """
        根据数据库类型创建相应的数据库接口实例。
        
        Args:
            db_type: 数据库类型，目前支持'influxdb'
            config: 配置信息，如为None则从默认配置文件读取
            
        Returns:
            TimeSeriesDatabaseInterface实例
        """
        logger = logging.getLogger(__name__)
        
        if db_type.lower() == 'influxdb':
            if config is None:
                return DatabaseFactory.create_influxdb_repository()
            else:
                return InfluxDBRepository(
                    url=config['url'],
                    token=config['token'],
                    bucket=config['bucket'],
                    measurement=config.get('measurement', 'market_data')
                )
        else:
            logger.error(f"不支持的数据库类型: {db_type}")
            raise ValueError(f"不支持的数据库类型: {db_type}") 