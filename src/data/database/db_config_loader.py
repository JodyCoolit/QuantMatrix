"""
数据库配置加载器。
用于加载和验证数据库配置。
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional


class DBConfigLoader:
    """
    数据库配置加载器类，用于加载和验证数据库配置。
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化配置加载器。
        
        Args:
            config_dir: 配置文件目录，默认为项目根目录下的config目录
        """
        self.logger = logging.getLogger(__name__)
        
        # 如果未指定配置目录，使用默认路径
        if config_dir is None:
            # 假设当前文件在src/data/database目录下
            # 项目根目录在三级目录之上
            project_root = Path(__file__).parent.parent.parent.parent
            self.config_dir = os.path.join(project_root, 'config')
        else:
            self.config_dir = config_dir
    
    def load_config(self, config_file: str = 'db_config.json') -> Dict[str, Any]:
        """
        加载数据库配置文件。
        
        Args:
            config_file: 配置文件名，默认为db_config.json
            
        Returns:
            配置信息字典
        """
        config_path = os.path.join(self.config_dir, config_file)
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.logger.info(f"成功加载数据库配置: {config_path}")
            return config
            
        except FileNotFoundError:
            self.logger.error(f"找不到数据库配置文件: {config_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"数据库配置文件格式错误: {config_path}")
            raise
        except Exception as e:
            self.logger.error(f"加载数据库配置失败: {e}")
            raise
    
    def get_database_factory_args(self, db_type: str = None) -> Dict[str, Any]:
        """
        获取数据库工厂创建实例所需的参数。
        
        Args:
            db_type: 数据库类型，默认为None，会根据配置自动确定
            
        Returns:
            数据库工厂参数字典
        """
        try:
            # 加载主配置文件获取数据库类型
            main_config = self.load_config('db_config.json')
            
            if db_type is None:
                # 如果没有指定数据库类型，使用配置中的默认类型
                db_type = main_config.get('default_db', 'timescaledb')
            
            # 根据数据库类型加载特定配置
            if db_type.lower() == 'timescaledb':
                config = self.load_config('timescaledb_config.json')
            elif db_type.lower() == 'influxdb':
                config = self.load_config('influxdb_config.json')
            else:
                raise ValueError(f"不支持的数据库类型: {db_type}")
            
            # 合并主配置和特定配置
            for key, value in main_config.items():
                if key not in config and key != 'default_db':
                    config[key] = value
            
            self.logger.info(f"成功生成{db_type}数据库工厂参数")
            return config
            
        except FileNotFoundError:
            # 如果主配置不存在，尝试直接加载特定数据库配置
            if db_type.lower() == 'timescaledb':
                return self.load_config('timescaledb_config.json')
            elif db_type.lower() == 'influxdb':
                return self.load_config('influxdb_config.json')
            else:
                raise ValueError(f"不支持的数据库类型: {db_type}")
        except Exception as e:
            self.logger.error(f"获取数据库工厂参数失败: {e}")
            raise
    
    def save_config(self, config: Dict[str, Any], config_file: str = 'db_config.json') -> None:
        """
        保存数据库配置文件。
        
        Args:
            config: 配置信息字典
            config_file: 配置文件名，默认为db_config.json
        """
        config_path = os.path.join(self.config_dir, config_file)
        
        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"成功保存数据库配置: {config_path}")
            
        except Exception as e:
            self.logger.error(f"保存数据库配置失败: {e}")
            raise 