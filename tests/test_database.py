"""
数据库模块测试。
测试数据库工厂、连接和操作功能。
"""
import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data.database.database_factory import DatabaseFactory
from src.data.database.db_config_loader import DBConfigLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('database_test')


class DatabaseTester:
    """数据库功能测试类"""
    
    def __init__(self, db_type="timescaledb"):
        """
        初始化数据库测试类
        
        Args:
            db_type: 要测试的数据库类型
        """
        self.db_type = db_type
        self.db = None
        self.is_connected = False
        self.test_symbols = ['TEST_AAPL', 'TEST_MSFT', 'TEST_GOOGL']
    
    def generate_sample_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        生成样本市场数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含示例数据的DataFrame
        """
        # 创建日期范围
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 为了能够标识测试数据，在每个股票前添加TEST_前缀
        np.random.seed(hash(symbol) % 10000)  # 使用股票代码作为随机数种子以确保可重现性
        
        n_days = len(date_range)
        
        # 生成随机价格数据
        daily_returns = np.random.normal(0.0005, 0.01, n_days)  # 轻微的正偏差
        
        # 计算价格
        prices = 100 * (1 + daily_returns).cumprod()
        
        # 创建DataFrame
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_days))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_days)
        }, index=date_range)
        
        # 确保high是最高值，low是最低值
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def setup_database(self):
        """设置数据库连接"""
        try:
            # 加载数据库配置
            config_loader = DBConfigLoader()
            db_args = config_loader.get_database_factory_args()
            
            logger.info(f"使用配置创建{self.db_type}数据库连接: {db_args}")
            self.db = DatabaseFactory.create(self.db_type, **db_args)
            
            # 连接到数据库
            logger.info("连接到数据库...")
            self.is_connected = self.db.connect()
            
            if self.is_connected:
                logger.info("数据库连接成功！")
                return True
            else:
                logger.error("数据库连接失败")
                return False
                
        except Exception as e:
            logger.error(f"设置数据库时出错: {e}")
            return False
    
    def cleanup(self):
        """清理测试期间创建的数据"""
        if not self.is_connected:
            logger.warning("数据库未连接，无法清理")
            return False
        
        try:
            logger.info("清理测试数据...")
            
            for symbol in self.test_symbols:
                logger.info(f"删除{symbol}的数据...")
                # 时间范围设定为很大，确保删除所有测试数据
                start_date = datetime(2000, 1, 1)
                end_date = datetime.now() + timedelta(days=365)
                self.db.delete_data(symbol, start_date, end_date)
            
            logger.info("测试数据清理完成")
            return True
        except Exception as e:
            logger.error(f"清理测试数据时出错: {e}")
            return False
        finally:
            # 断开数据库连接
            if self.is_connected:
                self.db.disconnect()
                logger.info("数据库连接已断开")
    
    def test_store_and_retrieve(self):
        """测试数据存储和检索功能"""
        if not self.is_connected:
            logger.error("数据库未连接，无法进行存储测试")
            return False
        
        success = True
        
        try:
            # 生成测试数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=10)
            
            for symbol in self.test_symbols:
                # 生成数据
                logger.info(f"为{symbol}生成测试数据...")
                data = self.generate_sample_data(symbol, start_date, end_date)
                
                # 存储数据
                logger.info(f"存储{symbol}的数据...")
                store_success = self.db.store_data(symbol, data, source='test')
                
                if store_success:
                    logger.info(f"成功存储{symbol}的{len(data)}条数据记录")
                    
                    # 检索数据
                    logger.info(f"检索{symbol}的数据...")
                    retrieved_data = self.db.get_data(symbol, start_date, end_date)
                    
                    if not retrieved_data.empty:
                        logger.info(f"成功检索到{len(retrieved_data)}条{symbol}的数据记录")
                        
                        # 数据验证 - 确保记录数量大致匹配
                        # 有时数据库时间戳处理可能导致轻微差异
                        if abs(len(retrieved_data) - len(data)) <= 1:
                            logger.info(f"{symbol}数据存储和检索测试通过")
                        else:
                            logger.warning(f"{symbol}数据记录数不匹配: 原始数据={len(data)}, 检索数据={len(retrieved_data)}")
                            success = False
                    else:
                        logger.error(f"无法检索{symbol}的数据")
                        success = False
                else:
                    logger.error(f"存储{symbol}的数据失败")
                    success = False
            
            return success
        
        except Exception as e:
            logger.error(f"测试数据存储和检索时出错: {e}")
            return False
    
    def test_metadata(self):
        """测试元数据功能，如获取可用股票代码和时间戳"""
        if not self.is_connected:
            logger.error("数据库未连接，无法进行元数据测试")
            return False
        
        try:
            # 获取可用股票代码
            logger.info("获取可用股票代码...")
            symbols = self.db.get_available_symbols()
            
            # 检查我们的测试符号是否存在
            test_symbols_found = [s for s in symbols if s['symbol'] in self.test_symbols]
            logger.info(f"找到{len(test_symbols_found)}个测试股票代码: {[s['symbol'] for s in test_symbols_found]}")
            
            if len(test_symbols_found) < len(self.test_symbols):
                logger.warning(f"未找到所有测试股票代码，预期{len(self.test_symbols)}个，实际{len(test_symbols_found)}个")
            
            # 检查第一个和最后一个时间戳
            for symbol in self.test_symbols:
                logger.info(f"获取{symbol}的时间戳范围...")
                
                first_ts = self.db.get_first_timestamp(symbol)
                last_ts = self.db.get_last_timestamp(symbol)
                
                if first_ts and last_ts:
                    logger.info(f"{symbol}的数据范围: {first_ts} 到 {last_ts}")
                else:
                    logger.warning(f"无法获取{symbol}的时间戳范围")
            
            return True
            
        except Exception as e:
            logger.error(f"测试元数据功能时出错: {e}")
            return False
    
    def test_update(self):
        """测试数据更新功能"""
        if not self.is_connected:
            logger.error("数据库未连接，无法进行更新测试")
            return False
        
        try:
            # 选一个测试股票代码
            symbol = self.test_symbols[0]
            
            # 为新的日期范围生成数据
            current_end = datetime.now()
            update_start = current_end + timedelta(days=1)
            update_end = update_start + timedelta(days=3)
            
            logger.info(f"为{symbol}生成更新数据，日期范围: {update_start} 到 {update_end}")
            update_data = self.generate_sample_data(symbol, update_start, update_end)
            
            # 更新数据
            logger.info(f"更新{symbol}的数据...")
            update_success = self.db.update_data(symbol, update_data, source='test')
            
            if update_success:
                logger.info(f"成功更新{symbol}的数据，添加了{len(update_data)}条记录")
                
                # 验证更新 - 获取扩展日期范围的数据
                fetch_start = datetime.now() - timedelta(days=10)
                fetch_end = update_end + timedelta(days=1)
                
                logger.info(f"验证更新，检索扩展日期范围: {fetch_start} 到 {fetch_end}")
                extended_data = self.db.get_data(symbol, fetch_start, fetch_end)
                
                if not extended_data.empty:
                    # 验证数据包含更新部分
                    last_date = extended_data.index.max()
                    logger.info(f"检索到的最后日期: {last_date}")
                    
                    # 检查最后日期是否接近预期的更新结束日期
                    if (last_date.date() - update_end.date()).days <= 1:
                        logger.info(f"{symbol}数据更新测试通过")
                        return True
                    else:
                        logger.warning(f"更新后的最后日期与预期不符: 预期接近{update_end}，实际{last_date}")
                        return False
                else:
                    logger.error(f"更新后无法检索{symbol}的扩展数据")
                    return False
            else:
                logger.error(f"更新{symbol}的数据失败")
                return False
                
        except Exception as e:
            logger.error(f"测试数据更新功能时出错: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有数据库测试"""
        try:
            # 设置数据库
            if not self.setup_database():
                logger.error("数据库设置失败，无法继续测试")
                return False
            
            # 清理之前的测试数据
            self.cleanup()
            
            # 重新连接
            if not self.setup_database():
                logger.error("数据库重新连接失败，无法继续测试")
                return False
            
            # 运行测试
            test_results = {
                "存储和检索": self.test_store_and_retrieve(),
                "元数据": self.test_metadata(),
                "数据更新": self.test_update()
            }
            
            # 打印测试结果
            logger.info("\n============= 数据库测试结果 =============")
            for test_name, passed in test_results.items():
                logger.info(f"{test_name}: {'✓' if passed else '✗'}")
            logger.info("===============================================\n")
            
            # 确定总体结果
            overall_result = all(test_results.values())
            if overall_result:
                logger.info("所有数据库测试通过！")
            else:
                logger.warning("部分数据库测试失败，请检查日志获取详细信息。")
            
            return overall_result
            
        finally:
            # 清理测试数据
            self.cleanup()


def main():
    """运行数据库测试"""
    tester = DatabaseTester()
    result = tester.run_all_tests()
    return result


if __name__ == "__main__":
    main() 