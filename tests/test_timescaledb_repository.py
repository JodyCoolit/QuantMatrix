"""
TimescaleDB存储库测试模块。
测试TimescaleDB存储库的存储和查询功能。
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('timescaledb_repository_test')


class TimescaleDBRepositoryTester:
    """TimescaleDB存储库测试类"""
    
    def __init__(self):
        """初始化测试器"""
        self.repo = None
        self.is_connected = False
        self.test_table = 'test_market_data'
        self.test_symbols = ['TEST_AAPL', 'TEST_MSFT', 'TEST_GOOG', 'TEST_AMZN', 'TEST_TSLA']
    
    def generate_sample_data(self, days=10):
        """
        生成样本数据。
        
        Args:
            days: 天数
            
        Returns:
            样本数据DataFrame
        """
        data = []
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)
        
        # 为每个股票生成数据
        for symbol in self.test_symbols:
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
    
    def setup_repository(self):
        """设置TimescaleDB存储库"""
        try:
            # 连接到TimescaleDB
            logger.info("连接到TimescaleDB...")
            self.repo = create_timescaledb_repository()
            self.is_connected = True
            logger.info("成功连接到TimescaleDB")
            return True
        except Exception as e:
            logger.error(f"连接到TimescaleDB时出错: {e}")
            self.is_connected = False
            return False
    
    def cleanup(self):
        """清理测试数据"""
        if not self.is_connected or not self.repo:
            logger.warning("未连接到数据库，无法清理")
            return False
        
        try:
            logger.info(f"清理测试表 {self.test_table}...")
            # 删除测试表
            drop_query = f"DROP TABLE IF EXISTS {self.test_table}"
            self.repo.execute(drop_query)
            logger.info(f"成功清理测试表 {self.test_table}")
            return True
        except Exception as e:
            logger.error(f"清理测试表时出错: {e}")
            return False
        finally:
            # 断开连接
            if self.is_connected:
                self.repo.disconnect()
                logger.info("断开TimescaleDB连接")
                self.is_connected = False
    
    def test_store_data(self):
        """测试数据存储功能"""
        if not self.is_connected:
            logger.error("未连接到数据库，无法进行存储测试")
            return False
        
        try:
            # 生成示例数据
            logger.info(f"生成样本数据，股票代码: {self.test_symbols}")
            sample_data = self.generate_sample_data()
            
            # 存储数据
            logger.info(f"将数据存储到表 {self.test_table}...")
            success = self.repo.store(sample_data, self.test_table)
            
            if success:
                logger.info(f"成功存储 {len(sample_data)} 条数据")
                return True
            else:
                logger.error("存储数据失败")
                return False
                
        except Exception as e:
            logger.error(f"测试数据存储时出错: {e}")
            return False
    
    def test_query_data(self):
        """测试数据查询功能"""
        if not self.is_connected:
            logger.error("未连接到数据库，无法进行查询测试")
            return False
        
        try:
            # 测试各种查询
            query_tests = {}
            
            # 示例1：查询特定股票
            logger.info(f"\n1. 查询特定股票 ({self.test_symbols[0]}):")
            apple_data = self.repo.query_data(table_name=self.test_table, symbol=self.test_symbols[0])
            if apple_data is not None:
                logger.info(f"找到 {len(apple_data)} 条记录")
                logger.info(f"数据示例:\n{apple_data.head() if len(apple_data) > 0 else '无数据'}")
                query_tests["查询特定股票"] = len(apple_data) > 0
            else:
                logger.error("查询特定股票失败")
                query_tests["查询特定股票"] = False
            
            # 示例2：查询特定时间范围
            one_week_ago = datetime.now() - timedelta(days=7)
            logger.info(f"\n2. 查询最近一周的数据:")
            recent_data = self.repo.query_data(
                table_name=self.test_table, 
                start_time=one_week_ago
            )
            if recent_data is not None:
                logger.info(f"找到 {len(recent_data)} 条记录")
                logger.info(f"数据示例:\n{recent_data.head() if len(recent_data) > 0 else '无数据'}")
                query_tests["查询时间范围"] = len(recent_data) > 0
            else:
                logger.error("查询时间范围失败")
                query_tests["查询时间范围"] = False
            
            # 示例3：同时按股票和时间范围查询
            three_days_ago = datetime.now() - timedelta(days=3)
            logger.info(f"\n3. 查询特定股票 ({self.test_symbols[1]}) 最近3天的数据:")
            msft_recent = self.repo.query_data(
                table_name=self.test_table,
                symbol=self.test_symbols[1],
                start_time=three_days_ago
            )
            if msft_recent is not None:
                logger.info(f"找到 {len(msft_recent)} 条记录")
                logger.info(f"数据示例:\n{msft_recent.head() if len(msft_recent) > 0 else '无数据'}")
                query_tests["组合查询"] = len(msft_recent) > 0
            else:
                logger.error("组合查询失败")
                query_tests["组合查询"] = False
            
            # 示例4：使用自定义SQL
            logger.info("\n4. 使用自定义SQL查询 (计算每只股票的平均收盘价):")
            avg_query = f"""
            SELECT 
                symbol, 
                AVG(close) as avg_close 
            FROM 
                {self.test_table} 
            GROUP BY 
                symbol
            """
            avg_results = self.repo.query(avg_query)
            if avg_results is not None:
                logger.info(f"查询结果:\n{avg_results}")
                query_tests["自定义SQL查询"] = len(avg_results) > 0
            else:
                logger.error("自定义SQL查询失败")
                query_tests["自定义SQL查询"] = False
            
            # 示例5：时间聚合查询 (使用TimescaleDB的时间桶函数)
            logger.info("\n5. 使用TimescaleDB的时间聚合函数:")
            time_bucket_query = f"""
            SELECT 
                time_bucket('1 day', time) AS day, 
                symbol,
                FIRST(open, time) as open,
                MAX(high) as high,
                MIN(low) as low,
                LAST(close, time) as close,
                SUM(volume) as volume
            FROM 
                {self.test_table} 
            GROUP BY 
                day, symbol
            ORDER BY 
                day DESC, symbol
            LIMIT 10
            """
            time_bucket_results = self.repo.query(time_bucket_query)
            if time_bucket_results is not None:
                logger.info(f"时间聚合结果:\n{time_bucket_results}")
                query_tests["时间聚合查询"] = len(time_bucket_results) > 0
            else:
                logger.error("时间聚合查询失败")
                query_tests["时间聚合查询"] = False
            
            # 打印查询测试结果汇总
            logger.info("\n查询测试结果汇总:")
            for test_name, passed in query_tests.items():
                logger.info(f"  {test_name}: {'✓' if passed else '✗'}")
            
            return all(query_tests.values())
            
        except Exception as e:
            logger.error(f"测试数据查询时出错: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        try:
            # 设置存储库
            if not self.setup_repository():
                logger.error("设置TimescaleDB存储库失败，无法继续测试")
                return False
            
            # 清理之前的测试数据
            self.cleanup()
            
            # 重新连接
            if not self.setup_repository():
                logger.error("重新连接到TimescaleDB失败，无法继续测试")
                return False
            
            # 运行测试
            test_results = {
                "数据存储": self.test_store_data(),
                "数据查询": self.test_query_data()
            }
            
            # 打印测试结果
            logger.info("\n============= TimescaleDB存储库测试结果 =============")
            for test_name, passed in test_results.items():
                logger.info(f"{test_name}: {'✓' if passed else '✗'}")
            logger.info("===============================================\n")
            
            # 确定总体结果
            overall_result = all(test_results.values())
            if overall_result:
                logger.info("所有TimescaleDB存储库测试通过！")
            else:
                logger.warning("部分TimescaleDB存储库测试失败，请检查日志获取详细信息。")
            
            return overall_result
            
        finally:
            # 清理测试数据
            self.cleanup()


def main():
    """运行TimescaleDB存储库测试"""
    tester = TimescaleDBRepositoryTester()
    result = tester.run_all_tests()
    return result


if __name__ == "__main__":
    main() 