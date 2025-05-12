"""
TimescaleDB连接测试模块。
用于验证TimescaleDB连接和配置是否正确。
"""
import os
import sys
import json
import logging
import psycopg2
import traceback
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# 配置日志 - 修改为输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('timescaledb_test')


class TimescaleDBTester:
    """TimescaleDB连接和功能测试类"""
    
    def __init__(self):
        """初始化测试器并加载配置"""
        self.config = self.load_config()
        self.connection_status = False
        self.connection_msg = ""
        self.extension_status = False
        self.extension_msg = ""
        self.hypertable_status = False
        self.hypertable_msg = ""
    
    def load_config(self):
        """加载TimescaleDB配置"""
        config_path = os.path.join(PROJECT_ROOT, 'config', 'timescaledb_config.json')
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            logger.info(f"配置内容: {config}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            traceback.print_exc()
            return None

    def test_connection(self):
        """测试与TimescaleDB的基本连接"""
        if not self.config:
            self.connection_status = False
            self.connection_msg = "配置加载失败"
            return False, "配置加载失败"
        
        try:
            logger.info(f"尝试连接到数据库: {self.config['host']}:{self.config['port']}/{self.config['database']}")
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            
            cursor = conn.cursor()
            cursor.execute('SELECT version();')
            version = cursor.fetchone()[0]
            
            logger.info(f"成功连接到PostgreSQL/TimescaleDB: {version}")
            self.connection_status = True
            self.connection_msg = version
            return True, version
        except Exception as e:
            logger.error(f"连接PostgreSQL/TimescaleDB失败: {e}")
            traceback.print_exc()
            self.connection_status = False
            self.connection_msg = str(e)
            return False, str(e)

    def test_timescaledb_extension(self):
        """测试TimescaleDB扩展是否已安装"""
        if not self.connection_status:
            logger.error("数据库连接未建立，无法检查扩展")
            self.extension_status = False
            self.extension_msg = "数据库连接未建立"
            return False, "数据库连接未建立"
        
        try:
            logger.info("检查TimescaleDB扩展是否已安装...")
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb';")
            result = cursor.fetchone()
            
            if result:
                logger.info(f"TimescaleDB扩展已安装: 版本 {result[1]}")
                self.extension_status = True
                self.extension_msg = result[1]
                return True, result[1]
            else:
                logger.warning("TimescaleDB扩展未安装")
                self.extension_status = False
                self.extension_msg = "扩展未安装"
                return False, "扩展未安装"
                
        except Exception as e:
            logger.error(f"检查TimescaleDB扩展失败: {e}")
            traceback.print_exc()
            self.extension_status = False
            self.extension_msg = str(e)
            return False, str(e)

    def test_create_hypertable(self):
        """测试创建超表功能"""
        if not self.extension_status:
            logger.error("TimescaleDB扩展不可用，无法测试超表功能")
            self.hypertable_status = False
            self.hypertable_msg = "TimescaleDB扩展不可用"
            return False, "TimescaleDB扩展不可用"
        
        conn = None
        try:
            logger.info("测试创建TimescaleDB超表...")
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            
            cursor = conn.cursor()
            
            # 创建测试表
            logger.info("1. 尝试创建测试表")
            cursor.execute("DROP TABLE IF EXISTS test_hypertable;")
            cursor.execute("""
            CREATE TABLE test_hypertable (
                time TIMESTAMP NOT NULL,
                value DOUBLE PRECISION,
                device_id TEXT
            );
            """)
            
            # 转换为超表
            logger.info("2. 尝试将表转换为超表")
            cursor.execute("SELECT create_hypertable('test_hypertable', 'time');")
            result = cursor.fetchone()
            logger.info(f"   转换结果: {result}")
            
            # 插入一些测试数据
            logger.info("3. 尝试插入测试数据")
            cursor.execute("""
            INSERT INTO test_hypertable VALUES 
                (NOW(), 22.5, 'dev1'),
                (NOW() - INTERVAL '1 hour', 23.1, 'dev1'),
                (NOW() - INTERVAL '2 hour', 21.8, 'dev2');
            """)
            
            # 查询一下数据
            logger.info("4. 查询插入的数据")
            cursor.execute("SELECT COUNT(*) FROM test_hypertable;")
            count = cursor.fetchone()[0]
            logger.info(f"   插入的数据条数: {count}")
            
            # 检查表是否是超表
            logger.info("5. 验证表是否为超表")
            cursor.execute("""
            SELECT hypertable_name FROM timescaledb_information.hypertables 
            WHERE hypertable_name = 'test_hypertable';
            """)
            is_hypertable = cursor.fetchone() is not None
            logger.info(f"   是超表: {is_hypertable}")
            
            conn.commit()
            
            if is_hypertable and count == 3:
                logger.info("成功创建超表并插入数据")
                self.hypertable_status = True
                self.hypertable_msg = "测试成功，已插入3条数据"
                return True, "测试成功，已插入3条数据"
            else:
                logger.warning(f"创建超表失败或数据插入不正确")
                self.hypertable_status = False
                self.hypertable_msg = f"测试失败, is_hypertable={is_hypertable}, count={count}"
                return False, f"测试失败, is_hypertable={is_hypertable}, count={count}"
                
        except Exception as e:
            logger.error(f"测试创建超表失败: {e}")
            traceback.print_exc()
            self.hypertable_status = False
            self.hypertable_msg = str(e)
            return False, str(e)
        finally:
            if conn:
                try:
                    logger.info("6. 清理测试表")
                    cursor = conn.cursor()
                    cursor.execute("DROP TABLE IF EXISTS test_hypertable;")
                    conn.commit()
                except:
                    pass
                conn.close()

    def run_all_tests(self):
        """运行所有测试并显示结果"""
        logger.info("开始测试TimescaleDB连接...")
        
        # 测试连接
        self.test_connection()
        
        # 如果连接成功，继续测试
        if self.connection_status:
            self.test_timescaledb_extension()
            self.test_create_hypertable()
        
        # 输出测试结果摘要
        logger.info("\n============= TimescaleDB测试结果 =============")
        logger.info(f"基本连接: {'✓' if self.connection_status else '✗'} - {self.connection_msg}")
        
        if self.connection_status:
            logger.info(f"TimescaleDB扩展: {'✓' if self.extension_status else '✗'} - {self.extension_msg}")
            logger.info(f"创建超表测试: {'✓' if self.hypertable_status else '✗'} - {self.hypertable_msg}")
        
        logger.info("===============================================\n")
        
        # 综合结果
        if self.connection_status and self.extension_status and self.hypertable_status:
            logger.info("所有测试通过，TimescaleDB连接配置正确！")
            return True
        else:
            if not self.connection_status:
                logger.error("无法连接到数据库，请检查连接配置。")
            elif not self.extension_status:
                logger.error("可以连接PostgreSQL，但TimescaleDB扩展未安装或不可用。")
            else:
                logger.error("基本功能测试通过，但创建超表测试失败，请检查用户权限。")
            return False


def main():
    """运行TimescaleDB测试"""
    tester = TimescaleDBTester()
    result = tester.run_all_tests()
    return result


if __name__ == "__main__":
    main() 