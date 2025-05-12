"""
TimescaleDB连接测试脚本。
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

def load_config():
    """加载TimescaleDB配置"""
    config_path = os.path.join(PROJECT_ROOT, 'config', 'timescaledb_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"成功加载配置文件: {config_path}")
        print(f"配置内容: {config}")
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        traceback.print_exc()
        return None

def test_connection(config):
    """测试与TimescaleDB的基本连接"""
    try:
        print(f"尝试连接到数据库: {config['host']}:{config['port']}/{config['database']}")
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        
        cursor = conn.cursor()
        cursor.execute('SELECT version();')
        version = cursor.fetchone()[0]
        
        print(f"成功连接到PostgreSQL/TimescaleDB: {version}")
        return True, version
    except Exception as e:
        print(f"连接PostgreSQL/TimescaleDB失败: {e}")
        traceback.print_exc()
        return False, str(e)

def test_timescaledb_extension(config):
    """测试TimescaleDB扩展是否已安装"""
    try:
        print("检查TimescaleDB扩展是否已安装...")
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'timescaledb';")
        result = cursor.fetchone()
        
        if result:
            print(f"TimescaleDB扩展已安装: 版本 {result[1]}")
            return True, result[1]
        else:
            print("TimescaleDB扩展未安装")
            return False, "扩展未安装"
            
    except Exception as e:
        print(f"检查TimescaleDB扩展失败: {e}")
        traceback.print_exc()
        return False, str(e)

def test_create_hypertable(config):
    """测试创建超表功能"""
    try:
        print("测试创建TimescaleDB超表...")
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        
        cursor = conn.cursor()
        
        # 创建测试表
        print("1. 尝试创建测试表")
        cursor.execute("DROP TABLE IF EXISTS test_hypertable;")
        cursor.execute("""
        CREATE TABLE test_hypertable (
            time TIMESTAMP NOT NULL,
            value DOUBLE PRECISION,
            device_id TEXT
        );
        """)
        
        # 转换为超表
        print("2. 尝试将表转换为超表")
        cursor.execute("SELECT create_hypertable('test_hypertable', 'time');")
        result = cursor.fetchone()
        print(f"   转换结果: {result}")
        
        # 插入一些测试数据
        print("3. 尝试插入测试数据")
        cursor.execute("""
        INSERT INTO test_hypertable VALUES 
            (NOW(), 22.5, 'dev1'),
            (NOW() - INTERVAL '1 hour', 23.1, 'dev1'),
            (NOW() - INTERVAL '2 hour', 21.8, 'dev2');
        """)
        
        # 查询一下数据
        print("4. 查询插入的数据")
        cursor.execute("SELECT COUNT(*) FROM test_hypertable;")
        count = cursor.fetchone()[0]
        print(f"   插入的数据条数: {count}")
        
        # 检查表是否是超表
        print("5. 验证表是否为超表")
        cursor.execute("""
        SELECT hypertable_name FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'test_hypertable';
        """)
        is_hypertable = cursor.fetchone() is not None
        print(f"   是超表: {is_hypertable}")
        
        conn.commit()
        
        if is_hypertable and count == 3:
            print("成功创建超表并插入数据")
            return True, "测试成功，已插入3条数据"
        else:
            print(f"创建超表失败或数据插入不正确")
            return False, f"测试失败, is_hypertable={is_hypertable}, count={count}"
            
    except Exception as e:
        print(f"测试创建超表失败: {e}")
        traceback.print_exc()
        return False, str(e)
    finally:
        if 'conn' in locals() and conn:
            try:
                print("6. 清理测试表")
                cursor = conn.cursor()
                cursor.execute("DROP TABLE IF EXISTS test_hypertable;")
                conn.commit()
            except:
                pass
            conn.close()

def main():
    """主函数"""
    print("开始测试TimescaleDB连接...")
    
    # 加载配置
    config = load_config()
    if not config:
        return
    
    # 初始化测试结果
    connection_status = False
    
    # 运行测试
    connection_status, connection_msg = test_connection(config)
    
    # 如果基本连接成功，继续测试扩展和超表功能
    if connection_status:
        extension_status, extension_msg = test_timescaledb_extension(config)
        hypertable_status, hypertable_msg = test_create_hypertable(config)
        
        # 输出测试结果摘要
        print("\n============= TimescaleDB测试结果 =============")
        print(f"基本连接: {'✓' if connection_status else '✗'} - {connection_msg}")
        print(f"TimescaleDB扩展: {'✓' if extension_status else '✗'} - {extension_msg}")
        print(f"创建超表测试: {'✓' if hypertable_status else '✗'} - {hypertable_msg}")
        print("===============================================\n")
        
        # 综合结果
        if extension_status and hypertable_status:
            print("所有测试通过，TimescaleDB连接配置正确！")
        elif extension_status:
            print("基本功能测试通过，但创建超表测试失败，请检查用户权限。")
        else:
            print("可以连接PostgreSQL，但TimescaleDB扩展未安装或不可用。")
    else:
        print("\n============= TimescaleDB测试结果 =============")
        print(f"基本连接: {'✓' if connection_status else '✗'} - {connection_msg}")
        print("===============================================\n")
        print("无法连接到数据库，请检查连接配置。")

if __name__ == "__main__":
    main() 