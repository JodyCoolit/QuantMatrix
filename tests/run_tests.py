"""
Test runner for the QuantMatrix project.
This script runs all the test modules.
"""

import sys
import os
import time
import logging
from pathlib import Path
import pandas as pd
import argparse

# Add the parent directory to the system path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('test_runner')

# Import test modules
from tests.test_data_sources import DataSourceTester, test_with_retry
from tests.test_timescaledb import TimescaleDBTester
from tests.test_database import DatabaseTester
from tests.test_timescaledb_repository import TimescaleDBRepositoryTester
from tests.test_trading_system import test_ma_crossover_strategy


def run_data_source_tests(tests_to_run=None, retry_interval=60, max_retries=3, use_offline=True):
    """
    Run data source tests.
    
    Args:
        tests_to_run (list): List of tests to run. If None, all tests are run.
        retry_interval (int): Number of seconds to wait between retries for rate-limited APIs
        max_retries (int): Maximum number of retry attempts for rate-limited APIs
        use_offline (bool): If True, uses offline mode to prevent API rate limits
    """
    logger.info("\n===== Running Data Source Tests =====")
    
    if tests_to_run is None:
        # By default, run all tests
        tests_to_run = ["yahoo", "binance", "indicators"]
    
    data_source_tester = DataSourceTester()
    data_source_tester.init_sources(offline_mode=use_offline)
    
    # Get symbols to test from command line arguments or use defaults
    yahoo_symbol = "AAPL"
    binance_symbol = "BTCUSDT"
    
    yahoo_results = None
    binance_results = None
    
    # Run the selected tests
    if "yahoo" in tests_to_run:
        logger.info(f"\nRunning Yahoo Finance tests for {yahoo_symbol}...")
        logger.info(f"Using {'OFFLINE' if use_offline else 'ONLINE'} mode for data retrieval")
        
        yahoo_results = test_with_retry(
            data_source_tester.test_yahoo_finance,
            symbol=yahoo_symbol,
            timeframe="1d",
            max_attempts=max_retries,
            retry_interval=retry_interval,
            switch_to_offline=True
        )
        
        # Log the results
        if yahoo_results and yahoo_results.get("historical") is not None and not yahoo_results.get("historical").empty:
            logger.info(f"Yahoo Finance tests for {yahoo_symbol} completed successfully.")
        else:
            logger.warning(f"Yahoo Finance tests for {yahoo_symbol} failed after {max_retries} attempts.")
    
    if "binance" in tests_to_run:
        logger.info(f"\nRunning Binance tests for {binance_symbol}...")
        binance_results = test_with_retry(
            data_source_tester.test_binance,
            symbol=binance_symbol,
            timeframe="1d",
            max_attempts=max_retries,
            retry_interval=retry_interval,
            switch_to_offline=True
        )
        
        # Log the results
        if binance_results and binance_results.get("historical") is not None and not binance_results.get("historical").empty:
            logger.info(f"Binance tests for {binance_symbol} completed successfully.")
        else:
            logger.warning(f"Binance tests for {binance_symbol} failed after {max_retries} attempts.")
    
    if "indicators" in tests_to_run:
        logger.info("\nRunning indicator tests...")
        
        # Choose the source based on results from earlier tests
        if yahoo_results and yahoo_results.get("historical") is not None and not yahoo_results.get("historical").empty:
            source = "yahoo"
            symbol = yahoo_symbol
            logger.info(f"Using Yahoo Finance data for {symbol} to test indicators")
        elif binance_results and binance_results.get("historical") is not None and not binance_results.get("historical").empty:
            source = "binance"
            symbol = binance_symbol
            logger.info(f"Using Binance data for {symbol} to test indicators")
        else:
            logger.warning("No data available from previous tests. Trying Yahoo Finance for indicators...")
            source = "yahoo"
            symbol = yahoo_symbol
        
        indicators_results = test_with_retry(
            data_source_tester.test_with_indicators,
            source=source,
            symbol=symbol,
            timeframe="1d",
            max_attempts=max_retries,
            retry_interval=retry_interval,
            switch_to_offline=True
        )
        
        # Log the results
        if indicators_results is not None and not (isinstance(indicators_results, pd.DataFrame) and indicators_results.empty):
            logger.info(f"Indicator tests for {symbol} using {source} completed successfully.")
        else:
            logger.warning(f"Indicator tests for {symbol} using {source} failed after {max_retries} attempts.")
    
    logger.info("\n===== Data Source Tests Completed =====")
    
    # Return True if all selected tests were successful
    if "yahoo" in tests_to_run and "binance" in tests_to_run and "indicators" in tests_to_run:
        yahoo_success = yahoo_results and yahoo_results.get("historical") is not None and not yahoo_results.get("historical").empty
        binance_success = binance_results and binance_results.get("historical") is not None and not binance_results.get("historical").empty
        indicators_success = indicators_results is not None and not (isinstance(indicators_results, pd.DataFrame) and indicators_results.empty)
        return yahoo_success and binance_success and indicators_success
    elif "yahoo" in tests_to_run and "binance" in tests_to_run:
        yahoo_success = yahoo_results and yahoo_results.get("historical") is not None and not yahoo_results.get("historical").empty
        binance_success = binance_results and binance_results.get("historical") is not None and not binance_results.get("historical").empty
        return yahoo_success and binance_success
    elif "yahoo" in tests_to_run:
        return yahoo_results and yahoo_results.get("historical") is not None and not yahoo_results.get("historical").empty
    elif "binance" in tests_to_run:
        return binance_results and binance_results.get("historical") is not None and not binance_results.get("historical").empty
    elif "indicators" in tests_to_run:
        return indicators_results is not None and not (isinstance(indicators_results, pd.DataFrame) and indicators_results.empty)
    else:
        return True  # No tests were run, so technically none failed


def run_database_tests(tests_to_run=None):
    """
    Run database-related tests.
    
    Args:
        tests_to_run (list): List of tests to run. If None, all tests are run.
    """
    logger.info("\n===== Running Database Tests =====")
    
    if tests_to_run is None:
        # By default, run all tests
        tests_to_run = ["timescaledb", "database", "repository"]
    
    # Run TimescaleDB connection tests
    if "timescaledb" in tests_to_run:
        logger.info("\nRunning TimescaleDB connection tests...")
        timescaledb_tester = TimescaleDBTester()
        timescaledb_result = timescaledb_tester.run_all_tests()
        logger.info(f"TimescaleDB connection tests {'passed' if timescaledb_result else 'failed'}.")
    
    # Run general database tests
    if "database" in tests_to_run and "timescaledb" in tests_to_run:
        logger.info("\nRunning general database tests...")
        database_tester = DatabaseTester(db_type="timescaledb")  # Can also test with timescaledb
        database_result = database_tester.run_all_tests()
        logger.info(f"General database tests {'passed' if database_result else 'failed'}.")
    
    # Run TimescaleDB repository tests
    if "repository" in tests_to_run and "timescaledb" in tests_to_run:
        logger.info("\nRunning TimescaleDB repository tests...")
        repository_tester = TimescaleDBRepositoryTester()
        repository_result = repository_tester.run_all_tests()
        logger.info(f"TimescaleDB repository tests {'passed' if repository_result else 'failed'}.")
    
    logger.info("\n===== Database Tests Completed =====")


def run_trading_system_tests(tests_to_run=None):
    """
    Run trading system tests.
    
    Args:
        tests_to_run (list): List of tests to run. If None, all tests are run.
    """
    logger.info("\n===== Running Trading System Tests =====")
    
    if tests_to_run is None:
        # By default, run all tests
        tests_to_run = ["ma_crossover"]
    
    # Run MA Crossover Strategy test
    if "ma_crossover" in tests_to_run:
        logger.info("\nRunning Moving Average Crossover Strategy test...")
        try:
            test_ma_crossover_strategy()
            logger.info("Moving Average Crossover Strategy test completed successfully.")
        except Exception as e:
            logger.error(f"Moving Average Crossover Strategy test failed: {e}")
    
    logger.info("\n===== Trading System Tests Completed =====")


def run_all_tests(data_source_tests=None, database_tests=None, trading_system_tests=None, use_offline=True):
    """
    Run all tests for the QuantMatrix project.
    
    Args:
        data_source_tests (list): List of data source tests to run.
        database_tests (list): List of database tests to run.
        trading_system_tests (list): List of trading system tests to run.
        use_offline (bool): Whether to use offline mode for data source tests
    """
    logger.info("===== Running QuantMatrix Tests =====")
    
    # Run data source tests
    if data_source_tests is not None:
        run_data_source_tests(tests_to_run=data_source_tests, use_offline=use_offline)
    
    # Run database tests
    if database_tests is not None:
        run_database_tests(tests_to_run=database_tests)
    
    # Run trading system tests
    if trading_system_tests is not None:
        run_trading_system_tests(tests_to_run=trading_system_tests)
    
    logger.info("\n===== All Tests Completed =====")


def main():
    """Main entry point for running tests"""
    parser = argparse.ArgumentParser(description="Run tests for QuantMatrix project")
    parser.add_argument("--select", action="store_true", help="Select test categories to run")
    parser.add_argument("--offline", action="store_true", help="Run data source tests in offline mode", default=True)
    parser.add_argument("--online", action="store_false", dest="offline", help="Run data source tests in online mode (may hit API limits)")
    
    args = parser.parse_args()
    
    # Check if specific tests are requested
    if args.select:
        logger.info("Select test categories to run:")
        logger.info("1. Data Source Tests")
        logger.info("2. Database Tests")
        logger.info("3. Trading System Tests")
        
        selection = input("Enter test categories to run (e.g., '1,2'): ").strip()
        categories = [item.strip() for item in selection.split(",")]
        
        data_source_tests = None
        database_tests = None
        trading_system_tests = None
        
        # If data source tests are selected
        if "1" in categories:
            logger.info("\nSelect data source tests to run:")
            logger.info("1. yahoo - Test Yahoo Finance data source")
            logger.info("2. binance - Test Binance data source")
            logger.info("3. indicators - Test technical indicators")
            
            ds_selection = input("Enter data source tests to run (e.g., 'binance,indicators' or 'all'): ").strip()
            
            if ds_selection.lower() == "all":
                data_source_tests = ["yahoo", "binance", "indicators"]
            else:
                # Map numeric selections to test names
                selection_map = {
                    "1": "yahoo",
                    "2": "binance", 
                    "3": "indicators",
                    "yahoo": "yahoo",
                    "binance": "binance",
                    "indicators": "indicators"
                }
                
                # Handle both numeric and text inputs
                data_source_tests = []
                for item in ds_selection.split(","):
                    item = item.strip()
                    if item in selection_map:
                        data_source_tests.append(selection_map[item])
        
        # If database tests are selected
        if "2" in categories:
            logger.info("\nSelect database tests to run:")
            logger.info("1. timescaledb - Test TimescaleDB connection")
            logger.info("2. database - Test general database operations")
            logger.info("3. repository - Test TimescaleDB repository")
            
            db_selection = input("Enter database tests to run (e.g., 'timescaledb,repository' or 'all'): ").strip()
            
            if db_selection.lower() == "all":
                database_tests = ["timescaledb", "database", "repository"]
            else:
                # Map numeric selections to test names
                selection_map = {
                    "1": "timescaledb",
                    "2": "database", 
                    "3": "repository",
                    "timescaledb": "timescaledb",
                    "database": "database",
                    "repository": "repository"
                }
                
                # Handle both numeric and text inputs
                database_tests = []
                for item in db_selection.split(","):
                    item = item.strip()
                    if item in selection_map:
                        database_tests.append(selection_map[item])
        
        # If trading system tests are selected
        if "3" in categories:
            logger.info("\nSelect trading system tests to run:")
            logger.info("1. ma_crossover - Test Moving Average Crossover Strategy")
            
            ts_selection = input("Enter trading system tests to run (e.g., 'ma_crossover' or 'all'): ").strip()
            
            if ts_selection.lower() == "all":
                trading_system_tests = ["ma_crossover"]
            else:
                # Map numeric selections to test names
                selection_map = {
                    "1": "ma_crossover",
                    "ma_crossover": "ma_crossover"
                }
                
                # Handle both numeric and text inputs
                trading_system_tests = []
                for item in ts_selection.split(","):
                    item = item.strip()
                    if item in selection_map:
                        trading_system_tests.append(selection_map[item])
        
        # Log selections
        if data_source_tests:
            logger.info(f"Running data source tests: {', '.join(data_source_tests)}")
            logger.info(f"Data source tests will run in {'OFFLINE' if args.offline else 'ONLINE'} mode")
            
        if database_tests:
            logger.info(f"Running database tests: {', '.join(database_tests)}")
        if trading_system_tests:
            logger.info(f"Running trading system tests: {', '.join(trading_system_tests)}")
        
        if not data_source_tests and not database_tests and not trading_system_tests:
            logger.info("No valid tests selected. Exiting.")
            sys.exit(0)
        
        # Run all selected tests
        run_all_tests(
            data_source_tests=data_source_tests, 
            database_tests=database_tests,
            trading_system_tests=trading_system_tests,
            use_offline=args.offline
        )
    else:
        # Ask if user wants to run all tests
        logger.info("Run all tests? (y/n): ")
        run_all = input().strip().lower() == 'y'
        
        if run_all:
            logger.info(f"Data source tests will run in {'OFFLINE' if args.offline else 'ONLINE'} mode")
            # Run all tests
            run_data_source_tests(use_offline=args.offline)
            run_database_tests()
            run_trading_system_tests()
        else:
            logger.info("Run with --select to choose specific tests. Exiting.")


if __name__ == "__main__":
    main() 