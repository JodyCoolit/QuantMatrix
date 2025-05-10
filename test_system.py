#!/usr/bin/env python
"""
Simple test script to verify the QuantMatrix trading system.
This will run a backtest using the moving average crossover strategy on AAPL data.
"""
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Import data components
from src.data.data_manager import DataManager
from src.data.data_processor import DataProcessor
from src.data.yahoo_finance_source import YahooFinanceSource

# Import strategy components
from src.strategy.strategy_manager import StrategyManager
from src.strategy.moving_average_crossover_strategy import MovingAverageCrossoverStrategy

# Import monitoring components
from src.monitoring.logger import Logger


def test_ma_crossover_strategy():
    """Test the moving average crossover strategy."""
    # Set up components
    logger = Logger()
    logger.log_info("Starting MA Crossover Strategy Test")
    
    # Set up data manager
    data_processor = DataProcessor()
    data_manager = DataManager(data_processor=data_processor)
    data_manager.add_data_source('yahoo', YahooFinanceSource())
    
    # Set up strategy
    strategy_manager = StrategyManager()
    ma_strategy = MovingAverageCrossoverStrategy(
        name="MA Crossover", 
        short_window=20, 
        long_window=50
    )
    strategy_manager.add_strategy(ma_strategy)
    
    # Define test parameters
    symbol = "AAPL"
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    # Fetch data
    logger.log_info(f"Fetching data for {symbol}")
    data = data_manager.get_historical_data(
        source='yahoo',
        symbol=symbol,
        timeframe='1d',
        start_date=start_date,
        end_date=end_date
    )
    
    # Skip if no data
    if data.empty:
        logger.log_error(f"No data available for {symbol}")
        return
    
    # Add symbol column if not present
    if 'symbol' not in data.columns:
        data['symbol'] = symbol
    
    # Run backtest
    logger.log_info("Running backtest")
    result = strategy_manager.run_backtest(ma_strategy, data)
    
    # Print results
    logger.log_info(f"Backtest result: {result}")
    logger.log_info(f"Total return: {result.total_return:.2%}")
    logger.log_info(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
    logger.log_info(f"Max drawdown: {result.max_drawdown:.2%}")
    
    # Plot results
    if hasattr(result, 'equity_curve') and not result.equity_curve.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(result.equity_curve.index, result.equity_curve['equity'])
        plt.title(f"{symbol} - MA Crossover Strategy Performance")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{symbol}_ma_crossover_performance.png")
        logger.log_info(f"Performance chart saved to {symbol}_ma_crossover_performance.png")
    
    logger.log_info("MA Crossover Strategy Test Complete")


if __name__ == "__main__":
    test_ma_crossover_strategy() 