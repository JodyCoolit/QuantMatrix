#!/usr/bin/env python
import os
import json
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import data components
from src.data.data_manager import DataManager
from src.data.data_processor import DataProcessor
from src.data.yahoo_finance_source import YahooFinanceSource
from src.data.binance_source import BinanceSource
from src.data.time_series_database import TimeSeriesDatabase

# Import strategy components
from src.strategy.strategy_manager import StrategyManager
from src.strategy.moving_average_crossover_strategy import MovingAverageCrossoverStrategy

# Import execution components
from src.execution.order_manager import OrderManager
from src.execution.risk_controller import RiskController
from src.execution.alpaca_broker import AlpacaBroker

# Import monitoring components
from src.monitoring.logger import Logger
from src.monitoring.alert_system import AlertSystem
from src.monitoring.dashboard import Dashboard

# Import analysis components
from src.analysis.performance import PerformanceAnalyzer, PerformanceMetrics
from src.analysis.statistics import MarketStatistics, StrategyStatistics
from src.analysis.visualization import PerformanceVisualization, MarketVisualization
from src.analysis.correlation import CorrelationAnalysis, CovarianceAnalysis
from src.analysis.optimization import StrategyOptimizer, PortfolioOptimization


def load_config(config_path):
    """Load a JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_data_sources(data_config, data_manager):
    """Set up data sources from configuration."""
    for name, config in data_config.items():
        if not config.get('enabled', False):
            continue
        
        source_type = config.get('type', '').lower()
        
        if source_type == 'yahoo':
            source = YahooFinanceSource()
            data_manager.add_data_source(name, source)
        elif source_type == 'binance':
            api_key = config.get('api_key', '')
            api_secret = config.get('api_secret', '')
            source = BinanceSource(api_key=api_key, api_secret=api_secret)
            data_manager.add_data_source(name, source)
        # Add more data sources as needed
    
    return data_manager


def setup_strategies(strategy_config, strategy_manager):
    """Set up strategies from configuration."""
    for name, config in strategy_config.items():
        if not config.get('enabled', False):
            continue
        
        strategy_type = config.get('type', '').lower()
        
        if strategy_type == 'moving_average_crossover':
            params = config.get('parameters', {})
            short_window = params.get('short_window', 20)
            long_window = params.get('long_window', 50)
            
            strategy = MovingAverageCrossoverStrategy(
                name=name, 
                short_window=short_window, 
                long_window=long_window
            )
            
            strategy_manager.add_strategy(strategy)
        # Add more strategy types as needed
    
    return strategy_manager


def setup_risk_controller(risk_config):
    """Set up risk controller from configuration."""
    max_position_size = risk_config.get('max_position_size', 0.1)
    max_drawdown = risk_config.get('max_drawdown', 0.05)
    position_limits = risk_config.get('position_limits', {})
    
    risk_controller = RiskController(
        position_limits=position_limits,
        max_position_size=max_position_size, 
        max_drawdown=max_drawdown
    )
    
    return risk_controller


def generate_sample_data(symbols, start_date=None, end_date=None):
    """
    Generate sample market data for demonstration.
    
    Args:
        symbols: List of symbols to generate data for
        start_date: Start date (defaults to 1 year ago)
        end_date: End date (defaults to today)
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    print(f"Generating sample data for {symbols}...")
    
    all_data = {}
    
    # Create date range - use only business days for more realistic data
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    for symbol in symbols:
        print(f"Generating data for {symbol}...")
        
        # Use symbol as seed for reproducibility
        np.random.seed(hash(symbol) % 10000)
        
        n_days = len(date_range)
        
        # Generate daily returns with realistic parameters
        daily_returns = np.random.normal(0.0005, 0.01, n_days)
        daily_returns = np.clip(daily_returns, -0.05, 0.05)  # Remove extreme values
        
        # Convert to price series (starting at $100)
        close_prices = 100 * (1 + daily_returns).cumprod()
        
        # Generate OHLC data
        high = close_prices * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
        low = close_prices * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
        open_prices = low + np.random.uniform(0, 1, n_days) * (high - low)
        
        # Generate trading volume
        volume = np.random.randint(100000, 1000000, n_days)
        
        # Create DataFrame with OHLCV data
        df = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close_prices,
            'volume': volume,
            'symbol': symbol
        }, index=date_range)
        
        # Make sure there are no NaN or inf values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        all_data[symbol] = df
        print(f"Generated {len(df)} rows for {symbol}")
    
    return all_data


def run_backtest_with_sample_data(strategy_manager, symbols, start_date=None, end_date=None):
    """Run a backtest using generated sample data."""
    if not start_date:
        start_date = datetime.now() - timedelta(days=365)
    if not end_date:
        end_date = datetime.now()
    
    results = {}
    
    # Generate sample data for all symbols
    all_data = generate_sample_data(symbols, start_date, end_date)
    
    for symbol, data in all_data.items():
        # Skip if no data
        if data.empty:
            print(f"No data available for {symbol}")
            continue
        
        # Run backtest for each strategy
        for strategy_name, strategy in strategy_manager.strategies.items():
            result = strategy_manager.run_backtest(strategy, data)
            
            # Store the result
            if symbol not in results:
                results[symbol] = {}
            results[symbol][strategy_name] = result
            
            print(f"Backtest result for {symbol} using {strategy_name}: {result}")
    
    return results


def analyze_backtest_results(results, logger):
    """Analyze backtest results using the analysis modules."""
    logger.log_info("Analyzing backtest results")
    
    # Initialize analyzers
    performance_analyzer = PerformanceAnalyzer()
    market_viz = MarketVisualization()
    performance_viz = PerformanceVisualization()
    
    analysis_results = {}
    
    for symbol, strategy_results in results.items():
        analysis_results[symbol] = {}
        
        for strategy_name, result in strategy_results.items():
            # Skip if no equity curve
            if not hasattr(result, 'equity_curve') or result.equity_curve.empty:
                logger.log_warning(f"No equity curve for {symbol} with {strategy_name}")
                continue
            
            # Calculate detailed performance metrics
            metrics = performance_analyzer.analyze_backtest(result)
            
            # Store analysis results
            analysis_results[symbol][strategy_name] = {
                'metrics': metrics,
                'equity_curve': result.equity_curve,
                'trades': result.trades
            }
            
            # Log key metrics
            logger.log_info(f"Analysis for {symbol} with {strategy_name}:")
            logger.log_info(f"  Total Return: {metrics.total_return:.2%}")
            logger.log_info(f"  Annualized Return: {metrics.annualized_return:.2%}")
            logger.log_info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            logger.log_info(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
            logger.log_info(f"  Win Rate: {metrics.win_rate:.2%}")
            
            # Generate and save visualization
            if 'returns' in result.equity_curve.columns:
                returns = result.equity_curve['returns']
                
                # Create monthly returns heatmap
                fig = performance_viz.plot_monthly_returns_heatmap(returns)
                fig.savefig(f"{symbol}_{strategy_name}_monthly_returns.png")
                
                # Create returns distribution
                fig = performance_viz.plot_returns_distribution(returns)
                fig.savefig(f"{symbol}_{strategy_name}_returns_dist.png")
            
            # Create equity curve with drawdowns
            fig = performance_viz.plot_equity_curve(
                result.equity_curve, 
                benchmark_data=None,
                show_drawdowns=True
            )
            fig.savefig(f"{symbol}_{strategy_name}_equity_curve.png")
    
    return analysis_results


def run_backtest(data_manager, strategy_manager, symbols, source='yahoo_finance', 
               timeframe='1d', start_date=None, end_date=None):
    """Run a backtest for all strategies."""
    if not start_date:
        start_date = datetime.now() - timedelta(days=365)
    if not end_date:
        end_date = datetime.now()
    
    results = {}
    
    for symbol in symbols:
        # Fetch data
        data = data_manager.get_historical_data(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Skip if no data
        if data.empty:
            print(f"No data available for {symbol}")
            continue
        
        # Add symbol column if not present
        if 'symbol' not in data.columns:
            data['symbol'] = symbol
        
        # Run backtest for each strategy
        for strategy_name, strategy in strategy_manager.strategies.items():
            result = strategy_manager.run_backtest(strategy, data)
            
            # Store the result
            if symbol not in results:
                results[symbol] = {}
            results[symbol][strategy_name] = result
            
            print(f"Backtest result for {symbol} using {strategy_name}: {result}")
    
    return results


def main():
    """Main application entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='QuantMatrix Trading System')
    parser.add_argument('--mode', choices=['backtest', 'live', 'analyze'], default='backtest',
                      help='Operation mode (backtest, live, or analyze)')
    parser.add_argument('--config-dir', default='config',
                      help='Configuration directory')
    parser.add_argument('--symbols', nargs='+',
                      help='Symbols to trade (overrides config)')
    parser.add_argument('--use-sample-data', action='store_true',
                      help='Use generated sample data instead of fetching from data sources')
    args = parser.parse_args()
    
    # Load configurations
    data_config = load_config(os.path.join(args.config_dir, 'data_sources.json'))
    strategy_config = load_config(os.path.join(args.config_dir, 'strategies.json'))
    risk_config = load_config(os.path.join(args.config_dir, 'risk.json'))
    
    # Set up logger
    logger = Logger()
    logger.log_info("Starting QuantMatrix Trading System")
    
    # Set up components
    data_processor = DataProcessor()
    db = TimeSeriesDatabase()
    data_manager = DataManager(data_processor=data_processor)
    strategy_manager = StrategyManager()
    risk_controller = setup_risk_controller(risk_config)
    alert_system = AlertSystem(logger=logger)
    dashboard = Dashboard()
    
    # Set up data sources
    data_manager = setup_data_sources(data_config, data_manager)
    
    # Set up strategies
    strategy_manager = setup_strategies(strategy_config, strategy_manager)
    
    # Get symbols to process
    symbols = args.symbols
    if not symbols:
        # Use symbols from first enabled data source
        for source_name, source_config in data_config.items():
            if source_config.get('enabled', False):
                symbols = source_config.get('symbols', [])
                break
    
    if not symbols:
        logger.log_error("No symbols specified")
        return
    
    # Run in specified mode
    if args.mode == 'backtest':
        logger.log_info(f"Running in backtest mode for symbols: {symbols}")
        
        # Run backtest
        if args.use_sample_data:
            logger.log_info("Using generated sample data for backtest")
            results = run_backtest_with_sample_data(
                strategy_manager=strategy_manager,
                symbols=symbols
            )
        else:
            results = run_backtest(
                data_manager=data_manager,
                strategy_manager=strategy_manager,
                symbols=symbols
            )
        
        # Analyze results
        analysis_results = analyze_backtest_results(results, logger)
        
        # Display results on dashboard
        for symbol, strategy_results in results.items():
            for strategy_name, result in strategy_results.items():
                if hasattr(result, 'equity_curve') and not result.equity_curve.empty:
                    dashboard.display_performance_chart(result.equity_curve)
                    dashboard.update_metrics({
                        'total_return': result.total_return,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'win_rate': result.win_rate
                    })
        
        dashboard.display_metrics_summary()
        dashboard.show()
    
    elif args.mode == 'analyze':
        logger.log_info(f"Running in analysis mode for symbols: {symbols}")
        
        # Initialize additional analysis components
        performance_analyzer = PerformanceAnalyzer()
        market_stats = MarketStatistics()
        strategy_stats = StrategyStatistics()
        correlation_analyzer = CorrelationAnalysis()
        
        # First, get data for all symbols
        symbol_data = {}
        returns_data = {}
        
        if args.use_sample_data:
            logger.log_info("Using generated sample data for analysis")
            # Generate sample data
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()
            
            # Get data
            symbol_data = generate_sample_data(symbols, start_date, end_date)
            
            # Calculate returns for each symbol
            for symbol, data in symbol_data.items():
                returns_data[symbol] = data['close'].pct_change().dropna()
        else:
            for symbol in symbols:
                data = data_manager.get_historical_data(
                    source='yahoo_finance',
                    symbol=symbol,
                    timeframe='1d',
                    start_date=datetime.now() - timedelta(days=365),
                    end_date=datetime.now()
                )
                
                if not data.empty:
                    symbol_data[symbol] = data
                    # Calculate returns for correlation analysis
                    returns_data[symbol] = data['close'].pct_change().dropna()
        
        # Correlation analysis for symbols
        if len(symbol_data) > 1:
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate correlation matrix
            corr_matrix = correlation_analyzer.calculate_correlation_matrix(returns_df)
            
            # Find highly correlated and lowly correlated pairs
            high_corr = correlation_analyzer.find_highest_correlations(corr_matrix, threshold=0.7)
            low_corr = correlation_analyzer.find_lowest_correlations(corr_matrix, threshold=0.3)
            
            # Print correlation analysis
            logger.log_info("Correlation Analysis:")
            if not high_corr.empty:
                logger.log_info("Highly correlated pairs:")
                for _, row in high_corr.iterrows():
                    logger.log_info(f"  {row['asset1']} - {row['asset2']}: {row['correlation']:.2f}")
            
            if not low_corr.empty:
                logger.log_info("Lowly correlated pairs (good for diversification):")
                for _, row in low_corr.iterrows():
                    logger.log_info(f"  {row['asset1']} - {row['asset2']}: {row['correlation']:.2f}")
            
            # Create correlation matrix visualization
            market_viz = MarketVisualization()
            fig = market_viz.plot_correlation_matrix(returns_df)
            fig.savefig("correlation_matrix.png")
            logger.log_info("Correlation matrix saved as correlation_matrix.png")
            
            # Portfolio optimization
            if len(symbols) >= 3:
                try:
                    logger.log_info("Performing portfolio optimization...")
                    
                    # Find minimum variance portfolio
                    min_var_portfolio = PortfolioOptimization.minimum_variance_portfolio(returns_df)
                    
                    # Find maximum Sharpe ratio portfolio
                    max_sharpe_portfolio = PortfolioOptimization.maximum_sharpe_portfolio(returns_df)
                    
                    # Log portfolio optimization results
                    logger.log_info("Minimum Variance Portfolio:")
                    for asset, weight in min_var_portfolio['weights'].items():
                        logger.log_info(f"  {asset}: {weight:.2%}")
                    logger.log_info(f"  Expected Return: {min_var_portfolio['return']:.2%}")
                    logger.log_info(f"  Expected Volatility: {min_var_portfolio['volatility']:.2%}")
                    
                    logger.log_info("Maximum Sharpe Ratio Portfolio:")
                    for asset, weight in max_sharpe_portfolio['weights'].items():
                        logger.log_info(f"  {asset}: {weight:.2%}")
                    logger.log_info(f"  Expected Return: {max_sharpe_portfolio['return']:.2%}")
                    logger.log_info(f"  Expected Volatility: {max_sharpe_portfolio['volatility']:.2%}")
                    logger.log_info(f"  Sharpe Ratio: {max_sharpe_portfolio['sharpe_ratio']:.2f}")
                    
                    # Calculate and plot efficient frontier
                    efficient_portfolios = PortfolioOptimization.efficient_frontier(returns_df, points=20)
                    fig = PortfolioOptimization.plot_efficient_frontier(
                        efficient_portfolios=efficient_portfolios,
                        asset_returns=returns_df,
                        risk_free_rate=0.01  # 1% risk-free rate
                    )
                    fig.savefig("efficient_frontier.png")
                    logger.log_info("Efficient frontier saved as efficient_frontier.png")
                except Exception as e:
                    logger.log_error(f"Portfolio optimization failed: {e}")
        
        # Individual asset analysis
        for symbol, data in symbol_data.items():
            logger.log_info(f"Analyzing {symbol}...")
            
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            try:
                # Descriptive statistics
                stats = market_stats.calculate_descriptive_stats(returns)
                logger.log_info(f"Descriptive statistics for {symbol}:")
                logger.log_info(f"  Mean: {stats['Mean']:.4f}")
                logger.log_info(f"  Std Dev: {stats['Std Dev']:.4f}")
                logger.log_info(f"  Skewness: {stats['Skewness']:.4f}")
                logger.log_info(f"  Kurtosis: {stats['Kurtosis']:.4f}")
                
                # Test normality
                normality = market_stats.test_normality(returns)
                logger.log_info(f"Normality test for {symbol}:")
                logger.log_info(f"  Normal distribution: {'Yes' if normality['jarque_bera_normal'] else 'No'}")
                
                # Test stationarity
                stationarity = market_stats.test_stationarity(returns)
                logger.log_info(f"Stationarity test for {symbol}:")
                logger.log_info(f"  Stationary: {'Yes' if stationarity['adf_stationary'] else 'No'}")
                
                # Create price chart
                fig = market_viz.plot_price_chart(data, volume=True)
                fig.savefig(f"{symbol}_price_chart.png")
                logger.log_info(f"Price chart saved as {symbol}_price_chart.png")
            except Exception as e:
                logger.log_error(f"Analysis failed for {symbol}: {e}")
    
    elif args.mode == 'live':
        logger.log_info(f"Running in live mode for symbols: {symbols}")
        
        # Set up broker
        try:
            broker = AlpacaBroker()
            order_manager = OrderManager(broker=broker, risk_controller=risk_controller)
            
            # Get account info
            account = broker.get_account_info()
            logger.log_info(f"Connected to broker. Account value: ${account.equity}")
            
            # In a real system, this would run continuously
            # For now, we'll just fetch some data and show it on the dashboard
            positions = broker.get_positions()
            dashboard.display_positions_table(positions)
            dashboard.update_metrics({
                'account_value': account.equity,
                'cash_balance': account.cash,
                'position_count': len(positions)
            })
            dashboard.display_metrics_summary()
            dashboard.show()
            
        except Exception as e:
            logger.log_error("Error connecting to broker", exception=e)
    
    logger.log_info("QuantMatrix Trading System shutdown complete")


if __name__ == "__main__":
    main() 