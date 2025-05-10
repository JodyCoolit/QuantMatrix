# QuantMatrix Analysis Module

The Analysis module provides comprehensive tools for analyzing trading strategies, market data, and portfolio performance. This module bridges the gap between strategy development, backtesting, and optimization.

## Components

### 1. Performance Analysis

The `performance.py` module provides tools for detailed performance analysis of trading strategies:

- **PerformanceMetrics**: Calculates a wide range of performance metrics including returns, volatility, Sharpe ratio, Sortino ratio, maximum drawdown, win rate, profit factor, and more.

- **PerformanceAnalyzer**: Analyzes backtest results, compares multiple strategies, and generates performance reports.

Example usage:

```python
from src.analysis.performance import PerformanceAnalyzer

# Analyze a backtest result
analyzer = PerformanceAnalyzer()
metrics = analyzer.analyze_backtest(backtest_result)

# Access performance metrics
print(f"Total Return: {metrics.total_return:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
```

### 2. Statistical Analysis

The `statistics.py` module offers statistical tools for analyzing market data and strategy performance:

- **MarketStatistics**: Provides methods for descriptive statistics, normality tests, stationarity tests, autocorrelation tests, and volatility estimation.

- **StrategyStatistics**: Calculates drawdowns, analyzes returns distribution, trade durations, and winning/losing streaks.

Example usage:

```python
from src.analysis.statistics import MarketStatistics

# Test if returns follow a normal distribution
market_stats = MarketStatistics()
normality_test = market_stats.test_normality(returns)
print(f"Is normal: {normality_test['jarque_bera_normal']}")

# Calculate the Hurst exponent
hurst = market_stats.calculate_hurst_exponent(price_series)
print(f"Hurst exponent: {hurst}")
```

### 3. Visualization

The `visualization.py` module provides visualization tools for market data and performance analysis:

- **PerformanceVisualization**: Creates equity curves, drawdown charts, monthly returns heatmaps, rolling metrics plots, and return distribution charts.

- **MarketVisualization**: Generates price charts, correlation matrices, and interactive candlestick charts.

Example usage:

```python
from src.analysis.visualization import PerformanceVisualization

# Create an equity curve with drawdowns highlighted
perf_vis = PerformanceVisualization()
fig = perf_vis.plot_equity_curve(
    equity_curve=backtest_result.equity_curve,
    benchmark_data=benchmark_data,
    show_drawdowns=True
)
fig.savefig("equity_curve.png")
```

### 4. Correlation Analysis

The `correlation.py` module provides tools for analyzing relationships between assets:

- **CorrelationAnalysis**: Calculates correlation matrices, rolling correlations, finds highly/lowly correlated pairs, tests correlation significance, and creates correlation networks.

- **CovarianceAnalysis**: Computes covariance matrices, rolling covariance, and EWMA covariance.

- **PrincipalComponentAnalysis**: Performs PCA to identify factors driving market returns.

Example usage:

```python
from src.analysis.correlation import CorrelationAnalysis

# Calculate correlation matrix
corr_analyzer = CorrelationAnalysis()
corr_matrix = corr_analyzer.calculate_correlation_matrix(returns_df)

# Find highly correlated pairs
high_corr_pairs = corr_analyzer.find_highest_correlations(corr_matrix, threshold=0.7)
```

### 5. Strategy Optimization

The `optimization.py` module enables parameter optimization and portfolio construction:

- **StrategyOptimizer**: Performs grid search, plots parameter surfaces/heatmaps, and conducts walk-forward optimization.

- **PortfolioOptimization**: Finds minimum variance portfolios, maximum Sharpe ratio portfolios, calculates efficient frontiers, and plots portfolio visualizations.

Example usage:

```python
from src.analysis.optimization import StrategyOptimizer

# Perform grid search
optimizer = StrategyOptimizer(strategy_manager)
results = optimizer.grid_search(
    strategy_class=MovingAverageCrossoverStrategy,
    data=market_data,
    parameters={
        'short_window': [5, 10, 20, 30],
        'long_window': [50, 100, 200]
    },
    metric='sharpe_ratio'
)

# Get best parameters
best_params = results['best_parameters']
```

## Integration with Other Modules

The Analysis module integrates with other QuantMatrix modules:

- Works with the **Strategy** module to optimize strategy parameters
- Processes backtest results from the **Strategy Manager**
- Analyzes trades recorded by the **Execution** module
- Provides visualizations for the **Monitoring** module's dashboard

## Command Line Usage

Run analysis as a standalone operation:

```bash
python main.py --mode analyze --symbols AAPL MSFT GOOG
```

This will:
1. Analyze each symbol individually
2. Perform correlation analysis between symbols
3. Run portfolio optimization if there are 3+ symbols
4. Generate visualizations and reports 