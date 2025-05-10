#!/usr/bin/env python
"""
Example script demonstrating the analysis module capabilities.
This script analyzes historical data for multiple stocks, performs correlation analysis,
optimizes a portfolio, and generates visualizations.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import time
import numpy as np

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import analysis components
from src.analysis.performance import PerformanceAnalyzer, PerformanceMetrics
from src.analysis.statistics import MarketStatistics, StrategyStatistics
from src.analysis.visualization import PerformanceVisualization, MarketVisualization
from src.analysis.correlation import CorrelationAnalysis, CovarianceAnalysis, PrincipalComponentAnalysis
from src.analysis.optimization import PortfolioOptimization


def download_data(symbols, start_date, end_date):
    """Download historical data for the given symbols."""
    print(f"Downloading data for {symbols}...")
    data = {}
    
    for symbol in symbols:
        try:
            print(f"Downloading {symbol}...")
            ticker_data = yf.download(symbol, start=start_date, end=end_date)
            
            # Add a small delay to avoid rate limiting
            time.sleep(2)
            
            if not ticker_data.empty:
                # Rename columns to lowercase for consistency
                ticker_data.columns = [col.lower() for col in ticker_data.columns]
                data[symbol] = ticker_data
                print(f"Downloaded {len(ticker_data)} rows for {symbol}")
            else:
                print(f"No data found for {symbol}")
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
    
    return data


def generate_sample_data(symbols, start_date, end_date):
    """Generate sample market data for demonstration."""
    print(f"Generating sample data for {symbols}...")
    data = {}
    
    # Convert date strings to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Create date range - use only business days for more realistic data
    date_range = pd.date_range(start=start, end=end, freq='B')
    
    for symbol in symbols:
        print(f"Generating data for {symbol}...")
        
        # Generate random price data with a slight upward trend
        np.random.seed(hash(symbol) % 10000)  # Use symbol name as seed for reproducibility
        
        n_days = len(date_range)
        
        # Generate more stable daily returns
        daily_returns = np.random.normal(0.0005, 0.01, n_days)  # Lower volatility
        
        # Remove any extreme outliers
        daily_returns = np.clip(daily_returns, -0.05, 0.05)
        
        # Convert to price series
        prices = 100 * (1 + daily_returns).cumprod()  # Start around $100
        
        # Generate OHLC data
        high = prices * (1 + np.abs(np.random.normal(0, 0.005, n_days)))  # Smaller range
        low = prices * (1 - np.abs(np.random.normal(0, 0.005, n_days)))   # Smaller range
        open_prices = low + np.random.uniform(0, 1, n_days) * (high - low)
        close_prices = prices
        
        # Generate volume
        volume = np.random.randint(100000, 1000000, n_days)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close_prices,
            'volume': volume
        }, index=date_range)
        
        # Make sure there are no NaN or inf values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        data[symbol] = df
        print(f"Generated {len(df)} rows for {symbol}")
    
    return data


def run_market_analysis(symbol_data):
    """Run statistical analysis on market data."""
    print("\n=== Market Statistics Analysis ===")
    
    market_stats = MarketStatistics()
    market_viz = MarketVisualization()
    
    # Analyze each symbol individually
    for symbol, data in symbol_data.items():
        print(f"\nAnalyzing {symbol}...")
        
        # Calculate returns - ensure column names are consistent
        # Use lower case column names
        data['returns'] = data['close'].pct_change().dropna()
        returns = data['returns'].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Get descriptive statistics
        stats = market_stats.calculate_descriptive_stats(returns)
        print("Descriptive statistics:")
        print(f"  Mean: {stats['Mean']:.6f}")
        print(f"  Std Dev: {stats['Std Dev']:.6f}")
        print(f"  Skewness: {stats['Skewness']:.4f}")
        print(f"  Kurtosis: {stats['Kurtosis']:.4f}")
        print(f"  Positive Days: {stats['Positive Days']:.2%}")
        print(f"  Negative Days: {stats['Negative Days']:.2%}")
        
        # Test normality - handle exceptions
        try:
            normality = market_stats.test_normality(returns)
            print("\nNormality tests:")
            print(f"  Jarque-Bera test: {'Normal' if normality['jarque_bera_normal'] else 'Not normal'} (p-value: {normality['jarque_bera_pvalue']:.4f})")
            print(f"  Shapiro-Wilk test: {'Normal' if normality['shapiro_wilk_normal'] else 'Not normal'} (p-value: {normality['shapiro_wilk_pvalue']:.4f})")
        except Exception as e:
            print(f"\nNormality tests failed: {e}")
        
        # Test stationarity - handle exceptions
        try:
            stationarity = market_stats.test_stationarity(returns)
            print("\nStationarity tests:")
            print(f"  ADF test: {'Stationary' if stationarity['adf_stationary'] else 'Not stationary'} (p-value: {stationarity['adf_pvalue']:.4f})")
            print(f"  KPSS test: {'Stationary' if stationarity['kpss_stationary'] else 'Not stationary'} (p-value: {stationarity['kpss_pvalue']:.4f})")
        except Exception as e:
            print(f"\nStationarity tests failed: {e}")
        
        # Calculate Hurst exponent - handle exceptions
        try:
            hurst = market_stats.calculate_hurst_exponent(data['close'])
            print(f"\nHurst exponent: {hurst:.4f}")
            if hurst > 0.5:
                print("  Market shows trend-following behavior (Hurst > 0.5)")
            elif hurst < 0.5:
                print("  Market shows mean-reverting behavior (Hurst < 0.5)")
            else:
                print("  Market follows a random walk (Hurst = 0.5)")
        except Exception as e:
            print(f"\nHurst exponent calculation failed: {e}")
        
        # Create price chart
        try:
            fig = market_viz.plot_price_chart(data)
            fig.savefig(f"examples/output/{symbol}_price_chart.png")
            print(f"Price chart saved as examples/output/{symbol}_price_chart.png")
        except Exception as e:
            print(f"Price chart creation failed: {e}")
        
        # Create returns distribution
        try:
            fig = PerformanceVisualization.plot_returns_distribution(returns)
            fig.savefig(f"examples/output/{symbol}_returns_dist.png")
            print(f"Returns distribution saved as examples/output/{symbol}_returns_dist.png")
        except Exception as e:
            print(f"Returns distribution chart creation failed: {e}")


def run_correlation_analysis(symbol_data):
    """Run correlation analysis on the symbols."""
    print("\n=== Correlation Analysis ===")
    
    try:
        # Create returns DataFrame
        returns_data = {}
        for symbol, data in symbol_data.items():
            returns_data[symbol] = data['close'].pct_change().dropna()
        
        # Align the indices
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # Skip if we have fewer than 2 symbols
        if len(returns_df.columns) < 2:
            print("Need at least 2 symbols for correlation analysis")
            return
        
        corr_analyzer = CorrelationAnalysis()
        market_viz = MarketVisualization()
        
        # Calculate correlation matrix
        corr_matrix = corr_analyzer.calculate_correlation_matrix(returns_df)
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(4))
        
        # Find highly correlated pairs
        high_corr = corr_analyzer.find_highest_correlations(corr_matrix, threshold=0.5)
        if not high_corr.empty:
            print("\nHighly correlated pairs (correlation > 0.5):")
            for _, row in high_corr.iterrows():
                print(f"  {row['asset1']} - {row['asset2']}: {row['correlation']:.4f}")
        
        # Find pairs with low correlation (good for diversification)
        low_corr = corr_analyzer.find_lowest_correlations(corr_matrix, threshold=0.3)
        if not low_corr.empty:
            print("\nPairs with low correlation (correlation < 0.3):")
            for _, row in low_corr.iterrows():
                print(f"  {row['asset1']} - {row['asset2']}: {row['correlation']:.4f}")
        
        # Create correlation matrix visualization
        fig = market_viz.plot_correlation_matrix(returns_df)
        fig.savefig("examples/output/correlation_matrix.png")
        print("Correlation matrix saved as examples/output/correlation_matrix.png")
        
        # If we have at least 3 symbols, perform PCA
        if len(returns_df.columns) >= 3:
            print("\nPerforming Principal Component Analysis...")
            try:
                pca_analyzer = PrincipalComponentAnalysis()
                pca_results = pca_analyzer.perform_pca(returns_df)
                
                # Print explained variance
                explained_var = pca_results['explained_variance']
                cumulative_var = pca_results['cumulative_variance']
                
                print("\nExplained variance by principal components:")
                for i, var in enumerate(explained_var):
                    print(f"  PC{i+1}: {var:.2%} (cumulative: {cumulative_var[i]:.2%})")
                
                # Show top contributors to first principal component
                top_contributors = pca_analyzer.get_top_contributors(pca_results, component=1)
                print("\nTop contributors to first principal component:")
                for _, row in top_contributors.iterrows():
                    print(f"  {row['Asset']}: {row['Loading']:.4f}")
            except Exception as e:
                print(f"PCA analysis failed: {e}")
    except Exception as e:
        print(f"Correlation analysis failed: {e}")


def run_portfolio_optimization(symbol_data):
    """Run portfolio optimization."""
    print("\n=== Portfolio Optimization ===")
    
    try:
        # Create returns DataFrame
        returns_data = {}
        for symbol, data in symbol_data.items():
            returns_data[symbol] = data['close'].pct_change().dropna()
        
        # Align the indices
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        # Skip if we have fewer than 3 symbols
        if len(returns_df.columns) < 3:
            print("Need at least 3 symbols for portfolio optimization")
            return
        
        # Find minimum variance portfolio
        try:
            min_var_portfolio = PortfolioOptimization.minimum_variance_portfolio(returns_df)
            
            print("\nMinimum Variance Portfolio:")
            for asset, weight in min_var_portfolio['weights'].items():
                print(f"  {asset}: {weight:.2%}")
            print(f"  Expected Annual Return: {min_var_portfolio['return']:.2%}")
            print(f"  Expected Annual Volatility: {min_var_portfolio['volatility']:.2%}")
            print(f"  Sharpe Ratio: {min_var_portfolio['sharpe_ratio']:.4f}")
        except Exception as e:
            print(f"Minimum variance portfolio calculation failed: {e}")
        
        # Find maximum Sharpe ratio portfolio
        try:
            max_sharpe_portfolio = PortfolioOptimization.maximum_sharpe_portfolio(returns_df)
            
            print("\nMaximum Sharpe Ratio Portfolio:")
            for asset, weight in max_sharpe_portfolio['weights'].items():
                print(f"  {asset}: {weight:.2%}")
            print(f"  Expected Annual Return: {max_sharpe_portfolio['return']:.2%}")
            print(f"  Expected Annual Volatility: {max_sharpe_portfolio['volatility']:.2%}")
            print(f"  Sharpe Ratio: {max_sharpe_portfolio['sharpe_ratio']:.4f}")
        except Exception as e:
            print(f"Maximum Sharpe ratio portfolio calculation failed: {e}")
        
        # Calculate efficient frontier
        try:
            print("\nCalculating efficient frontier...")
            efficient_portfolios = PortfolioOptimization.efficient_frontier(returns_df, points=20)
            
            # Plot efficient frontier
            fig = PortfolioOptimization.plot_efficient_frontier(
                efficient_portfolios=efficient_portfolios,
                asset_returns=returns_df,
                risk_free_rate=0.01  # 1% risk-free rate
            )
            fig.savefig("examples/output/efficient_frontier.png")
            print("Efficient frontier saved as examples/output/efficient_frontier.png")
        except Exception as e:
            print(f"Efficient frontier calculation failed: {e}")
    except Exception as e:
        print(f"Portfolio optimization failed: {e}")


def main():
    """Main function."""
    # Create output directory if it doesn't exist
    os.makedirs("examples/output", exist_ok=True)
    
    # Parameters - use 3 symbols for demonstration
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Generate sample data instead of downloading
    symbol_data = generate_sample_data(symbols, start_date, end_date)
    
    # Run analyses
    run_market_analysis(symbol_data)
    run_correlation_analysis(symbol_data)
    run_portfolio_optimization(symbol_data)
    
    print("\nAnalysis complete! All outputs saved to examples/output directory.")


if __name__ == "__main__":
    main() 