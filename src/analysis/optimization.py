import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

from ..strategy.strategy_manager import StrategyManager, BacktestResult, Strategy


class StrategyOptimizer:
    """
    Class for optimizing strategy parameters.
    """
    
    def __init__(self, strategy_manager: StrategyManager):
        """
        Initialize the strategy optimizer.
        
        Args:
            strategy_manager: StrategyManager instance
        """
        self.strategy_manager = strategy_manager
    
    def grid_search(self, strategy_class, data: pd.DataFrame, parameters: Dict[str, List[Any]], 
                  metric: str = 'sharpe_ratio', maximize: bool = True) -> Dict[str, Any]:
        """
        Perform grid search to find optimal parameters.
        
        Args:
            strategy_class: The class of the strategy to optimize
            data: DataFrame with market data
            parameters: Dictionary of parameter names to lists of values to test
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            maximize: Whether to maximize (True) or minimize (False) the metric
            
        Returns:
            Dictionary with optimization results
        """
        # Run optimization
        optimization_result = self.strategy_manager.run_optimization(
            strategy_class=strategy_class,
            data=data,
            parameters=parameters
        )
        
        # Sort results by metric
        results = []
        
        for result in optimization_result.results:
            params = result['parameters']
            backtest_result = result['backtest_result']
            
            # Get the specified metric
            if hasattr(backtest_result, metric):
                metric_value = getattr(backtest_result, metric)
                
                results.append({
                    'parameters': params,
                    'metric_value': metric_value,
                    'backtest_result': backtest_result
                })
        
        # Sort by metric value
        results.sort(key=lambda x: x['metric_value'], reverse=maximize)
        
        # Create summary
        summary = {
            'best_parameters': results[0]['parameters'] if results else None,
            'best_metric_value': results[0]['metric_value'] if results else None,
            'best_backtest_result': results[0]['backtest_result'] if results else None,
            'all_results': results,
            'strategy_name': strategy_class().name,
            'metric': metric,
            'maximize': maximize
        }
        
        return summary
    
    def plot_parameter_surface(self, results: List[Dict[str, Any]], param1: str, param2: str, 
                             metric: str, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot a 3D surface of parameter combinations.
        
        Args:
            results: List of optimization results
            param1: First parameter name
            param2: Second parameter name
            metric: Metric to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Extract unique parameter values
        param1_values = sorted(list(set([r['parameters'][param1] for r in results])))
        param2_values = sorted(list(set([r['parameters'][param2] for r in results])))
        
        # Create a grid of values
        X, Y = np.meshgrid(param1_values, param2_values)
        Z = np.zeros_like(X, dtype=float)
        
        # Fill the grid with metric values
        for i, p1 in enumerate(param1_values):
            for j, p2 in enumerate(param2_values):
                # Find the result with these parameters
                for result in results:
                    params = result['parameters']
                    if params[param1] == p1 and params[param2] == p2:
                        Z[j, i] = result['metric_value']
                        break
        
        # Create the figure
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        # Add colorbar
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        
        # Add labels
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_zlabel(metric)
        ax.set_title(f'Parameter Surface: {metric}')
        
        plt.tight_layout()
        
        return fig
    
    def plot_parameter_heatmap(self, results: List[Dict[str, Any]], param1: str, param2: str, 
                             metric: str, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot a heatmap of parameter combinations.
        
        Args:
            results: List of optimization results
            param1: First parameter name
            param2: Second parameter name
            metric: Metric to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Extract unique parameter values
        param1_values = sorted(list(set([r['parameters'][param1] for r in results])))
        param2_values = sorted(list(set([r['parameters'][param2] for r in results])))
        
        # Create a dataframe for the heatmap
        data = []
        
        for result in results:
            params = result['parameters']
            metric_value = result['metric_value']
            
            data.append({
                param1: params[param1],
                param2: params[param2],
                metric: metric_value
            })
        
        df = pd.DataFrame(data)
        
        # Pivot the data
        pivot_table = df.pivot(index=param2, columns=param1, values=metric)
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(pivot_table, annot=True, fmt=".4g", cmap="viridis", ax=ax)
        
        # Add labels
        ax.set_title(f'Parameter Heatmap: {metric}')
        
        plt.tight_layout()
        
        return fig
    
    def walk_forward_optimization(self, strategy_class, data: pd.DataFrame, 
                                parameters: Dict[str, List[Any]], 
                                train_size: int, test_size: int, 
                                metric: str = 'sharpe_ratio', 
                                maximize: bool = True) -> Dict[str, Any]:
        """
        Perform walk-forward optimization.
        
        Args:
            strategy_class: The class of the strategy to optimize
            data: DataFrame with market data
            parameters: Dictionary of parameter names to lists of values to test
            train_size: Size of training window in days
            test_size: Size of test window in days
            metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            maximize: Whether to maximize (True) or minimize (False) the metric
            
        Returns:
            Dictionary with walk-forward optimization results
        """
        # Check if data has enough samples
        if len(data) < train_size + test_size:
            raise ValueError("Not enough data for walk-forward optimization")
        
        # Calculate number of walk-forward steps
        total_days = len(data)
        steps = (total_days - train_size) // test_size
        
        # Initialize results
        walk_forward_results = []
        
        for step in range(steps):
            # Calculate indices for this step
            train_start = step * test_size
            train_end = train_start + train_size
            test_start = train_end
            test_end = min(test_start + test_size, total_days)
            
            # Skip if we don't have enough data for testing
            if test_end - test_start < test_size / 2:
                break
            
            # Get training and testing data
            train_data = data.iloc[train_start:train_end].copy()
            test_data = data.iloc[test_start:test_end].copy()
            
            # Optimize on training data
            optimization = self.grid_search(
                strategy_class, 
                train_data, 
                parameters, 
                metric, 
                maximize
            )
            
            # Test optimal parameters on test data
            best_params = optimization['best_parameters']
            strategy = strategy_class(**best_params)
            
            test_result = self.strategy_manager.run_backtest(strategy, test_data)
            
            # Store results
            walk_forward_results.append({
                'step': step,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_parameters': best_params,
                'train_metric': optimization['best_metric_value'],
                'test_metric': getattr(test_result, metric),
                'test_result': test_result
            })
        
        # Calculate overall performance
        all_test_metrics = [r['test_metric'] for r in walk_forward_results]
        combined_equity = pd.DataFrame()
        
        for result in walk_forward_results:
            if hasattr(result['test_result'], 'equity_curve') and not result['test_result'].equity_curve.empty:
                # Add this segment to the combined equity curve
                segment = result['test_result'].equity_curve[['equity']].copy()
                
                # If not the first segment, adjust starting value to continue from previous segment
                if not combined_equity.empty and not segment.empty:
                    scale_factor = combined_equity['equity'].iloc[-1] / segment['equity'].iloc[0]
                    segment['equity'] = segment['equity'] * scale_factor
                
                combined_equity = pd.concat([combined_equity, segment])
        
        # Calculate overall metrics
        if not combined_equity.empty:
            returns = combined_equity['equity'].pct_change().dropna()
            
            overall_metrics = {
                'total_return': combined_equity['equity'].iloc[-1] / combined_equity['equity'].iloc[0] - 1 if len(combined_equity) > 1 else 0,
                'annualized_return': np.power(combined_equity['equity'].iloc[-1] / combined_equity['equity'].iloc[0], 252 / len(combined_equity)) - 1 if len(combined_equity) > 1 else 0,
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'max_drawdown': ((combined_equity['equity'] / combined_equity['equity'].cummax()) - 1).min() if not combined_equity.empty else 0
            }
        else:
            overall_metrics = {}
        
        # Create summary
        summary = {
            'walk_forward_results': walk_forward_results,
            'parameter_stability': {param: [r['best_parameters'][param] for r in walk_forward_results] for param in parameters.keys()},
            'mean_test_metric': np.mean(all_test_metrics) if all_test_metrics else None,
            'median_test_metric': np.median(all_test_metrics) if all_test_metrics else None,
            'combined_equity': combined_equity,
            'overall_metrics': overall_metrics
        }
        
        return summary


class PortfolioOptimization:
    """
    Class for portfolio optimization.
    """
    
    @staticmethod
    def minimum_variance_portfolio(returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Find the minimum variance portfolio.
        
        Args:
            returns_df: DataFrame of asset returns (each column is an asset)
            
        Returns:
            Dictionary with optimization results
        """
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Number of assets
        n = len(returns_df.columns)
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n) / n
        
        # Constraints (weights sum to 1)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds (no short selling)
        bounds = tuple((0, 1) for asset in range(n))
        
        # Objective function (portfolio variance)
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Optimize
        result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        # Get optimal weights
        optimal_weights = result['x']
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(mean_returns * optimal_weights) * 252  # Annualized
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)) * 252)  # Annualized
        portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Create result dictionary
        weights_dict = {asset: weight for asset, weight in zip(returns_df.columns, optimal_weights)}
        
        optimization_result = {
            'weights': weights_dict,
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': portfolio_sharpe
        }
        
        return optimization_result
    
    @staticmethod
    def maximum_sharpe_portfolio(returns_df: pd.DataFrame, risk_free_rate: float = 0.0) -> Dict[str, Any]:
        """
        Find the maximum Sharpe ratio portfolio.
        
        Args:
            returns_df: DataFrame of asset returns (each column is an asset)
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Dictionary with optimization results
        """
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Number of assets
        n = len(returns_df.columns)
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n) / n
        
        # Constraints (weights sum to 1)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds (no short selling)
        bounds = tuple((0, 1) for asset in range(n))
        
        # Convert daily returns to annual returns
        annual_return = mean_returns * 252
        
        # Objective function (negative Sharpe ratio to maximize)
        def neg_sharpe_ratio(weights):
            portfolio_return = np.sum(annual_return * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252)
            return -(portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Optimize
        result = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        # Get optimal weights
        optimal_weights = result['x']
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(mean_returns * optimal_weights) * 252  # Annualized
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)) * 252)  # Annualized
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Create result dictionary
        weights_dict = {asset: weight for asset, weight in zip(returns_df.columns, optimal_weights)}
        
        optimization_result = {
            'weights': weights_dict,
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': portfolio_sharpe
        }
        
        return optimization_result
    
    @staticmethod
    def efficient_frontier(returns_df: pd.DataFrame, points: int = 20) -> List[Dict[str, Any]]:
        """
        Calculate the efficient frontier.
        
        Args:
            returns_df: DataFrame of asset returns (each column is an asset)
            points: Number of points to calculate on the frontier
            
        Returns:
            List of portfolios on the efficient frontier
        """
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Number of assets
        n = len(returns_df.columns)
        
        # Constraints (weights sum to 1)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds (no short selling)
        bounds = tuple((0, 1) for asset in range(n))
        
        # First find the minimum variance portfolio
        min_var_result = PortfolioOptimization.minimum_variance_portfolio(returns_df)
        min_return = min_var_result['return']
        
        # Then find a high return portfolio to determine the range
        def neg_portfolio_return(weights):
            return -np.sum(mean_returns * weights) * 252
        
        high_return_result = minimize(neg_portfolio_return, np.ones(n) / n, method='SLSQP',
                                    bounds=bounds, constraints=constraints)
        high_return = -neg_portfolio_return(high_return_result['x'])
        
        # Create a range of target returns
        target_returns = np.linspace(min_return, high_return, points)
        efficient_portfolios = []
        
        # For each target return, find the minimum variance portfolio
        for target in target_returns:
            # Add return constraint
            return_constraint = {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) * 252 - target}
            constraints_with_return = (constraints, return_constraint)
            
            # Initial guess (equal weights)
            initial_weights = np.ones(n) / n
            
            # Objective function (portfolio variance)
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights)) * 252
            
            # Optimize
            result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                             bounds=bounds, constraints=constraints_with_return)
            
            # Skip if optimization failed
            if not result['success']:
                continue
            
            # Get optimal weights
            optimal_weights = result['x']
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(mean_returns * optimal_weights) * 252  # Annualized
            portfolio_vol = np.sqrt(portfolio_variance(optimal_weights))  # Annualized
            portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            
            # Create result dictionary
            weights_dict = {asset: weight for asset, weight in zip(returns_df.columns, optimal_weights)}
            
            portfolio = {
                'weights': weights_dict,
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': portfolio_sharpe
            }
            
            efficient_portfolios.append(portfolio)
        
        return efficient_portfolios
    
    @staticmethod
    def plot_efficient_frontier(efficient_portfolios: List[Dict[str, Any]], 
                              asset_returns: pd.DataFrame,
                              risk_free_rate: float = 0.0,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the efficient frontier.
        
        Args:
            efficient_portfolios: List of portfolios on the efficient frontier
            asset_returns: DataFrame of asset returns (for plotting individual assets)
            risk_free_rate: Risk-free rate (annualized)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Extract returns and volatilities
        returns = [p['return'] for p in efficient_portfolios]
        vols = [p['volatility'] for p in efficient_portfolios]
        sharpes = [p['sharpe_ratio'] for p in efficient_portfolios]
        
        # Find the maximum Sharpe ratio portfolio
        max_sharpe_idx = np.argmax(sharpes)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the efficient frontier
        ax.plot(vols, returns, 'b-', linewidth=3, label='Efficient Frontier')
        
        # Highlight the maximum Sharpe ratio portfolio
        ax.scatter(vols[max_sharpe_idx], returns[max_sharpe_idx], marker='*', color='r', s=200, 
                 label='Maximum Sharpe Ratio')
        
        # Plot the capital market line if risk-free rate is provided
        if risk_free_rate is not None:
            max_sharpe_return = returns[max_sharpe_idx]
            max_sharpe_vol = vols[max_sharpe_idx]
            
            # Calculate the capital market line
            cml_x = [0, max_sharpe_vol * 1.5]
            cml_y = [risk_free_rate, risk_free_rate + (max_sharpe_return - risk_free_rate) * 1.5 / max_sharpe_vol]
            
            ax.plot(cml_x, cml_y, 'g--', label='Capital Market Line')
            ax.plot(0, risk_free_rate, 'go', label='Risk-Free Rate')
        
        # Plot individual assets
        asset_returns_annual = asset_returns.mean() * 252
        asset_vols_annual = asset_returns.std() * np.sqrt(252)
        
        ax.scatter(asset_vols_annual, asset_returns_annual, marker='o', color='grey', 
                  alpha=0.6, label='Individual Assets')
        
        # Add asset labels
        for i, asset in enumerate(asset_returns.columns):
            ax.annotate(asset, 
                       (asset_vols_annual[i], asset_returns_annual[i]),
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8)
        
        # Formatting
        ax.set_title('Efficient Frontier', fontsize=14)
        ax.set_xlabel('Volatility (Annualized)')
        ax.set_ylabel('Return (Annualized)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        return fig 