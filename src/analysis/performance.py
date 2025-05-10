import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from ..strategy.strategy_manager import BacktestResult


class PerformanceMetrics:
    """
    Class containing calculated performance metrics for a strategy.
    """
    
    def __init__(self, equity_curve: pd.DataFrame = None, trades: List = None):
        """
        Initialize performance metrics calculator.
        
        Args:
            equity_curve: DataFrame with strategy equity curve
            trades: List of trade objects
        """
        self.equity_curve = equity_curve
        self.trades = trades
        
        # Performance metrics
        self.total_return = 0.0
        self.annualized_return = 0.0
        self.volatility = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_duration = 0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.avg_win_loss_ratio = 0.0
        self.expectancy = 0.0
        self.calmar_ratio = 0.0
        self.recovery_factor = 0.0
        
        # Calculate metrics if equity curve is provided
        if equity_curve is not None and not equity_curve.empty:
            self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate all performance metrics.
        
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Calculate metrics from equity curve
        if self.equity_curve is not None and not self.equity_curve.empty:
            metrics.update(self._calculate_return_metrics())
            metrics.update(self._calculate_risk_metrics())
            metrics.update(self._calculate_drawdown_metrics())
            
            # Store metrics as attributes
            for key, value in metrics.items():
                setattr(self, key, value)
        
        # Calculate metrics from trades
        if self.trades is not None and len(self.trades) > 0:
            trade_metrics = self._calculate_trade_metrics()
            metrics.update(trade_metrics)
            
            # Store metrics as attributes
            for key, value in trade_metrics.items():
                setattr(self, key, value)
        
        return metrics
    
    def _calculate_return_metrics(self) -> Dict[str, float]:
        """
        Calculate return-based metrics from equity curve.
        
        Returns:
            Dictionary with return metrics
        """
        metrics = {}
        
        # Extract returns series
        if 'returns' in self.equity_curve.columns:
            returns = self.equity_curve['returns']
        else:
            # Calculate returns from equity
            if 'equity' in self.equity_curve.columns:
                returns = self.equity_curve['equity'].pct_change().fillna(0)
            else:
                return metrics
        
        # Total return
        if 'cumulative_returns' in self.equity_curve.columns:
            metrics['total_return'] = self.equity_curve['cumulative_returns'].iloc[-1]
        else:
            metrics['total_return'] = (1 + returns).prod() - 1
        
        # Annualized return
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        if days > 0:
            metrics['annualized_return'] = (1 + metrics['total_return']) ** (365 / days) - 1
        
        # Volatility (annualized)
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 0)
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
        
        # Sortino ratio (using only negative returns for denominator)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                metrics['sortino_ratio'] = metrics['annualized_return'] / downside_deviation
        
        # Calmar ratio (return / max drawdown)
        if metrics.get('max_drawdown', 0) > 0:
            metrics['calmar_ratio'] = metrics['annualized_return'] / metrics['max_drawdown']
        
        return metrics
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate risk-based metrics from equity curve.
        
        Returns:
            Dictionary with risk metrics
        """
        metrics = {}
        
        # Extract returns series
        if 'returns' in self.equity_curve.columns:
            returns = self.equity_curve['returns']
        else:
            # Calculate returns from equity
            if 'equity' in self.equity_curve.columns:
                returns = self.equity_curve['equity'].pct_change().fillna(0)
            else:
                return metrics
        
        # Value at Risk (VaR)
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['var_99'] = np.percentile(returns, 1)
        
        # Conditional VaR / Expected Shortfall
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()
        
        # Beta (if benchmark is available)
        if 'benchmark_returns' in self.equity_curve.columns:
            benchmark_returns = self.equity_curve['benchmark_returns']
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            if benchmark_variance > 0:
                metrics['beta'] = covariance / benchmark_variance
        
        # Alpha (if benchmark is available)
        if 'benchmark_returns' in self.equity_curve.columns and 'beta' in metrics:
            benchmark_returns = self.equity_curve['benchmark_returns']
            metrics['alpha'] = returns.mean() - metrics['beta'] * benchmark_returns.mean()
            metrics['alpha_annualized'] = metrics['alpha'] * 252
        
        return metrics
    
    def _calculate_drawdown_metrics(self) -> Dict[str, Any]:
        """
        Calculate drawdown-based metrics from equity curve.
        
        Returns:
            Dictionary with drawdown metrics
        """
        metrics = {}
        
        # Extract equity series
        if 'equity' in self.equity_curve.columns:
            equity = self.equity_curve['equity']
        elif 'cumulative_returns' in self.equity_curve.columns:
            equity = 1 + self.equity_curve['cumulative_returns']
        else:
            return metrics
        
        # Calculate running maximum
        running_max = equity.cummax()
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max
        
        # Maximum drawdown
        metrics['max_drawdown'] = abs(drawdown.min())
        
        # Drawdown durations
        is_in_drawdown = equity < running_max
        
        # Find start and end of all drawdowns
        drawdown_start = is_in_drawdown.shift(1).fillna(False) & is_in_drawdown
        drawdown_end = is_in_drawdown & (~is_in_drawdown.shift(-1).fillna(False))
        
        max_duration = 0
        recovery_time = 0
        
        # Calculate max drawdown duration
        if is_in_drawdown.any():
            start_dates = self.equity_curve.index[drawdown_start]
            end_dates = self.equity_curve.index[drawdown_end]
            
            if len(start_dates) > 0 and len(end_dates) > 0:
                durations = [(end - start).days for start, end in zip(start_dates, end_dates)]
                if durations:
                    max_duration = max(durations)
        
        metrics['max_drawdown_duration'] = max_duration
        
        # Recovery factor
        if metrics['max_drawdown'] > 0:
            metrics['recovery_factor'] = metrics.get('total_return', 0) / metrics['max_drawdown']
        
        return metrics
    
    def _calculate_trade_metrics(self) -> Dict[str, float]:
        """
        Calculate trade-based metrics.
        
        Returns:
            Dictionary with trade metrics
        """
        metrics = {}
        
        if not self.trades:
            return metrics
        
        # Filter completed trades
        completed_trades = [
            t for t in self.trades 
            if hasattr(t, 'exit_price') and t.exit_price is not None
        ]
        
        if not completed_trades:
            return metrics
        
        # Calculate trade profits
        profits = [
            t.calculate_pnl() if hasattr(t, 'calculate_pnl') else t.pnl 
            for t in completed_trades
        ]
        
        # Win rate
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        metrics['win_rate'] = len(winning_trades) / len(profits) if profits else 0
        
        # Average win/loss
        metrics['avg_win'] = np.mean(winning_trades) if winning_trades else 0
        metrics['avg_loss'] = abs(np.mean(losing_trades)) if losing_trades else 0
        
        # Win/loss ratio
        if metrics['avg_loss'] > 0:
            metrics['avg_win_loss_ratio'] = metrics['avg_win'] / metrics['avg_loss']
        
        # Profit factor
        total_win = sum(winning_trades) if winning_trades else 0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0
        
        if total_loss > 0:
            metrics['profit_factor'] = total_win / total_loss
        
        # Expectancy
        metrics['expectancy'] = (
            metrics['win_rate'] * metrics['avg_win'] - 
            (1 - metrics['win_rate']) * metrics['avg_loss']
        )
        
        return metrics
    
    def __repr__(self) -> str:
        return (f"PerformanceMetrics(return={self.total_return:.2%}, "
                f"sharpe={self.sharpe_ratio:.2f}, drawdown={self.max_drawdown:.2%})")


class PerformanceAnalyzer:
    """
    Analyze the performance of trading strategies.
    """
    
    def __init__(self):
        """Initialize the performance analyzer."""
        pass
    
    def analyze_backtest(self, result: BacktestResult) -> PerformanceMetrics:
        """
        Analyze a backtest result.
        
        Args:
            result: BacktestResult from strategy manager
            
        Returns:
            PerformanceMetrics with calculated metrics
        """
        if not hasattr(result, 'equity_curve') or result.equity_curve.empty:
            raise ValueError("Backtest result must contain an equity curve")
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            equity_curve=result.equity_curve,
            trades=result.trades
        )
        
        # Calculate metrics
        metrics.calculate_metrics()
        
        return metrics
    
    def compare_strategies(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """
        Compare multiple strategy backtest results.
        
        Args:
            results: Dictionary of strategy name to BacktestResult
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison = {}
        
        for strategy_name, result in results.items():
            # Calculate metrics
            metrics = self.analyze_backtest(result)
            
            # Store key metrics for comparison
            comparison[strategy_name] = {
                'Total Return': metrics.total_return,
                'Annualized Return': metrics.annualized_return,
                'Sharpe Ratio': metrics.sharpe_ratio,
                'Sortino Ratio': metrics.sortino_ratio,
                'Max Drawdown': metrics.max_drawdown,
                'Max Drawdown Duration': metrics.max_drawdown_duration,
                'Win Rate': metrics.win_rate,
                'Profit Factor': metrics.profit_factor,
                'Expectancy': metrics.expectancy
            }
        
        # Convert to DataFrame
        return pd.DataFrame(comparison).T
    
    def get_monthly_returns(self, equity_curve: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monthly returns from equity curve.
        
        Args:
            equity_curve: DataFrame with strategy equity curve
            
        Returns:
            DataFrame with monthly returns
        """
        if 'equity' not in equity_curve.columns:
            raise ValueError("Equity curve must contain an 'equity' column")
        
        # Resample to month-end and calculate returns
        monthly_equity = equity_curve['equity'].resample('M').last()
        monthly_returns = monthly_equity.pct_change().fillna(0)
        
        # Convert to DataFrame with month and year as indices
        result = pd.DataFrame({
            'Return': monthly_returns
        })
        
        # Add year and month columns
        result['Year'] = result.index.year
        result['Month'] = result.index.month
        
        # Create a pivot table
        pivot = result.pivot_table(
            values='Return',
            index='Year',
            columns='Month'
        )
        
        # Rename columns to month names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        pivot = pivot.rename(columns=month_names)
        
        # Add yearly total
        pivot['Year Total'] = (1 + pivot.fillna(0)).prod(axis=1) - 1
        
        return pivot 