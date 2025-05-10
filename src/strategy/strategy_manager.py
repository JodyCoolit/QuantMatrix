from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from .strategy import Strategy


class BacktestResult:
    """Class to store backtest results."""
    
    def __init__(self, strategy_name: str, symbol: str, start_date: datetime, end_date: datetime):
        """
        Initialize backtest results.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Symbol that was traded
            start_date: Start date of the backtest
            end_date: End date of the backtest
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
        # Performance metrics
        self.total_return = 0.0
        self.annualized_return = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        
        # Trade history
        self.trades = []
        
        # Daily performance
        self.equity_curve = pd.DataFrame()
    
    def __repr__(self) -> str:
        return (f"BacktestResult(strategy={self.strategy_name}, symbol={self.symbol}, "
                f"return={self.total_return:.2%}, sharpe={self.sharpe_ratio:.2f}, "
                f"drawdown={self.max_drawdown:.2%})")


class OptimizationResult:
    """Class to store optimization results."""
    
    def __init__(self, strategy_name: str, symbol: str, parameters: Dict[str, Any]):
        """
        Initialize optimization results.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Symbol that was traded
            parameters: Parameters that were optimized
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.parameters = parameters
        
        # Results for each parameter combination
        self.results = []
        
        # Best parameter combination
        self.best_parameters = {}
        self.best_result = None
    
    def __repr__(self) -> str:
        return (f"OptimizationResult(strategy={self.strategy_name}, symbol={self.symbol}, "
                f"best_parameters={self.best_parameters})")


class StrategyManager:
    """
    StrategyManager handles strategy registration, backtesting and optimization.
    """
    
    def __init__(self):
        """Initialize the StrategyManager."""
        self.strategies: Dict[str, Strategy] = {}
    
    def add_strategy(self, strategy: Strategy) -> None:
        """
        Add a strategy to the manager.
        
        Args:
            strategy: A Strategy implementation
        """
        self.strategies[strategy.name] = strategy
    
    def get_strategy(self, name: str) -> Optional[Strategy]:
        """
        Get a strategy by name.
        
        Args:
            name: The name of the strategy
            
        Returns:
            The strategy instance or None if not found
        """
        return self.strategies.get(name)
    
    def run_backtest(self, strategy: Strategy, data: pd.DataFrame) -> BacktestResult:
        """
        Run a backtest for a strategy.
        
        Args:
            strategy: The strategy to test
            data: DataFrame with market data
            
        Returns:
            BacktestResult with performance metrics
        """
        if data.empty:
            raise ValueError("Cannot backtest on empty data")
        
        # Initialize the strategy
        strategy.initialize()
        
        # Extract symbol and date info
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'UNKNOWN'
        start_date = data.index[0]
        end_date = data.index[-1]
        
        # Create result object
        result = BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Process data and generate signals
        strategy.process_data(data)
        signals = strategy.generate_signals()
        
        if not signals:
            print(f"No signals generated for {strategy.name}")
            return result
        
        # Create equity curve and calculate returns
        equity_curve = self._calculate_equity_curve(data, signals)
        
        # Calculate performance metrics
        self._calculate_performance_metrics(result, equity_curve)
        
        # Store the equity curve
        result.equity_curve = equity_curve
        
        return result
    
    def run_optimization(self, strategy_class, data: pd.DataFrame, parameters: Dict[str, List[Any]]) -> OptimizationResult:
        """
        Optimize strategy parameters.
        
        Args:
            strategy_class: The class of the strategy to optimize
            data: DataFrame with market data
            parameters: Dictionary of parameter names to lists of values to test
            
        Returns:
            OptimizationResult with the best parameters
        """
        if data.empty:
            raise ValueError("Cannot optimize on empty data")
        
        # Extract symbol
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'UNKNOWN'
        
        # Create result object using a temporary strategy instance
        temp_strategy = strategy_class()
        result = OptimizationResult(
            strategy_name=temp_strategy.name,
            symbol=symbol,
            parameters=parameters
        )
        
        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(parameters)
        
        # Test each parameter combination
        best_sharpe = -float('inf')
        best_params = None
        best_result = None
        
        for params in param_combinations:
            # Create a new strategy instance with these parameters
            strategy = strategy_class(**params)
            
            # Run backtest
            backtest_result = self.run_backtest(strategy, data)
            
            # Store the result
            result.results.append({
                'parameters': params,
                'backtest_result': backtest_result
            })
            
            # Check if this is the best result so far
            if backtest_result.sharpe_ratio > best_sharpe:
                best_sharpe = backtest_result.sharpe_ratio
                best_params = params
                best_result = backtest_result
        
        # Store the best parameters and result
        result.best_parameters = best_params
        result.best_result = best_result
        
        return result
    
    def _calculate_equity_curve(self, data: pd.DataFrame, signals: List) -> pd.DataFrame:
        """
        Calculate the equity curve based on signals.
        
        Args:
            data: DataFrame with market data
            signals: List of trading signals
            
        Returns:
            DataFrame with equity curve
        """
        # Start with a DataFrame of just the close prices
        equity = data[['close']].copy()
        
        # Add columns for position and returns
        equity['position'] = 0  # 1 for long, -1 for short, 0 for flat
        equity['returns'] = 0.0
        
        # Process each signal to build the position series
        for signal in signals:
            if signal.timestamp not in equity.index:
                continue
                
            if signal.direction == 'buy':
                equity.loc[signal.timestamp:, 'position'] = 1
            elif signal.direction == 'sell':
                equity.loc[signal.timestamp:, 'position'] = -1
        
        # Calculate returns based on positions
        equity['returns'] = equity['close'].pct_change() * equity['position'].shift(1)
        equity.loc[equity.index[0], 'returns'] = 0  # First row has no return
        
        # Calculate cumulative returns
        equity['cumulative_returns'] = (1 + equity['returns']).cumprod() - 1
        
        # Add equity value (assuming starting with $10,000)
        equity['equity'] = 10000 * (1 + equity['cumulative_returns'])
        
        return equity
    
    def _calculate_performance_metrics(self, result: BacktestResult, equity_curve: pd.DataFrame) -> None:
        """
        Calculate performance metrics from the equity curve.
        
        Args:
            result: BacktestResult object to store metrics in
            equity_curve: DataFrame with equity curve data
        """
        # Total return
        if not equity_curve.empty and 'cumulative_returns' in equity_curve:
            result.total_return = equity_curve['cumulative_returns'].iloc[-1]
        
        # Annualized return
        days = (result.end_date - result.start_date).days
        if days > 0 and result.total_return > -1:
            result.annualized_return = (1 + result.total_return) ** (365 / days) - 1
        
        # Sharpe ratio (assuming risk-free rate of 0)
        if 'returns' in equity_curve and not equity_curve['returns'].empty:
            daily_returns = equity_curve['returns']
            if daily_returns.std() > 0:
                result.sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
        
        # Maximum drawdown
        if 'equity' in equity_curve:
            equity = equity_curve['equity']
            max_equity = equity.cummax()
            drawdown = (equity - max_equity) / max_equity
            result.max_drawdown = drawdown.min() if not drawdown.empty else 0
        
        # Win rate
        if 'returns' in equity_curve:
            returns = equity_curve['returns']
            wins = (returns > 0).sum()
            trades = (returns != 0).sum()
            result.win_rate = wins / trades if trades > 0 else 0
    
    def _generate_parameter_combinations(self, parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all combinations of parameters for optimization.
        
        Args:
            parameters: Dictionary of parameter names to lists of values
            
        Returns:
            List of dictionaries, each representing a parameter combination
        """
        import itertools
        
        param_names = list(parameters.keys())
        param_values = list(parameters.values())
        
        combinations = []
        for values in itertools.product(*param_values):
            combination = {name: value for name, value in zip(param_names, values)}
            combinations.append(combination)
        
        return combinations 