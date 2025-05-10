import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from arch import arch_model


class MarketStatistics:
    """
    Statistical analysis tools for market data.
    """
    
    @staticmethod
    def calculate_descriptive_stats(returns: pd.Series) -> pd.Series:
        """
        Calculate descriptive statistics for a returns series.
        
        Args:
            returns: Series of returns to analyze
            
        Returns:
            Series with descriptive statistics
        """
        stats_dict = {
            'Mean': returns.mean(),
            'Median': returns.median(),
            'Min': returns.min(),
            'Max': returns.max(),
            'Std Dev': returns.std(),
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis(),
            'Positive Days': (returns > 0).sum() / len(returns),
            'Negative Days': (returns < 0).sum() / len(returns)
        }
        
        return pd.Series(stats_dict)
    
    @staticmethod
    def test_normality(returns: pd.Series) -> Dict[str, Union[float, bool]]:
        """
        Test if returns follow a normal distribution.
        
        Args:
            returns: Series of returns to test
            
        Returns:
            Dictionary with test results
        """
        # Jarque-Bera test
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # Shapiro-Wilk test
        sw_stat, sw_pvalue = stats.shapiro(returns)
        
        # D'Agostino-Pearson test
        dp_stat, dp_pvalue = stats.normaltest(returns)
        
        results = {
            'jarque_bera_statistic': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'jarque_bera_normal': jb_pvalue > 0.05,
            'shapiro_wilk_statistic': sw_stat,
            'shapiro_wilk_pvalue': sw_pvalue,
            'shapiro_wilk_normal': sw_pvalue > 0.05,
            'dagostino_pearson_statistic': dp_stat,
            'dagostino_pearson_pvalue': dp_pvalue,
            'dagostino_pearson_normal': dp_pvalue > 0.05
        }
        
        return results
    
    @staticmethod
    def test_stationarity(series: pd.Series) -> Dict[str, Union[float, bool]]:
        """
        Test if a time series is stationary.
        
        Args:
            series: Time series to test
            
        Returns:
            Dictionary with test results
        """
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series, autolag='AIC')
        
        # KPSS test
        kpss_result = kpss(series, regression='c')
        
        results = {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_stationary': adf_result[1] < 0.05,
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'kpss_stationary': kpss_result[1] > 0.05
        }
        
        return results
    
    @staticmethod
    def test_autocorrelation(returns: pd.Series, lags: int = 10) -> Dict[str, Union[float, bool]]:
        """
        Test for autocorrelation in returns.
        
        Args:
            returns: Series of returns to test
            lags: Number of lags to test
            
        Returns:
            Dictionary with test results
        """
        # Ljung-Box test
        lb_stat, lb_pvalue = sm.stats.acorr_ljungbox(returns, lags=[lags], return_df=False)
        
        # Durbin-Watson test
        dw_stat = sm.stats.stattools.durbin_watson(returns)
        
        # Calculate autocorrelation and partial autocorrelation functions
        acf_values = acf(returns, nlags=lags)
        pacf_values = pacf(returns, nlags=lags)
        
        results = {
            'ljung_box_statistic': lb_stat[0],
            'ljung_box_pvalue': lb_pvalue[0],
            'ljung_box_autocorrelation': lb_pvalue[0] < 0.05,
            'durbin_watson_statistic': dw_stat,
            'durbin_watson_autocorrelation': dw_stat < 1.5 or dw_stat > 2.5,
            'acf': acf_values,
            'pacf': pacf_values
        }
        
        return results
    
    @staticmethod
    def estimate_volatility(returns: pd.Series, p: int = 1, q: int = 1) -> Dict[str, any]:
        """
        Estimate volatility using a GARCH model.
        
        Args:
            returns: Series of returns
            p: GARCH lag order
            q: ARCH lag order
            
        Returns:
            Dictionary with volatility model results
        """
        # Fit GARCH(p,q) model
        model = arch_model(returns, vol='Garch', p=p, q=q)
        model_fit = model.fit(disp='off')
        
        # Get conditional volatility
        conditional_vol = model_fit.conditional_volatility
        
        results = {
            'model': model_fit,
            'conditional_volatility': conditional_vol,
            'unconditional_volatility': np.sqrt(model_fit.params['omega'] / (1 - model_fit.params['alpha[1]'] - model_fit.params['beta[1]'])),
            'persistence': model_fit.params['alpha[1]'] + model_fit.params['beta[1]'],
            'half_life': np.log(0.5) / np.log(model_fit.params['alpha[1]'] + model_fit.params['beta[1]']),
            'forecast': model_fit.forecast()
        }
        
        return results
    
    @staticmethod
    def calculate_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate the Hurst exponent to measure long-memory in a time series.
        
        Args:
            series: Time series to analyze
            max_lag: Maximum lag for calculation
            
        Returns:
            Hurst exponent (0.5=random walk, >0.5=trending, <0.5=mean-reverting)
        """
        lags = range(2, max_lag)
        
        # Calculate variance of differences
        tau = [np.std(np.subtract(series.values[lag:], series.values[:-lag])) for lag in lags]
        
        # Fit power law
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        
        # Hurst exponent is the slope
        return poly[0]


class StrategyStatistics:
    """
    Statistical analysis tools for trading strategies.
    """
    
    @staticmethod
    def calculate_drawdowns(equity_curve: pd.Series) -> pd.DataFrame:
        """
        Calculate all drawdowns in an equity curve.
        
        Args:
            equity_curve: Series of equity values
            
        Returns:
            DataFrame with drawdown information
        """
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown percentage
        drawdown = (equity_curve - running_max) / running_max
        
        # Identify drawdown periods
        is_drawdown = equity_curve < running_max
        
        # Find start and end of drawdowns
        # A new drawdown starts when we reach a new peak and then decline
        drawdown_start = (~is_drawdown).shift(1).fillna(False) & is_drawdown
        
        # A drawdown ends when we return to the previous peak
        drawdown_end = is_drawdown & (~is_drawdown.shift(-1).fillna(False))
        
        # Get start and end dates
        start_dates = equity_curve.index[drawdown_start]
        end_dates = equity_curve.index[drawdown_end]
        
        # Create drawdown data
        drawdowns = []
        
        for i in range(min(len(start_dates), len(end_dates))):
            start_date = start_dates[i]
            end_date = end_dates[i]
            
            # Find the lowest point in this drawdown
            section = drawdown.loc[start_date:end_date]
            lowest_date = section.idxmin()
            max_drawdown = section.min()
            
            # Calculate recovery time in days
            recovery_time = (end_date - lowest_date).days
            
            # Add to list
            drawdowns.append({
                'start_date': start_date,
                'lowest_date': lowest_date,
                'end_date': end_date,
                'max_drawdown': max_drawdown,
                'recovery_time': recovery_time,
                'duration': (end_date - start_date).days
            })
        
        # Create DataFrame
        if drawdowns:
            return pd.DataFrame(drawdowns).sort_values('max_drawdown', ascending=True)
        else:
            return pd.DataFrame(columns=[
                'start_date', 'lowest_date', 'end_date',
                'max_drawdown', 'recovery_time', 'duration'
            ])
    
    @staticmethod
    def analyze_returns_distribution(returns: pd.Series) -> Dict[str, float]:
        """
        Analyze the distribution of returns.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with distribution metrics
        """
        # Calculate percentiles
        percentiles = {
            'percentile_1': returns.quantile(0.01),
            'percentile_5': returns.quantile(0.05),
            'percentile_10': returns.quantile(0.1),
            'percentile_25': returns.quantile(0.25),
            'percentile_50': returns.quantile(0.5),
            'percentile_75': returns.quantile(0.75),
            'percentile_90': returns.quantile(0.9),
            'percentile_95': returns.quantile(0.95),
            'percentile_99': returns.quantile(0.99)
        }
        
        # Calculate higher moments
        moments = {
            'mean': returns.mean(),
            'std': returns.std(),
            'skew': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
        
        # Calculate tail ratios
        tail_metrics = {
            'upside_capture': returns[returns > 0].mean() / abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else float('inf'),
            'downside_deviation': returns[returns < 0].std(),
            'sortino_denominator': np.sqrt((returns[returns < 0] ** 2).mean()),
            'gain_to_pain': returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')
        }
        
        # Combine all metrics
        results = {**percentiles, **moments, **tail_metrics}
        
        return results
    
    @staticmethod
    def analyze_trade_duration(trades: List) -> pd.DataFrame:
        """
        Analyze the duration of trades.
        
        Args:
            trades: List of trade objects with entry_time and exit_time
            
        Returns:
            DataFrame with trade duration statistics
        """
        # Extract durations
        durations = []
        
        for trade in trades:
            if hasattr(trade, 'exit_time') and trade.exit_time is not None:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / (60 * 60 * 24)  # in days
                
                durations.append({
                    'symbol': trade.symbol,
                    'side': trade.side.value if hasattr(trade.side, 'value') else trade.side,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'duration_days': duration,
                    'pnl': trade.calculate_pnl() if hasattr(trade, 'calculate_pnl') else trade.pnl
                })
        
        if not durations:
            return pd.DataFrame()
        
        df = pd.DataFrame(durations)
        
        # Calculate summary statistics
        results = {
            'mean_duration': df['duration_days'].mean(),
            'median_duration': df['duration_days'].median(),
            'min_duration': df['duration_days'].min(),
            'max_duration': df['duration_days'].max(),
            'std_duration': df['duration_days'].std()
        }
        
        # Split by winning/losing trades
        if 'pnl' in df.columns:
            winning = df[df['pnl'] > 0]
            losing = df[df['pnl'] < 0]
            
            if not winning.empty:
                results['winning_mean_duration'] = winning['duration_days'].mean()
                results['winning_median_duration'] = winning['duration_days'].median()
            
            if not losing.empty:
                results['losing_mean_duration'] = losing['duration_days'].mean()
                results['losing_median_duration'] = losing['duration_days'].median()
        
        return pd.Series(results)
    
    @staticmethod
    def analyze_win_loss_streaks(trades: List) -> Dict[str, any]:
        """
        Analyze winning and losing streaks in trades.
        
        Args:
            trades: List of trade objects with PnL
            
        Returns:
            Dictionary with streak statistics
        """
        if not trades:
            return {}
        
        # Extract PnLs
        pnls = []
        
        for trade in trades:
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                pnls.append(trade.pnl)
            elif hasattr(trade, 'calculate_pnl'):
                pnls.append(trade.calculate_pnl())
        
        if not pnls:
            return {}
        
        # Convert to binary outcomes (win/loss)
        outcomes = [1 if pnl > 0 else -1 if pnl < 0 else 0 for pnl in pnls]
        
        # Find streaks
        current_streak = 0
        current_type = None
        streaks = []
        
        for outcome in outcomes:
            if outcome == 0:  # Ignore flat trades
                continue
                
            if outcome != current_type:
                if current_streak > 0:
                    streaks.append({
                        'type': 'win' if current_type == 1 else 'loss',
                        'length': current_streak
                    })
                current_streak = 1
                current_type = outcome
            else:
                current_streak += 1
        
        # Add the last streak
        if current_streak > 0:
            streaks.append({
                'type': 'win' if current_type == 1 else 'loss',
                'length': current_streak
            })
        
        # Analyze streaks
        win_streaks = [s['length'] for s in streaks if s['type'] == 'win']
        loss_streaks = [s['length'] for s in streaks if s['type'] == 'loss']
        
        results = {
            'win_streaks': win_streaks,
            'loss_streaks': loss_streaks,
            'max_win_streak': max(win_streaks) if win_streaks else 0,
            'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
            'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'win_streak_std': np.std(win_streaks) if len(win_streaks) > 1 else 0,
            'loss_streak_std': np.std(loss_streaks) if len(loss_streaks) > 1 else 0
        }
        
        return results 