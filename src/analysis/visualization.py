import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.ticker as mticker

from ..strategy.strategy_manager import BacktestResult


class PerformanceVisualization:
    """
    Visualization tools for strategy performance analysis.
    """
    
    @staticmethod
    def plot_equity_curve(equity_curve: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None,
                        figsize: Tuple[int, int] = (12, 6), show_drawdowns: bool = True) -> plt.Figure:
        """
        Plot equity curve with optional benchmark comparison.
        
        Args:
            equity_curve: DataFrame with equity curve data
            benchmark_data: Optional DataFrame with benchmark data
            figsize: Figure size
            show_drawdowns: Whether to highlight drawdown periods
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract equity series
        if 'equity' in equity_curve.columns:
            equity = equity_curve['equity']
        elif 'cumulative_returns' in equity_curve.columns:
            # Convert cumulative returns to equity (starting with 10000)
            equity = 10000 * (1 + equity_curve['cumulative_returns'])
        else:
            raise ValueError("Equity curve must contain 'equity' or 'cumulative_returns' column")
        
        # Plot equity curve
        ax.plot(equity.index, equity, label='Strategy', linewidth=2)
        
        # Plot benchmark if provided
        if benchmark_data is not None and not benchmark_data.empty:
            # Normalize benchmark to start at the same value as strategy
            if 'close' in benchmark_data.columns:
                start_equity = equity.iloc[0]
                benchmark_returns = benchmark_data['close'] / benchmark_data['close'].iloc[0]
                benchmark_equity = start_equity * benchmark_returns
                
                ax.plot(benchmark_data.index, benchmark_equity, label='Benchmark', 
                       linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Highlight drawdown periods
        if show_drawdowns:
            # Calculate running maximum for drawdowns
            running_max = equity.cummax()
            
            # Find periods where equity is below its previous peak
            is_drawdown = equity < running_max
            
            # Highlight major drawdowns (greater than 5%)
            drawdown_pct = (equity - running_max) / running_max
            is_major_drawdown = drawdown_pct < -0.05
            
            # Get contiguous regions of drawdown
            drawdown_regions = []
            in_drawdown = False
            start_idx = None
            
            for i, (is_dd, is_major) in enumerate(zip(is_drawdown, is_major_drawdown)):
                if is_major and not in_drawdown:
                    # Start of new drawdown
                    in_drawdown = True
                    start_idx = i
                elif not is_major and in_drawdown:
                    # End of drawdown
                    in_drawdown = False
                    if start_idx is not None:
                        drawdown_regions.append((equity.index[start_idx], equity.index[i]))
            
            # Add final drawdown if still in one
            if in_drawdown and start_idx is not None:
                drawdown_regions.append((equity.index[start_idx], equity.index[-1]))
            
            # Highlight drawdown regions
            for start, end in drawdown_regions:
                ax.axvspan(start, end, alpha=0.2, color='red')
        
        # Formatting
        ax.set_title('Strategy Performance', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Format y-axis with dollar amounts
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_drawdowns(equity_curve: pd.Series, top_n: int = 5, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot the largest drawdowns.
        
        Args:
            equity_curve: Series of equity values
            top_n: Number of largest drawdowns to highlight
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate drawdowns
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot full drawdown curve
        ax.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3, label='Drawdown')
        ax.plot(drawdown.index, drawdown, color='red', alpha=0.5)
        
        # Find worst drawdowns
        worst_idx = drawdown.nsmallest(top_n).index
        
        # Find the recovery points for each drawdown
        colors = plt.cm.tab10(np.linspace(0, 1, top_n))
        
        for i, idx in enumerate(worst_idx):
            # Find the recovery point (if any)
            recovery_idx = drawdown.loc[idx:].ge(0).idxmax()
            
            if recovery_idx == idx or recovery_idx == drawdown.index[-1] and drawdown.iloc[-1] < 0:
                # No recovery yet, use last date
                recovery_idx = drawdown.index[-1]
            
            # Highlight this drawdown
            ax.plot(drawdown.loc[idx:recovery_idx].index, 
                   drawdown.loc[idx:recovery_idx], 
                   color=colors[i], 
                   linewidth=2, 
                   label=f'#{i+1}: {drawdown[idx]:.2%} ({idx.strftime("%Y-%m-%d")})')
        
        # Formatting
        ax.set_title('Portfolio Drawdowns', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_monthly_returns_heatmap(returns: pd.Series, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create a heatmap of monthly returns.
        
        Args:
            returns: Series of returns with datetime index
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Resample to get monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create a DataFrame with year and month as indices
        returns_df = pd.DataFrame({
            'Return': monthly_returns
        })
        
        returns_df['Year'] = returns_df.index.year
        returns_df['Month'] = returns_df.index.month
        
        # Pivot to create year x month matrix
        pivot_table = returns_df.pivot_table(
            values='Return',
            index='Year',
            columns='Month'
        )
        
        # Rename columns to month names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        pivot_table = pivot_table.rename(columns=month_names)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            pivot_table, 
            annot=True, 
            fmt=".2%", 
            cmap="RdYlGn", 
            center=0, 
            cbar=True,
            ax=ax
        )
        
        # Formatting
        ax.set_title('Monthly Returns (%)', fontsize=14)
        ax.set_ylabel('Year')
        ax.set_xlabel('')
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_rolling_metrics(returns: pd.Series, window: int = 60, 
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            returns: Series of returns
            window: Rolling window size in days
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate rolling metrics
        rolling_return = returns.rolling(window).apply(lambda x: (1 + x).prod() - 1)
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (returns.rolling(window).mean() * 252) / (returns.rolling(window).std() * np.sqrt(252))
        
        # Calculate rolling max drawdown
        rolling_dd = pd.Series(index=returns.index, dtype=float)
        
        for i in range(len(returns) - window + 1):
            window_returns = returns.iloc[i:i+window]
            equity_curve = (1 + window_returns).cumprod()
            peak = equity_curve.cummax()
            drawdown = (equity_curve - peak) / peak
            rolling_dd.iloc[i+window-1] = drawdown.min()
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot rolling return
        axes[0].plot(rolling_return.index, rolling_return, color='blue')
        axes[0].set_title(f'Rolling {window}-Day Return')
        axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        axes[0].grid(True, alpha=0.3)
        
        # Plot rolling volatility
        axes[1].plot(rolling_vol.index, rolling_vol, color='orange')
        axes[1].set_title(f'Rolling {window}-Day Volatility (Annualized)')
        axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        axes[1].grid(True, alpha=0.3)
        
        # Plot rolling Sharpe ratio
        axes[2].plot(rolling_sharpe.index, rolling_sharpe, color='green')
        axes[2].set_title(f'Rolling {window}-Day Sharpe Ratio')
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[2].grid(True, alpha=0.3)
        
        # Plot rolling max drawdown
        axes[3].fill_between(rolling_dd.index, 0, rolling_dd, color='red', alpha=0.3)
        axes[3].plot(rolling_dd.index, rolling_dd, color='red')
        axes[3].set_title(f'Rolling {window}-Day Maximum Drawdown')
        axes[3].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        axes[3].grid(True, alpha=0.3)
        
        # Format x-axis dates
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_returns_distribution(returns: pd.Series, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot the distribution of returns with normal distribution overlay.
        
        Args:
            returns: Series of returns to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram of returns
        sns.histplot(returns, kde=True, ax=ax, stat='density', alpha=0.6)
        
        # Plot normal distribution for comparison
        x = np.linspace(returns.min(), returns.max(), 100)
        mu = returns.mean()
        sigma = returns.std()
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
        ax.plot(x, y, 'r--', linewidth=2, label='Normal Distribution')
        
        # Add vertical lines for key metrics
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Zero')
        ax.axvline(x=mu, color='green', linestyle='-', alpha=0.5, label='Mean')
        ax.axvline(x=returns.quantile(0.05), color='red', linestyle='-', alpha=0.5, label='5% VaR')
        
        # Formatting
        ax.set_title('Distribution of Returns', fontsize=14)
        ax.set_xlabel('Return')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Show more information as text
        info_text = (
            f"Mean: {mu:.4f}\n"
            f"Std Dev: {sigma:.4f}\n"
            f"Skewness: {returns.skew():.4f}\n"
            f"Kurtosis: {returns.kurtosis():.4f}\n"
            f"5% VaR: {returns.quantile(0.05):.4f}"
        )
        
        # Position the text in the upper right corner
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        return fig


class MarketVisualization:
    """
    Visualization tools for market data analysis.
    """
    
    @staticmethod
    def plot_price_chart(data: pd.DataFrame, volume: bool = True, 
                        moving_averages: List[int] = None,
                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot a price chart with optional volume and moving averages.
        
        Args:
            data: DataFrame with OHLC data
            volume: Whether to include volume subplot
            moving_averages: List of periods for moving averages
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if moving_averages is None:
            moving_averages = [20, 50, 200]
        
        # Create figure
        if volume:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                          gridspec_kw={'height_ratios': [3, 1]}, 
                                          sharex=True)
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
        
        # Plot price
        ax1.plot(data.index, data['close'], label='Close Price')
        
        # Add moving averages
        for ma in moving_averages:
            if len(data) > ma:
                ma_series = data['close'].rolling(ma).mean()
                ax1.plot(data.index, ma_series, label=f'{ma}-day MA')
        
        # Formatting for price chart
        ax1.set_title('Price Chart', fontsize=14)
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Format y-axis with appropriate scale
        if data['close'].mean() > 10:
            ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
        else:
            ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.2f}'))
        
        # Add volume subplot if requested
        if volume and 'volume' in data.columns:
            # Plot volume bars
            ax2.bar(data.index, data['volume'], color='blue', alpha=0.5)
            
            # Calculate and plot volume moving average
            volume_ma = data['volume'].rolling(20).mean()
            ax2.plot(data.index, volume_ma, color='red', alpha=0.8)
            
            # Formatting for volume chart
            ax2.set_ylabel('Volume')
            ax2.grid(True, alpha=0.3)
            
            # Format y-axis with appropriate scale
            max_vol = data['volume'].max()
            if max_vol > 1e6:
                ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}M').format_data_short)
            else:
                ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        
        # Format x-axis dates
        if volume:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
        else:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_correlation_matrix(returns_df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot a correlation matrix of asset returns.
        
        Args:
            returns_df: DataFrame of asset returns (each column is an asset)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            cmap=cmap, 
            vmax=1, 
            vmin=-1, 
            center=0,
            annot=True, 
            fmt=".2f", 
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            ax=ax
        )
        
        # Formatting
        ax.set_title('Correlation Matrix', fontsize=14)
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_interactive_candlestick(data: pd.DataFrame, moving_averages: List[int] = None,
                                   indicators: Dict[str, pd.Series] = None) -> go.Figure:
        """
        Create an interactive candlestick chart using Plotly.
        
        Args:
            data: DataFrame with OHLC data
            moving_averages: List of periods for moving averages
            indicators: Dictionary of indicator name to Series
            
        Returns:
            Plotly figure
        """
        if moving_averages is None:
            moving_averages = [20, 50, 200]
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        for ma in moving_averages:
            if len(data) > ma:
                ma_series = data['close'].rolling(ma).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma_series,
                        name=f'{ma}-day MA',
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        
        # Add additional indicators
        if indicators:
            for name, series in indicators.items():
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=series,
                        name=name,
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        
        # Add volume trace
        if 'volume' in data.columns:
            # Color the volume bars based on price change
            colors = ['green' if data['close'][i] > data['open'][i] else 'red' 
                     for i in range(len(data))]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    marker=dict(color=colors, opacity=0.5)
                ),
                row=2, col=1
            )
            
            # Add volume moving average
            vol_ma = data['volume'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=vol_ma,
                    name='Volume MA',
                    line=dict(color='blue', width=1)
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Interactive Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=800,
            width=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig 