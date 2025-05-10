from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

from ..execution.broker import Position


class Dashboard:
    """
    Dashboard for displaying system status and performance metrics.
    """
    
    def __init__(self, title: str = "QuantMatrix Trading Dashboard"):
        """
        Initialize the dashboard.
        
        Args:
            title: Dashboard title
        """
        self.title = title
        self.metrics: Dict[str, Any] = {}
        self.figures = {}
        self.update_time = datetime.now()
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update dashboard metrics.
        
        Args:
            metrics: Dictionary of metrics to update
        """
        self.metrics.update(metrics)
        self.update_time = datetime.now()
    
    def display_performance_chart(self, equity_curve: pd.DataFrame, 
                                benchmark_data: Optional[pd.DataFrame] = None,
                                figure_id: str = "performance") -> None:
        """
        Create and display a performance chart.
        
        Args:
            equity_curve: DataFrame with equity curve data
            benchmark_data: Optional DataFrame with benchmark data
            figure_id: ID for the figure
        """
        if not isinstance(equity_curve, pd.DataFrame) or equity_curve.empty:
            return
        
        # Create figure
        fig = go.Figure()
        
        # Add strategy equity curve
        if 'equity' in equity_curve.columns:
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve['equity'],
                mode='lines',
                name='Strategy Equity'
            ))
        
        # Add benchmark if provided
        if benchmark_data is not None and not benchmark_data.empty:
            # Normalize benchmark to start at the same value as strategy
            if 'close' in benchmark_data.columns and 'equity' in equity_curve.columns:
                start_equity = equity_curve['equity'].iloc[0]
                benchmark_returns = benchmark_data['close'] / benchmark_data['close'].iloc[0]
                benchmark_equity = start_equity * benchmark_returns
                
                fig.add_trace(go.Scatter(
                    x=benchmark_data.index,
                    y=benchmark_equity,
                    mode='lines',
                    name='Benchmark',
                    line=dict(dash='dash')
                ))
        
        # Update layout
        fig.update_layout(
            title="Strategy Performance",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            legend_title="Legend",
            template="plotly_white"
        )
        
        # Store the figure
        self.figures[figure_id] = fig
    
    def display_positions_table(self, positions: List[Position], figure_id: str = "positions") -> None:
        """
        Create and display a positions table.
        
        Args:
            positions: List of current positions
            figure_id: ID for the figure
        """
        if not positions:
            return
        
        # Create DataFrame from positions
        data = []
        for position in positions:
            data.append({
                'Symbol': position.symbol,
                'Quantity': position.quantity,
                'Entry Price': position.entry_price,
                'Current Price': position.current_price,
                'Market Value': position.market_value,
                'Unrealized P&L': position.unrealized_pnl,
                'Unrealized P&L %': 100 * position.unrealized_pnl / position.cost_basis if position.cost_basis > 0 else 0
            })
        
        df = pd.DataFrame(data)
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill_color='lavender',
                align='left',
                format=[None, None, '.2f', '.2f', '.2f', '.2f', '.2f%']
            )
        )])
        
        fig.update_layout(
            title="Current Positions",
            height=len(data) * 30 + 100,  # Adjust height based on number of positions
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        # Store the figure
        self.figures[figure_id] = fig
    
    def display_metrics_summary(self, figure_id: str = "metrics") -> None:
        """
        Create and display a summary of key metrics.
        
        Args:
            figure_id: ID for the figure
        """
        if not self.metrics:
            return
        
        # Extract metrics for display
        metrics_to_show = {
            'Total Return': f"{self.metrics.get('total_return', 0) * 100:.2f}%",
            'Sharpe Ratio': f"{self.metrics.get('sharpe_ratio', 0):.2f}",
            'Max Drawdown': f"{self.metrics.get('max_drawdown', 0) * 100:.2f}%",
            'Win Rate': f"{self.metrics.get('win_rate', 0) * 100:.2f}%",
            'Account Value': f"${self.metrics.get('account_value', 0):,.2f}",
            'Cash Balance': f"${self.metrics.get('cash_balance', 0):,.2f}",
            'Position Count': str(self.metrics.get('position_count', 0)),
            'Last Update': self.update_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create a DataFrame for the table
        df = pd.DataFrame({
            'Metric': list(metrics_to_show.keys()),
            'Value': list(metrics_to_show.values())
        })
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[df['Metric'], df['Value']],
                fill_color='lavender',
                align=['left', 'right']
            )
        )])
        
        fig.update_layout(
            title="Key Metrics",
            height=len(metrics_to_show) * 30 + 100,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        # Store the figure
        self.figures[figure_id] = fig
    
    def display_trade_history_chart(self, trades: List, figure_id: str = "trades") -> None:
        """
        Create and display a chart of trade history.
        
        Args:
            trades: List of trade objects
            figure_id: ID for the figure
        """
        if not trades:
            return
        
        # Create DataFrame from trades
        data = []
        for trade in trades:
            if hasattr(trade, 'exit_time') and trade.exit_time:
                data.append({
                    'Symbol': trade.symbol,
                    'Side': trade.side.value if hasattr(trade.side, 'value') else trade.side,
                    'Entry Time': trade.entry_time,
                    'Exit Time': trade.exit_time,
                    'Entry Price': trade.entry_price,
                    'Exit Price': trade.exit_price,
                    'P&L': trade.calculate_pnl() if hasattr(trade, 'calculate_pnl') else trade.pnl
                })
        
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        # Create scatter plot of trades
        fig = px.scatter(
            df,
            x='Exit Time',
            y='P&L',
            color='Symbol',
            size=abs(df['P&L']),
            hover_data=['Side', 'Entry Price', 'Exit Price'],
            title="Trade History"
        )
        
        fig.update_layout(
            xaxis_title="Exit Time",
            yaxis_title="Profit/Loss ($)",
            legend_title="Symbol",
            template="plotly_white"
        )
        
        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=df['Exit Time'].min(),
            y0=0,
            x1=df['Exit Time'].max(),
            y1=0,
            line=dict(color="black", width=2, dash="dash")
        )
        
        # Store the figure
        self.figures[figure_id] = fig
    
    def get_figure(self, figure_id: str) -> Optional[Any]:
        """
        Get a specific figure by ID.
        
        Args:
            figure_id: ID of the figure to get
            
        Returns:
            The figure if found, None otherwise
        """
        return self.figures.get(figure_id)
    
    def show(self) -> None:
        """Show all dashboard figures."""
        for fig_id, fig in self.figures.items():
            fig.show()
    
    def save_figure(self, figure_id: str, filename: str) -> bool:
        """
        Save a specific figure to a file.
        
        Args:
            figure_id: ID of the figure to save
            filename: Path to save the figure to
            
        Returns:
            Whether the save was successful
        """
        if figure_id not in self.figures:
            return False
        
        try:
            self.figures[figure_id].write_image(filename)
            return True
        except Exception as e:
            print(f"Error saving figure: {e}")
            return False 