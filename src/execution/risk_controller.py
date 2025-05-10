from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .broker import Order, OrderSide, AccountInfo, Position


class RiskMetrics:
    """Class containing risk metrics for a portfolio."""
    
    def __init__(self):
        """Initialize risk metrics."""
        self.total_position_value = 0.0
        self.largest_position_value = 0.0
        self.largest_position_symbol = ""
        self.portfolio_concentration = 0.0  # Largest position as % of total
        self.long_exposure = 0.0
        self.short_exposure = 0.0
        self.net_exposure = 0.0
        self.gross_exposure = 0.0
        self.position_count = 0
    
    def __repr__(self) -> str:
        return (f"RiskMetrics(positions={self.position_count}, "
                f"net_exposure={self.net_exposure:.2f}, "
                f"concentration={self.portfolio_concentration:.2%})")


class RiskController:
    """
    RiskController is responsible for validating orders against risk limits.
    """
    
    def __init__(self, position_limits: Dict[str, float] = None, 
               max_position_size: float = 0.1, max_drawdown: float = 0.05):
        """
        Initialize the RiskController.
        
        Args:
            position_limits: Dictionary of symbol to max position size (as fraction of portfolio)
            max_position_size: Default maximum position size as fraction of portfolio
            max_drawdown: Maximum allowed drawdown before stopping trading
        """
        self.position_limits = position_limits or {}
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        
        # Trading limits
        self.max_orders_per_day = 100
        self.max_daily_loss = 0.02  # 2% of account
        
        # Tracking
        self.daily_orders_count = 0
        self.daily_loss = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Current drawdown
        self.peak_equity = 0.0
        self.current_drawdown = 0.0
    
    def validate_order(self, order: Order, account: AccountInfo) -> bool:
        """
        Validate an order against risk limits.
        
        Args:
            order: The order to validate
            account: Current account information
            
        Returns:
            Whether the order is within risk limits
        """
        # Reset daily limits if it's a new day
        self._reset_daily_limits_if_needed()
        
        # Check daily order count
        if self.daily_orders_count >= self.max_orders_per_day:
            print(f"Risk check failed: Exceeded maximum orders per day ({self.max_orders_per_day})")
            return False
        
        # Check daily loss limit
        if self.daily_loss >= account.equity * self.max_daily_loss:
            print(f"Risk check failed: Exceeded maximum daily loss ({self.max_daily_loss:.2%} of account)")
            return False
        
        # Check drawdown limit
        if self.current_drawdown >= self.max_drawdown:
            print(f"Risk check failed: Exceeded maximum drawdown ({self.max_drawdown:.2%})")
            return False
        
        # Calculate order value
        order_value = order.quantity * (order.price or 0.0)
        
        # If no price specified (market order), we need to estimate it
        if order_value <= 0:
            # In a real system, we would get the current market price
            # For now, we'll use a conservative estimate of account value
            order_value = order.quantity * account.equity / 100  # Rough estimate
        
        # Check position size limit
        position_limit = self.position_limits.get(order.symbol, self.max_position_size)
        max_position_value = account.equity * position_limit
        
        if order_value > max_position_value:
            print(f"Risk check failed: Order value ({order_value:.2f}) exceeds position limit ({max_position_value:.2f})")
            return False
        
        # Check available buying power
        if order.side == OrderSide.BUY and order_value > account.buying_power:
            print(f"Risk check failed: Order value ({order_value:.2f}) exceeds buying power ({account.buying_power:.2f})")
            return False
        
        # All checks passed
        self.daily_orders_count += 1
        return True
    
    def calculate_risk_metrics(self, positions: List[Position], account_equity: float) -> RiskMetrics:
        """
        Calculate risk metrics for the current portfolio.
        
        Args:
            positions: List of current positions
            account_equity: Total account equity
            
        Returns:
            RiskMetrics object
        """
        metrics = RiskMetrics()
        
        if not positions:
            return metrics
        
        # Basic position metrics
        metrics.position_count = len(positions)
        
        # Calculate exposures
        for position in positions:
            position_value = abs(position.quantity * position.current_price)
            metrics.total_position_value += position_value
            
            # Track largest position
            if position_value > metrics.largest_position_value:
                metrics.largest_position_value = position_value
                metrics.largest_position_symbol = position.symbol
            
            # Long/short exposure
            if position.quantity > 0:
                metrics.long_exposure += position_value
            else:
                metrics.short_exposure += position_value
        
        # Calculate derived metrics
        if metrics.total_position_value > 0:
            metrics.portfolio_concentration = metrics.largest_position_value / metrics.total_position_value
        
        if account_equity > 0:
            metrics.net_exposure = (metrics.long_exposure - metrics.short_exposure) / account_equity
            metrics.gross_exposure = (metrics.long_exposure + metrics.short_exposure) / account_equity
        
        return metrics
    
    def update_drawdown(self, current_equity: float) -> float:
        """
        Update drawdown tracking based on current equity.
        
        Args:
            current_equity: Current account equity
            
        Returns:
            Current drawdown as a percentage
        """
        # Update peak equity if we have a new high
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        return self.current_drawdown
    
    def record_trade_pnl(self, pnl: float) -> None:
        """
        Record the P&L of a completed trade for daily tracking.
        
        Args:
            pnl: Profit/loss amount
        """
        # If it's a loss, add to daily loss counter
        if pnl < 0:
            self.daily_loss += abs(pnl)
    
    def _reset_daily_limits_if_needed(self) -> None:
        """Reset daily limits if it's a new day."""
        today = datetime.now().date()
        
        if today > self.last_reset_date:
            self.daily_orders_count = 0
            self.daily_loss = 0.0
            self.last_reset_date = today 