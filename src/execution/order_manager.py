from typing import Dict, List, Optional
from datetime import datetime

from .broker import Broker, Order, OrderResponse, OrderType, OrderSide, OrderStatus
from .risk_controller import RiskController


class Trade:
    """Class representing a completed trade."""
    
    def __init__(self, order: Order, entry_price: float, exit_price: Optional[float] = None,
               entry_time: datetime = None, exit_time: Optional[datetime] = None,
               pnl: Optional[float] = None):
        """
        Initialize a trade.
        
        Args:
            order: The order that created this trade
            entry_price: Entry price
            exit_price: Exit price (None if position still open)
            entry_time: Time of entry
            exit_time: Time of exit (None if position still open)
            pnl: Profit/loss of the trade
        """
        self.symbol = order.symbol
        self.side = order.side
        self.quantity = order.quantity
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.entry_time = entry_time or datetime.now()
        self.exit_time = exit_time
        self.pnl = pnl
        self.order_id = order.id
    
    def calculate_pnl(self) -> float:
        """Calculate profit/loss for this trade if not already set."""
        if self.pnl is not None:
            return self.pnl
        
        if self.exit_price is None:
            return 0.0
        
        if self.side == OrderSide.BUY:
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        else:
            self.pnl = (self.entry_price - self.exit_price) * self.quantity
        
        return self.pnl
    
    def __repr__(self) -> str:
        return (f"Trade(symbol={self.symbol}, side={self.side.value}, "
                f"qty={self.quantity}, entry={self.entry_price}, exit={self.exit_price}, "
                f"pnl={self.pnl})")


class OrderManager:
    """
    OrderManager is responsible for creating and managing orders.
    """
    
    def __init__(self, broker: Broker, risk_controller: RiskController = None):
        """
        Initialize the OrderManager.
        
        Args:
            broker: A Broker implementation for order execution
            risk_controller: Optional RiskController for risk management
        """
        self.broker = broker
        self.risk_controller = risk_controller or RiskController()
        
        # Track active orders and trades
        self.active_orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
    
    def create_order(self, symbol: str, order_type: OrderType, side: OrderSide,
                   quantity: float, price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> Order:
        """
        Create an order (but don't send it yet).
        
        Args:
            symbol: The ticker symbol
            order_type: Type of order (market, limit, etc.)
            side: Buy or sell
            quantity: Amount to buy/sell
            price: Limit price (required for limit and stop-limit orders)
            stop_price: Stop price (required for stop and stop-limit orders)
            
        Returns:
            Created Order object
        """
        order = Order(
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        return order
    
    def send_order(self, order: Order) -> OrderResponse:
        """
        Send an order to the broker after risk checks.
        
        Args:
            order: The order to send
            
        Returns:
            OrderResponse with result
        """
        # Get account info for risk checks
        account = self.broker.get_account_info()
        
        # Check with risk controller
        if not self.risk_controller.validate_order(order, account):
            return OrderResponse(
                success=False,
                error="Order rejected by risk controller"
            )
        
        # Send the order to the broker
        response = self.broker.place_order(order)
        
        # If successful, track the order
        if response.success and response.order_id:
            order.id = response.order_id
            self.active_orders[response.order_id] = order
        
        return response
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: The order ID to cancel
            
        Returns:
            Whether the cancellation was successful
        """
        success = self.broker.cancel_order(order_id)
        
        # If successful, update our tracking
        if success and order_id in self.active_orders:
            self.active_orders[order_id].status = OrderStatus.CANCELLED
        
        return success
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get the current status of an order.
        
        Args:
            order_id: The order ID to check
            
        Returns:
            Updated Order object or None if not found
        """
        # Try to get from broker (most up-to-date)
        order = self.broker.get_order_status(order_id)
        
        # Update our tracking
        if order and order_id in self.active_orders:
            self.active_orders[order_id] = order
            
            # If order is filled, record as a trade
            if order.status == OrderStatus.FILLED and order.filled_price:
                self._record_trade(order)
                # Remove from active orders
                del self.active_orders[order_id]
        
        return order
    
    def get_active_orders(self) -> List[Order]:
        """
        Get all active orders.
        
        Returns:
            List of active orders
        """
        return list(self.active_orders.values())
    
    def get_trades(self) -> List[Trade]:
        """
        Get all recorded trades.
        
        Returns:
            List of trades
        """
        return self.trades
    
    def update_orders(self) -> None:
        """Update the status of all active orders."""
        for order_id in list(self.active_orders.keys()):
            self.get_order_status(order_id)
    
    def _record_trade(self, order: Order) -> None:
        """
        Record a completed trade.
        
        Args:
            order: The filled order
        """
        if not order.filled_price:
            return
        
        trade = Trade(
            order=order,
            entry_price=order.filled_price,
            entry_time=order.filled_at or datetime.now()
        )
        
        self.trades.append(trade) 