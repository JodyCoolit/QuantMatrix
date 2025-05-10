from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    """Order types supported by brokers."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides (buy/sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Possible statuses for an order."""
    CREATED = "created"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


class Order:
    """Class representing a trading order."""
    
    def __init__(self, symbol: str, order_type: OrderType, side: OrderSide, 
               quantity: float, price: Optional[float] = None, 
               stop_price: Optional[float] = None):
        """
        Initialize an order.
        
        Args:
            symbol: The ticker symbol or asset identifier
            order_type: Type of order (market, limit, etc.)
            side: Buy or sell
            quantity: Amount to buy/sell
            price: Limit price (required for limit and stop-limit orders)
            stop_price: Stop price (required for stop and stop-limit orders)
        """
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        
        self.status = OrderStatus.CREATED
        self.id = None  # Will be set by broker
        self.filled_quantity = 0.0
        self.filled_price = None
        self.created_at = datetime.now()
        self.filled_at = None
    
    def __repr__(self) -> str:
        return (f"Order(id={self.id}, symbol={self.symbol}, "
                f"type={self.order_type.value}, side={self.side.value}, "
                f"quantity={self.quantity}, price={self.price}, "
                f"status={self.status.value})")


class OrderResponse:
    """Response from a broker after placing an order."""
    
    def __init__(self, success: bool, order_id: Optional[str] = None, 
               message: Optional[str] = None, error: Optional[str] = None):
        """
        Initialize an order response.
        
        Args:
            success: Whether the order was successfully placed
            order_id: The broker's order ID if successful
            message: Optional message
            error: Error message if not successful
        """
        self.success = success
        self.order_id = order_id
        self.message = message
        self.error = error
    
    def __repr__(self) -> str:
        if self.success:
            return f"OrderResponse(success=True, order_id={self.order_id})"
        else:
            return f"OrderResponse(success=False, error={self.error})"


class Position:
    """Class representing a trading position."""
    
    def __init__(self, symbol: str, quantity: float, entry_price: float, 
               current_price: Optional[float] = None):
        """
        Initialize a position.
        
        Args:
            symbol: The ticker symbol or asset identifier
            quantity: Current position size (positive for long, negative for short)
            entry_price: Average entry price
            current_price: Current market price
        """
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.current_price = current_price
        
        # Calculate metrics
        self.cost_basis = abs(quantity) * entry_price
        self.market_value = abs(quantity) * (current_price or entry_price)
        self.unrealized_pnl = (current_price - entry_price) * quantity if current_price else 0
    
    def __repr__(self) -> str:
        return (f"Position(symbol={self.symbol}, quantity={self.quantity}, "
                f"entry_price={self.entry_price}, current_price={self.current_price})")


class AccountInfo:
    """Class representing account information from a broker."""
    
    def __init__(self, account_id: str, cash: float, equity: float):
        """
        Initialize account information.
        
        Args:
            account_id: The broker's account ID
            cash: Available cash balance
            equity: Total account value including positions
        """
        self.account_id = account_id
        self.cash = cash
        self.equity = equity
        self.buying_power = cash  # Simplified, in reality depends on broker and account type
    
    def __repr__(self) -> str:
        return f"AccountInfo(id={self.account_id}, cash={self.cash}, equity={self.equity})"


class Broker(ABC):
    """Abstract base class for all broker integrations."""
    
    @abstractmethod
    def place_order(self, order: Order) -> OrderResponse:
        """
        Place an order with the broker.
        
        Args:
            order: The order to place
            
        Returns:
            OrderResponse with result
        """
        pass
    
    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Get account information from the broker.
        
        Returns:
            AccountInfo with account details
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get current positions from the broker.
        
        Returns:
            List of current positions
        """
        pass
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: The broker's order ID to cancel
            
        Returns:
            Whether the cancellation was successful
        """
        pass
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get the current status of an order.
        
        Args:
            order_id: The broker's order ID to check
            
        Returns:
            Updated Order object or None if not found
        """
        pass 