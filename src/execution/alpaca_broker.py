from typing import List, Dict, Any, Optional
import os
from datetime import datetime

import alpaca_trade_api as tradeapi

from .broker import (
    Broker, Order, OrderResponse, Position, AccountInfo,
    OrderType, OrderSide, OrderStatus
)


class AlpacaBroker(Broker):
    """
    Implementation of the Broker interface for Alpaca Markets.
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, base_url: str = None):
        """
        Initialize Alpaca broker connection.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: Alpaca API base URL
        """
        # Use environment variables if not provided
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY')
        self.api_secret = api_secret or os.environ.get('ALPACA_API_SECRET')
        self.base_url = base_url or os.environ.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not provided")
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')
        
        # Cache account info
        self._account = None
    
    def place_order(self, order: Order) -> OrderResponse:
        """
        Place an order with Alpaca.
        
        Args:
            order: The order to place
            
        Returns:
            OrderResponse with result
        """
        try:
            # Map order types from our enum to Alpaca's strings
            order_type_map = {
                OrderType.MARKET: 'market',
                OrderType.LIMIT: 'limit',
                OrderType.STOP: 'stop',
                OrderType.STOP_LIMIT: 'stop_limit'
            }
            
            # Map order sides from our enum to Alpaca's strings
            order_side_map = {
                OrderSide.BUY: 'buy',
                OrderSide.SELL: 'sell'
            }
            
            # Create order parameters
            params = {
                'symbol': order.symbol,
                'qty': str(order.quantity),
                'side': order_side_map[order.side],
                'type': order_type_map[order.order_type],
                'time_in_force': 'gtc'  # Good 'til cancelled
            }
            
            # Add price for limit and stop-limit orders
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if order.price is None:
                    return OrderResponse(
                        success=False,
                        error="Limit price required for limit and stop-limit orders"
                    )
                params['limit_price'] = str(order.price)
            
            # Add stop price for stop and stop-limit orders
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                if order.stop_price is None:
                    return OrderResponse(
                        success=False,
                        error="Stop price required for stop and stop-limit orders"
                    )
                params['stop_price'] = str(order.stop_price)
            
            # Submit the order to Alpaca
            alpaca_order = self.api.submit_order(**params)
            
            # Update our order object
            order.id = alpaca_order.id
            order.status = self._map_alpaca_status(alpaca_order.status)
            
            return OrderResponse(
                success=True,
                order_id=alpaca_order.id,
                message=f"Order placed successfully: {alpaca_order.id}"
            )
        
        except Exception as e:
            return OrderResponse(
                success=False,
                error=f"Error placing order: {str(e)}"
            )
    
    def get_account_info(self) -> AccountInfo:
        """
        Get account information from Alpaca.
        
        Returns:
            AccountInfo with account details
        """
        try:
            account = self.api.get_account()
            self._account = account
            
            return AccountInfo(
                account_id=account.id,
                cash=float(account.cash),
                equity=float(account.equity)
            )
        
        except Exception as e:
            print(f"Error getting account info: {e}")
            
            # Return cached account info if available
            if self._account:
                return AccountInfo(
                    account_id=self._account.id,
                    cash=float(self._account.cash),
                    equity=float(self._account.equity)
                )
            
            # Return default values if no cached info
            return AccountInfo(
                account_id="Unknown",
                cash=0.0,
                equity=0.0
            )
    
    def get_positions(self) -> List[Position]:
        """
        Get current positions from Alpaca.
        
        Returns:
            List of current positions
        """
        try:
            alpaca_positions = self.api.list_positions()
            positions = []
            
            for pos in alpaca_positions:
                positions.append(Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price)
                ))
            
            return positions
        
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order with Alpaca.
        
        Args:
            order_id: The Alpaca order ID to cancel
            
        Returns:
            Whether the cancellation was successful
        """
        try:
            self.api.cancel_order(order_id)
            return True
        
        except Exception as e:
            print(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        Get the current status of an order from Alpaca.
        
        Args:
            order_id: The Alpaca order ID to check
            
        Returns:
            Updated Order object or None if not found
        """
        try:
            alpaca_order = self.api.get_order(order_id)
            
            # Create a new Order object with the current status
            order = Order(
                symbol=alpaca_order.symbol,
                order_type=self._map_to_order_type(alpaca_order.type),
                side=OrderSide.BUY if alpaca_order.side == 'buy' else OrderSide.SELL,
                quantity=float(alpaca_order.qty),
                price=float(alpaca_order.limit_price) if hasattr(alpaca_order, 'limit_price') and alpaca_order.limit_price else None,
                stop_price=float(alpaca_order.stop_price) if hasattr(alpaca_order, 'stop_price') and alpaca_order.stop_price else None
            )
            
            # Update order details
            order.id = alpaca_order.id
            order.status = self._map_alpaca_status(alpaca_order.status)
            order.filled_quantity = float(alpaca_order.filled_qty) if hasattr(alpaca_order, 'filled_qty') else 0.0
            order.filled_price = float(alpaca_order.filled_avg_price) if hasattr(alpaca_order, 'filled_avg_price') and alpaca_order.filled_avg_price else None
            
            # Parse dates
            if hasattr(alpaca_order, 'created_at') and alpaca_order.created_at:
                order.created_at = datetime.fromisoformat(alpaca_order.created_at.replace('Z', '+00:00'))
            
            if hasattr(alpaca_order, 'filled_at') and alpaca_order.filled_at:
                order.filled_at = datetime.fromisoformat(alpaca_order.filled_at.replace('Z', '+00:00'))
            
            return order
        
        except Exception as e:
            print(f"Error getting order status for {order_id}: {e}")
            return None
    
    def _map_alpaca_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca order status to our OrderStatus enum."""
        status_map = {
            'new': OrderStatus.SUBMITTED,
            'accepted': OrderStatus.SUBMITTED,
            'pending_new': OrderStatus.SUBMITTED,
            'accepted_for_bidding': OrderStatus.SUBMITTED,
            'stopped': OrderStatus.SUBMITTED,
            'rejected': OrderStatus.REJECTED,
            'suspended': OrderStatus.REJECTED,
            'calculated': OrderStatus.SUBMITTED,
            'partially_filled': OrderStatus.PARTIAL,
            'filled': OrderStatus.FILLED,
            'done_for_day': OrderStatus.SUBMITTED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.CANCELLED,
            'replaced': OrderStatus.SUBMITTED,
            'pending_cancel': OrderStatus.SUBMITTED,
            'pending_replace': OrderStatus.SUBMITTED
        }
        
        return status_map.get(alpaca_status, OrderStatus.CREATED)
    
    def _map_to_order_type(self, alpaca_type: str) -> OrderType:
        """Map Alpaca order type to our OrderType enum."""
        type_map = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop': OrderType.STOP,
            'stop_limit': OrderType.STOP_LIMIT
        }
        
        return type_map.get(alpaca_type, OrderType.MARKET) 