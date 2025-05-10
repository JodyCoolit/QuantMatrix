import logging
from datetime import datetime
import os
from typing import Any, Dict, Optional

from ..execution.order_manager import Trade


class Logger:
    """
    Logger is responsible for logging system events and trading activity.
    """
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        """
        self.log_dir = log_dir
        self.log_level = log_level
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up system logger
        self.system_logger = logging.getLogger("system")
        self.system_logger.setLevel(log_level)
        
        # Set up trade logger
        self.trade_logger = logging.getLogger("trades")
        self.trade_logger.setLevel(log_level)
        
        # Configure handlers and formatters
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Set up logging handlers and formatters."""
        # Create formatters
        system_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        trade_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create file handlers
        system_file = os.path.join(self.log_dir, "system.log")
        trade_file = os.path.join(self.log_dir, "trades.log")
        
        system_handler = logging.FileHandler(system_file)
        system_handler.setFormatter(system_formatter)
        
        trade_handler = logging.FileHandler(trade_file)
        trade_handler.setFormatter(trade_formatter)
        
        # Add console handler for system logs
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(system_formatter)
        
        # Add handlers to loggers
        self.system_logger.addHandler(system_handler)
        self.system_logger.addHandler(console_handler)
        self.trade_logger.addHandler(trade_handler)
        
        # Avoid duplicate logging
        self.system_logger.propagate = False
        self.trade_logger.propagate = False
    
    def log_info(self, message: str) -> None:
        """
        Log an info message.
        
        Args:
            message: The message to log
        """
        self.system_logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """
        Log a warning message.
        
        Args:
            message: The message to log
        """
        self.system_logger.warning(message)
    
    def log_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """
        Log an error message with optional exception.
        
        Args:
            message: The error message
            exception: Optional exception to include
        """
        if exception:
            self.system_logger.error(f"{message}: {str(exception)}", exc_info=True)
        else:
            self.system_logger.error(message)
    
    def log_debug(self, message: str) -> None:
        """
        Log a debug message.
        
        Args:
            message: The message to log
        """
        self.system_logger.debug(message)
    
    def log_trade(self, trade: Trade) -> None:
        """
        Log a trade event.
        
        Args:
            trade: The trade to log
        """
        trade_msg = (
            f"TRADE: {trade.symbol} | "
            f"Side: {trade.side.value} | "
            f"Qty: {trade.quantity} | "
            f"Entry: {trade.entry_price} | "
            f"Exit: {trade.exit_price or 'OPEN'} | "
            f"P&L: {trade.pnl or 'N/A'}"
        )
        
        self.trade_logger.info(trade_msg)
        self.system_logger.info(f"New trade logged: {trade.symbol}")
    
    def log_strategy(self, strategy_name: str, action: str, details: Dict[str, Any] = None) -> None:
        """
        Log a strategy action.
        
        Args:
            strategy_name: Name of the strategy
            action: The action taken (e.g., 'signal_generated', 'backtest_complete')
            details: Additional details about the action
        """
        msg = f"STRATEGY: {strategy_name} | Action: {action}"
        
        if details:
            detail_strs = [f"{k}={v}" for k, v in details.items()]
            msg += f" | Details: {', '.join(detail_strs)}"
        
        self.system_logger.info(msg) 