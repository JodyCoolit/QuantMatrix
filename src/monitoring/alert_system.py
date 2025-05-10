from typing import List, Dict, Any, Callable, Optional
from datetime import datetime
import pandas as pd

from .logger import Logger


class AlertCondition:
    """Class representing an alert condition."""
    
    def __init__(self, name: str, condition_fn: Callable[[Dict[str, Any]], bool], 
               message_template: str, severity: str = "info"):
        """
        Initialize an alert condition.
        
        Args:
            name: Name of the alert
            condition_fn: Function that takes a data dictionary and returns True if alert should trigger
            message_template: Template for alert message
            severity: Alert severity (info, warning, error)
        """
        self.name = name
        self.condition_fn = condition_fn
        self.message_template = message_template
        self.severity = severity
        self.last_triggered = None
    
    def check(self, data: Dict[str, Any]) -> bool:
        """
        Check if the alert condition is met.
        
        Args:
            data: Data dictionary to check against
            
        Returns:
            Whether the alert should trigger
        """
        try:
            result = self.condition_fn(data)
            
            if result:
                self.last_triggered = datetime.now()
            
            return result
        except Exception as e:
            print(f"Error checking alert condition '{self.name}': {e}")
            return False
    
    def format_message(self, data: Dict[str, Any]) -> str:
        """
        Format the alert message with data.
        
        Args:
            data: Data dictionary to use for formatting
            
        Returns:
            Formatted alert message
        """
        try:
            return self.message_template.format(**data)
        except Exception as e:
            return f"Alert '{self.name}' triggered (error formatting message: {e})"


class Alert:
    """Class representing a triggered alert."""
    
    def __init__(self, condition: AlertCondition, message: str, triggered_at: datetime = None):
        """
        Initialize an alert.
        
        Args:
            condition: The condition that triggered this alert
            message: Alert message
            triggered_at: When the alert was triggered
        """
        self.condition = condition
        self.message = message
        self.triggered_at = triggered_at or datetime.now()
        self.severity = condition.severity
        self.is_active = True
    
    def acknowledge(self) -> None:
        """Mark the alert as acknowledged."""
        self.is_active = False
    
    def __repr__(self) -> str:
        return f"Alert({self.condition.name}, {self.severity}, {self.triggered_at})"


class AlertSystem:
    """
    AlertSystem is responsible for checking alert conditions and sending notifications.
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the AlertSystem.
        
        Args:
            logger: Optional Logger for logging alerts
        """
        self.conditions: List[AlertCondition] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.logger = logger
    
    def add_alert_condition(self, condition: AlertCondition) -> None:
        """
        Add an alert condition to the system.
        
        Args:
            condition: The alert condition to add
        """
        self.conditions.append(condition)
    
    def check_alerts(self, data: Dict[str, Any]) -> List[Alert]:
        """
        Check all alert conditions against the provided data.
        
        Args:
            data: Data dictionary to check against
            
        Returns:
            List of new alerts that were triggered
        """
        new_alerts = []
        
        for condition in self.conditions:
            if condition.check(data):
                message = condition.format_message(data)
                alert = Alert(condition, message)
                
                self.active_alerts.append(alert)
                self.alert_history.append(alert)
                new_alerts.append(alert)
                
                # Log the alert if a logger is available
                if self.logger:
                    if condition.severity == "error":
                        self.logger.log_error(f"ALERT: {message}")
                    elif condition.severity == "warning":
                        self.logger.log_warning(f"ALERT: {message}")
                    else:
                        self.logger.log_info(f"ALERT: {message}")
        
        return new_alerts
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get all active alerts.
        
        Returns:
            List of active alerts
        """
        return self.active_alerts
    
    def acknowledge_alert(self, alert_idx: int) -> bool:
        """
        Acknowledge an alert by index.
        
        Args:
            alert_idx: Index of the alert in the active alerts list
            
        Returns:
            Whether the acknowledgement was successful
        """
        if 0 <= alert_idx < len(self.active_alerts):
            self.active_alerts[alert_idx].acknowledge()
            self.active_alerts.pop(alert_idx)
            return True
        return False
    
    def send_notification(self, alert: Alert) -> bool:
        """
        Send a notification for an alert.
        
        Args:
            alert: The alert to send notification for
            
        Returns:
            Whether the notification was successfully sent
        """
        # In a real system, this would send an email, SMS, etc.
        # For now, we'll just print the notification
        print(f"NOTIFICATION: {alert.severity.upper()} - {alert.message}")
        
        # Log the notification if a logger is available
        if self.logger:
            self.logger.log_info(f"Notification sent for alert: {alert.message}")
        
        return True
    
    def create_price_alert(self, symbol: str, price_threshold: float, is_above: bool = True) -> AlertCondition:
        """
        Create a price alert condition.
        
        Args:
            symbol: The ticker symbol to monitor
            price_threshold: Price threshold to trigger alert
            is_above: Whether to trigger when price is above threshold (True) or below (False)
            
        Returns:
            Created AlertCondition
        """
        name = f"{symbol} {'above' if is_above else 'below'} {price_threshold}"
        
        def condition_fn(data: Dict[str, Any]) -> bool:
            if 'symbol' not in data or data['symbol'] != symbol:
                return False
            
            price = data.get('price', 0)
            if is_above:
                return price > price_threshold
            else:
                return price < price_threshold
        
        message_template = f"{symbol} price is {{price}} which is {'above' if is_above else 'below'} {price_threshold}"
        
        condition = AlertCondition(name, condition_fn, message_template, "info")
        self.add_alert_condition(condition)
        
        return condition
    
    def create_volume_alert(self, symbol: str, volume_threshold: float) -> AlertCondition:
        """
        Create a volume spike alert condition.
        
        Args:
            symbol: The ticker symbol to monitor
            volume_threshold: Volume threshold to trigger alert
            
        Returns:
            Created AlertCondition
        """
        name = f"{symbol} volume above {volume_threshold}"
        
        def condition_fn(data: Dict[str, Any]) -> bool:
            if 'symbol' not in data or data['symbol'] != symbol:
                return False
            
            volume = data.get('volume', 0)
            return volume > volume_threshold
        
        message_template = f"{symbol} volume is {{volume}} which is above {volume_threshold}"
        
        condition = AlertCondition(name, condition_fn, message_template, "info")
        self.add_alert_condition(condition)
        
        return condition 