from typing import List, Dict, Any
import pandas as pd

from .strategy import Strategy, Signal


class MovingAverageCrossoverStrategy(Strategy):
    """
    Strategy that generates signals based on moving average crossovers.
    
    Generates a buy signal when the short-term moving average crosses above
    the long-term moving average, and a sell signal when it crosses below.
    """
    
    def __init__(self, name: str = "MA Crossover", short_window: int = 20, long_window: int = 50):
        """
        Initialize the Moving Average Crossover strategy.
        
        Args:
            name: The name of the strategy
            short_window: The window for the short-term moving average
            long_window: The window for the long-term moving average
        """
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
        self.position = 0  # No position
    
    def initialize(self) -> None:
        """Initialize the strategy."""
        self.position = 0
        self.signals = []
    
    def process_data(self, data: pd.DataFrame) -> None:
        """
        Process market data by calculating moving averages.
        
        Args:
            data: DataFrame with market data
        """
        if data.empty:
            return
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure we have a 'close' column
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have a 'close' column")
        
        # Calculate moving averages
        df[f'short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df[f'long_ma'] = df['close'].rolling(window=self.long_window).mean()
        
        # Calculate crossover signal (1 for buy, -1 for sell, 0 for hold)
        df['signal'] = 0
        df['signal'] = ((df['short_ma'] > df['long_ma']) & (df['short_ma'].shift(1) <= df['long_ma'].shift(1))).astype(int)
        df['signal'] = df['signal'] - ((df['short_ma'] < df['long_ma']) & (df['short_ma'].shift(1) >= df['long_ma'].shift(1))).astype(int)
        
        # Save the processed data
        self.data = df
    
    def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals based on moving average crossovers.
        
        Returns:
            List of trading signals
        """
        if self.data is None or self.data.empty:
            return []
        
        # Get the symbol from the data (assuming it's in the columns)
        symbol = self.data['symbol'].iloc[0] if 'symbol' in self.data.columns else 'UNKNOWN'
        
        # Clear existing signals
        self.signals = []
        
        # Generate signals for each crossover
        for idx, row in self.data.iterrows():
            if pd.isna(row['signal']) or row['signal'] == 0:
                continue
            
            if row['signal'] == 1:  # Buy signal
                signal = Signal(
                    symbol=symbol,
                    timestamp=idx,
                    signal_type='entry',
                    direction='buy',
                    strength=1.0,
                    metadata={
                        'close': row['close'],
                        'short_ma': row['short_ma'],
                        'long_ma': row['long_ma']
                    }
                )
                self.signals.append(signal)
                self.position = 1
            
            elif row['signal'] == -1:  # Sell signal
                signal = Signal(
                    symbol=symbol,
                    timestamp=idx,
                    signal_type='exit',
                    direction='sell',
                    strength=1.0,
                    metadata={
                        'close': row['close'],
                        'short_ma': row['short_ma'],
                        'long_ma': row['long_ma']
                    }
                )
                self.signals.append(signal)
                self.position = -1
        
        return self.signals 