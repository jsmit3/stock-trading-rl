"""
action/interpreter.py

Interprets actions from the agent into trading decisions.

This module converts raw action values from the agent into
meaningful trading parameters like position size and stop-loss levels.

The action space consists of 3 dimensions:
1. Position Size [0,1]: Controls the percentage of allowed capital to deploy
2. Risk Parameters [0,1] × [0,1]: Controls stop-loss and take-profit settings
3. Exit Signal [0,1]: Controls whether to exit an existing position

Author: [Your Name]
Date: March 10, 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple


class ActionInterpreter:
    """
    Interprets raw actions from the agent into concrete trading decisions.
    
    This class handles:
    - Position sizing with volatility adjustment
    - Stop loss and take profit settings with appropriate scaling
    - Exit signal interpretation
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.25,
        min_position_pct: float = 0.05,
        stop_loss_min: float = 0.01,
        stop_loss_max: float = 0.10,
        take_profit_scaling: str = 'exponential'
    ):
        """
        Initialize the action interpreter.
        
        Args:
            max_position_pct: Maximum position size as percentage of capital
            min_position_pct: Minimum position size as percentage of capital
            stop_loss_min: Minimum stop loss percentage
            stop_loss_max: Maximum stop loss percentage
            take_profit_scaling: Type of scaling for take profit ('linear' or 'exponential')
        """
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.stop_loss_min = stop_loss_min
        self.stop_loss_max = stop_loss_max
        self.take_profit_scaling = take_profit_scaling
    
    def interpret_action(
        self, 
        action: np.ndarray, 
        market_data: pd.Series
    ) -> Tuple[float, float, float, float]:
        """
        Interpret the raw action values into concrete trading parameters.
        
        Args:
            action: Raw action from the agent [position_size, stop_loss, take_profit, exit_signal]
            market_data: Current market data including volatility metrics
        
        Returns:
            position_size_pct: Position size as percentage of available capital
            stop_loss_pct: Stop loss percentage below entry price
            take_profit_pct: Take profit percentage above entry price
            exit_signal: Signal to exit position (>0.5 means exit)
        """
        # Ensure action is within bounds [0, 1]
        action = np.clip(action, 0, 1)
        
        # Extract action components
        position_size_raw = action[0]
        stop_loss_raw = action[1]
        take_profit_raw = action[2]
        exit_signal = action[3]
        
        # Interpret position size (ensure it's positive)
        position_size_pct = self._interpret_position_size(position_size_raw)
        
        # Interpret stop loss
        stop_loss_pct = self._interpret_stop_loss(stop_loss_raw)
        
        # Interpret take profit
        take_profit_pct = self._interpret_take_profit(take_profit_raw)
        
        # Adjust position size based on stop loss
        position_size_pct = self._adjust_position_size_for_risk(
            position_size_pct, 
            stop_loss_pct, 
            market_data
        )
        
        # Ensure position size is positive
        position_size_pct = max(0, position_size_pct)
        
        return position_size_pct, stop_loss_pct, take_profit_pct, exit_signal
    
    def _interpret_position_size(self, position_size_raw: float) -> float:
        """
        Interpret the raw position size value.
        
        Args:
            position_size_raw: Raw position size action [0, 1]
            
        Returns:
            position_size_pct: Position size as percentage of available capital
        """
        # Linear scaling between min and max position percentages
        position_size_pct = self.min_position_pct + position_size_raw * (self.max_position_pct - self.min_position_pct)
        return max(0.0, position_size_pct)  # Ensure it's non-negative
    
    def _interpret_stop_loss(self, stop_loss_raw: float) -> float:
        """
        Interpret the raw stop loss value.
        
        Args:
            stop_loss_raw: Raw stop loss action [0, 1]
            
        Returns:
            stop_loss_pct: Stop loss percentage below entry
        """
        # Linear scaling between min and max stop loss percentages
        stop_loss_pct = self.stop_loss_min + stop_loss_raw * (self.stop_loss_max - self.stop_loss_min)
        return max(0.005, stop_loss_pct)  # Minimum 0.5% stop loss
    
    def _interpret_take_profit(self, take_profit_raw: float) -> float:
        """
        Interpret the raw take profit value.
        
        Args:
            take_profit_raw: Raw take profit action [0, 1]
            
        Returns:
            take_profit_pct: Take profit percentage above entry
        """
        if self.take_profit_scaling == 'linear':
            # Linear scaling (1% to 30%)
            take_profit_pct = 0.01 + take_profit_raw * 0.29
        else:
            # Exponential scaling as specified in the project brief
            # 0 → 1%, 0.25 → 3%, 0.5 → 7%, 0.75 → 15%, 1.0 → 30%
            if take_profit_raw < 0.25:
                # Map [0, 0.25] to [0.01, 0.03]
                take_profit_pct = 0.01 + (take_profit_raw / 0.25) * 0.02
            elif take_profit_raw < 0.5:
                # Map [0.25, 0.5] to [0.03, 0.07]
                take_profit_pct = 0.03 + ((take_profit_raw - 0.25) / 0.25) * 0.04
            elif take_profit_raw < 0.75:
                # Map [0.5, 0.75] to [0.07, 0.15]
                take_profit_pct = 0.07 + ((take_profit_raw - 0.5) / 0.25) * 0.08
            else:
                # Map [0.75, 1.0] to [0.15, 0.3]
                take_profit_pct = 0.15 + ((take_profit_raw - 0.75) / 0.25) * 0.15
                
        return max(0.005, take_profit_pct)  # Minimum 0.5% take profit
    
    def _adjust_position_size_for_risk(
        self, 
        position_size_pct: float, 
        stop_loss_pct: float, 
        market_data: pd.Series
    ) -> float:
        """
        Adjust position size based on volatility and risk parameters.
        
        Follows the risk-adjusted position sizing formula:
        Position Size = (Capital × 0.02) ÷ (Entry Price × Stop Distance)
        
        Args:
            position_size_pct: Initial position size percentage
            stop_loss_pct: Stop loss percentage
            market_data: Market data including volatility metrics
            
        Returns:
            adjusted_position_size_pct: Risk-adjusted position size percentage
        """
        # Ensure we have valid inputs before adjusting
        if position_size_pct <= 0 or stop_loss_pct <= 0:
            return 0.0
            
        # Get volatility information
        volatility = market_data.get('atr_pct', market_data.get('volatility', 0.01))
        
        # Calculate risk-adjusted position size based on 2% risk per trade
        # Formula: Position Size = (Capital × 0.02) ÷ (Entry Price × Stop Distance)
        # We're just calculating the adjustment factor here, not the actual shares
        risk_factor = 0.02 / max(stop_loss_pct, 0.005)
        
        # Volatility adjustment - reduce position size in higher volatility
        volatility_factor = 1.0 / max(1.0, volatility / 0.01)
        
        # Apply adjustments
        adjusted_position_size_pct = position_size_pct * min(risk_factor, 1.0) * min(volatility_factor, 1.0)
        
        # Ensure position size stays within bounds
        adjusted_position_size_pct = max(
            min(adjusted_position_size_pct, self.max_position_pct),
            self.min_position_pct if position_size_pct > 0 else 0
        )
        
        return adjusted_position_size_pct
    
    def get_random_action(self) -> np.ndarray:
        """
        Generate a random action for exploration or testing.
        
        Returns:
            action: Random action array
        """
        # Generate random components with better distribution for stable behavior
        position_size = np.random.random()
        stop_loss = np.random.random() * 0.7 + 0.15  # Between 0.15 and 0.85
        take_profit = np.random.random() * 0.7 + 0.25  # Between 0.25 and 0.95
        exit_signal = np.random.random() * 0.4  # Biased toward staying in position
        
        return np.array([position_size, stop_loss, take_profit, exit_signal], dtype=np.float32)