"""
env/state.py

Tracks the state of the stock trading environment.

This module manages all state variables including:
- Portfolio and cash values
- Current positions and P&L
- Trading history
- Position characteristics (entry points, stops, etc.)

Author: [Your Name]
Date: March 10, 2025
"""

import numpy as np
from typing import List, Dict, Any, Optional


class EnvironmentState:
    """
    Tracks the state of the trading environment.
    
    This class manages all state variables and provides methods
    to update them as trading progresses.
    """
    
    def __init__(self, initial_capital: float, window_size: int):
        """
        Initialize the environment state.
        
        Args:
            initial_capital: Starting capital amount
            window_size: Observation window size
        """
        self.initial_capital = initial_capital
        self.window_size = window_size
        
        # Initialize state variables
        self.reset()
    
    def reset(self, start_idx: Optional[int] = None, initial_capital: Optional[float] = None):
        """
        Reset the state to start a new episode.
        
        Args:
            start_idx: Starting index (time step) for the episode
            initial_capital: Optional override for initial capital
        """
        # If new initial capital is provided, use it
        if initial_capital is not None:
            self.initial_capital = initial_capital
        
        # Financial state
        self.cash_balance = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        self.drawdown = 0.0
        
        # Position state
        self.current_position = 0.0
        self.position_entry_price = 0.0
        self.position_entry_step = 0
        self.stop_loss_pct = 0.0
        self.take_profit_pct = 0.0
        self.position_pnl = 0.0
        self.total_pnl = 0.0
        
        # Set current step if provided
        if start_idx is not None:
            self.current_step = start_idx
        
        # Reset tracking variables for this episode
        self.returns_history: List[float] = []
        self.positions_history: List[float] = []
        self.actions_history: List[np.ndarray] = []
        self.rewards_history: List[float] = []
        self.trade_history: List[Dict[str, Any]] = []
    
    def update_portfolio_value(self, current_price: float) -> None:
        """
        Update the portfolio value and related metrics.
        
        Args:
            current_price: Current price of the asset
        """
        # Calculate total portfolio value
        holdings_value = self.current_position * current_price
        self.portfolio_value = self.cash_balance + holdings_value
        
        # Update maximum portfolio value
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
            
        # Update drawdown
        if self.max_portfolio_value > 0:
            self.drawdown = 1 - (self.portfolio_value / self.max_portfolio_value)
            
        # Record portfolio value
        self.returns_history.append(self.portfolio_value)