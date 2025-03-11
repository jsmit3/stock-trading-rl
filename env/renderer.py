"""
env/renderer.py

Handles visualization of the trading environment state.

This module provides methods for visualizing the environment
in both console and graphical formats.

Author: [Your Name]
Date: March 10, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Any


class EnvironmentRenderer:
    """
    Handles visualization of the trading environment.
    
    Provides human-readable console output and graphical visualizations
    using matplotlib.
    """
    
    def __init__(self):
        """Initialize the renderer."""
        self.figure = None
    
    def render(self, state, current_step, dates, price_data, mode='human'):
        """
        Render the environment.
        
        Args:
            state: Environment state
            current_step: Current step in the episode
            dates: List of dates
            price_data: Price data DataFrame
            mode: The rendering mode ('human' or 'rgb_array')
            
        Returns:
            None for 'human' mode or image for 'rgb_array' mode
        """
        if mode == 'human':
            return self._render_human(state, current_step, dates)
        elif mode == 'rgb_array':
            return self._render_rgb_array(state, current_step, dates, price_data)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _render_human(self, state, current_step, dates):
        """
        Render the environment as text for console output.
        
        Args:
            state: Environment state
            current_step: Current step in the episode
            dates: List of dates
        """
        print(f"Step: {current_step}, Date: {dates[current_step if current_step < len(dates) else -1]}")
        print(f"Price: ${state.position_entry_price:.2f}")
        print(f"Portfolio Value: ${state.portfolio_value:.2f}")
        print(f"Cash Balance: ${state.cash_balance:.2f}")
        print(f"Position: {state.current_position:.2f} shares")
        print(f"Position P&L: {state.position_pnl:.2f}%")
        print(f"Total P&L: {((state.portfolio_value / state.initial_capital) - 1) * 100:.2f}%")
        print(f"Drawdown: {state.drawdown * 100:.2f}%")
        print("-" * 50)
    
    def _render_rgb_array(self, state, current_step, dates, price_data):
        """
        Render the environment as an RGB image.
        
        Args:
            state: Environment state
            current_step: Current step in the episode
            dates: List of dates
            price_data: Price data DataFrame
            
        Returns:
            image: RGB array representing the rendered environment
        """
        # Create figure if it doesn't exist
        if self.figure is None:
            self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(
                3, 1, figsize=(10, 8), sharex=True
            )
        
        # Define the visible window (last 100 steps or all available steps)
        visible_start = max(0, current_step - 100)
        visible_end = min(current_step + 1, len(dates))
        
        # Extract relevant data
        visible_dates = dates[visible_start:visible_end]
        prices = price_data.iloc[visible_start:visible_end]['close'].values
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot price
        self.ax1.plot(visible_dates, prices, color='blue')
        self.ax1.set_title('Stock Price')
        self.ax1.set_ylabel('Price ($)')
        
        # Plot portfolio value
        portfolio_values = np.array(state.returns_history)[-len(visible_dates):] if len(state.returns_history) > 0 else []
        if len(portfolio_values) > 0:
            self.ax2.plot(visible_dates[-len(portfolio_values):], portfolio_values, color='green')
        self.ax2.set_title('Portfolio Value')
        self.ax2.set_ylabel('Value ($)')
        
        # Plot position sizes
        position_sizes = np.array(state.positions_history)[-len(visible_dates):] if len(state.positions_history) > 0 else []
        if len(position_sizes) > 0:
            self.ax3.bar(visible_dates[-len(position_sizes):], position_sizes, color='orange')
        self.ax3.set_title('Position Size')
        self.ax3.set_ylabel('Shares')
        self.ax3.set_xlabel('Date')
        
        # Format the figure
        plt.tight_layout()
        
        try:
            # Convert the figure to an image
            self.figure.canvas.draw()
            image = np.frombuffer(self.figure.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(self.figure.canvas.get_width_height()[::-1] + (3,))
            return image
        except Exception as e:
            print(f"Error rendering: {e}")
            return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up any resources."""
        plt.close('all')
        self.figure = None