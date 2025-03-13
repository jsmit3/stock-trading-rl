"""
observation/generator.py

Generates observations for the trading agent.

This module creates comprehensive state observations with features for:
- Price history
- Volume data
- Technical indicators
- Position information
- Account status
- Time information

Author: [Your Name]
Date: March 13, 2025
"""

import numpy as np
import pandas as pd
import traceback
from typing import List, Dict, Optional, Tuple, Union
import random


class ObservationGenerator:
    """
    Generates comprehensive state observations for the trading agent.
    
    This class extracts useful features from market data and position
    information, normalizes them, and assembles them into a single
    observation vector.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        include_sentiment: bool = False,
        use_pca: bool = False,
        max_features: int = 50,
        include_market_context: bool = False,
        fixed_dim: Optional[int] = None
    ):
        """
        Initialize the observation generator.
        
        Args:
            window_size: Number of past days to include in observation
            include_sentiment: Whether to include sentiment features
            use_pca: Whether to use PCA to reduce dimensionality
            max_features: Maximum number of features after PCA
            include_market_context: Whether to include market index context
            fixed_dim: Fixed dimension to use for all observations (default: None)
        """
        self.window_size = window_size
        self.include_sentiment = include_sentiment
        self.use_pca = use_pca
        self.max_features = max_features
        self.include_market_context = include_market_context
        
        # Set observation dimension to fixed size if provided
        self.observation_dim = fixed_dim
        if fixed_dim:
            print(f"ObservationGenerator initialized with fixed dimension: {fixed_dim}")
        
        # Feature flags to enable/disable feature groups
        self.feature_flags = {
            'price_data': True,
            'volume_data': True,
            'trend_indicators': True,
            'momentum_indicators': True,
            'volatility_indicators': True,
            'volume_indicators': True,
            'position_info': True,
            'account_status': True,
            'time_features': True,
            'market_context': include_market_context
        }
    
    def generate_observation(
        self,
        market_data: pd.DataFrame,
        current_step: int,
        current_position: float,
        position_entry_price: float,
        position_entry_step: int,
        stop_loss_pct: float,
        take_profit_pct: float,
        portfolio_value: float,
        initial_capital: float,
        cash_balance: float,
        max_portfolio_value: float,
        drawdown: float,
        market_index_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Generate observation vector with consistent dimensions.
        
        Args:
            market_data: Market data for observation window
            current_step: Current time step
            current_position: Current position size
            position_entry_price: Entry price of current position
            position_entry_step: Step at which position was entered
            stop_loss_pct: Current stop loss percentage
            take_profit_pct: Current take profit percentage
            portfolio_value: Current portfolio value
            initial_capital: Initial capital
            cash_balance: Current cash balance
            max_portfolio_value: Maximum portfolio value
            drawdown: Current drawdown
            market_index_data: Optional market index data
            
        Returns:
            observation: Observation vector with consistent dimensions
        """
        try:
            # If observation_dim is not set, calculate expected dimension
            if self.observation_dim is None:
                self.observation_dim = self._calculate_expected_dimension()
            
            # Create a fixed-size observation of zeros
            fixed_observation = np.zeros(self.observation_dim, dtype=np.float32)
            
            # Extract features in small chunks that we can control
            features_used = 0
            
            # Get latest data point
            latest_data = market_data.iloc[-1]
            
            # Only include features up to our dimension limit
            max_features = self.observation_dim - 20  # Reserve some space for position and account features
            
            # 1. Price features (close prices)
            if self.feature_flags['price_data'] and features_used < max_features:
                # Take only the last 20 close prices (or fewer if not available)
                close_prices = market_data['close'].values[-min(20, len(market_data)):]
                # Normalize by the last price
                last_price = close_prices[-1]
                if last_price > 0:
                    normalized_prices = close_prices / last_price - 1.0
                else:
                    normalized_prices = close_prices
                
                # Copy into fixed observation array (up to available space)
                features_to_use = min(len(normalized_prices), max_features - features_used)
                fixed_observation[features_used:features_used + features_to_use] = normalized_prices[-features_to_use:]
                features_used += features_to_use
            
            # 2. Basic technical indicators (5 most important ones)
            if features_used < max_features:
                # RSI (if available)
                if 'rsi' in market_data.columns:
                    rsi = market_data['rsi'].values[-1] / 100.0  # Normalize to [0,1]
                    fixed_observation[features_used] = rsi
                    features_used += 1
                
                # MACD (if available)
                if 'macd' in market_data.columns and features_used < max_features:
                    macd = market_data['macd'].values[-1]
                    fixed_observation[features_used] = macd
                    features_used += 1
                
                # Volatility (if available)
                if 'volatility' in market_data.columns and features_used < max_features:
                    vol = market_data['volatility'].values[-1]
                    fixed_observation[features_used] = vol
                    features_used += 1
                
                # Volume ratio (if available)
                if 'volume_ratio' in market_data.columns and features_used < max_features:
                    vol_ratio = market_data['volume_ratio'].values[-1]
                    fixed_observation[features_used] = vol_ratio
                    features_used += 1
                
                # ATR percentage (if available)
                if 'atr_pct' in market_data.columns and features_used < max_features:
                    atr_pct = market_data['atr_pct'].values[-1]
                    fixed_observation[features_used] = atr_pct
                    features_used += 1
            
            # 3. Position Information (always include - important for decision making)
            # Use the last 12 slots for position and account features
            position_start = self.observation_dim - 12
            
            # 3.1 Normalized position size (0 to 1)
            fixed_observation[position_start] = min(current_position / 100, 1.0) if current_position > 0 else 0
            
            # 3.2 Position indicator (1 if has position, 0 otherwise)
            fixed_observation[position_start + 1] = 1.0 if current_position > 0 else 0.0
            
            if current_position > 0 and position_entry_price > 0:
                # Get current close price
                current_price = latest_data['close']
                
                # 3.3 Price change since entry (as %)
                price_change = (current_price - position_entry_price) / position_entry_price
                fixed_observation[position_start + 2] = price_change
                
                # 3.4 Days held (normalized)
                days_held = (current_step - position_entry_step) if current_step >= position_entry_step else 0
                fixed_observation[position_start + 3] = min(days_held / 20, 1.0)  # Normalize to [0, 1] with 20 days as max
                
                # 3.5 Stop loss information
                if stop_loss_pct > 0:
                    stop_price = position_entry_price * (1 - stop_loss_pct)
                    # Distance to stop loss as % of current price
                    stop_distance = (current_price - stop_price) / current_price
                    fixed_observation[position_start + 4] = max(min(stop_distance, 1.0), -1.0)  # Clip to [-1, 1]
                
                # 3.6 Take profit information
                if take_profit_pct > 0:
                    take_profit_price = position_entry_price * (1 + take_profit_pct)
                    # Distance to take profit as % of current price
                    tp_distance = (take_profit_price - current_price) / current_price
                    fixed_observation[position_start + 5] = max(min(tp_distance, 1.0), -1.0)  # Clip to [-1, 1]
            
            # 4. Account Status (always include - important for decision making)
            # 4.1 Total return vs initial capital
            if initial_capital > 0:
                total_return = (portfolio_value / initial_capital) - 1
                fixed_observation[position_start + 6] = max(min(total_return, 1.0), -1.0)  # Clip to [-1, 1]
                
            # 4.2 Cash ratio (proportion of portfolio in cash)
            if portfolio_value > 0:
                cash_ratio = cash_balance / portfolio_value
                fixed_observation[position_start + 7] = min(cash_ratio, 1.0)  # Clip to [0, 1]
                
            # 4.3 Drawdown
            fixed_observation[position_start + 8] = min(drawdown, 1.0)  # Clip to [0, 1]
            
            # 4.4 Proximity to peak (1 - drawdown)
            fixed_observation[position_start + 9] = 1.0 - fixed_observation[position_start + 8]
            
            # 4.5 Relative portfolio size compared to initial capital
            if initial_capital > 0:
                relative_size = portfolio_value / initial_capital
                fixed_observation[position_start + 10] = min(relative_size, 5.0) / 5.0  # Normalize to [0, 1] with 5x as max
            
            return fixed_observation
            
        except Exception as e:
            print(f"Error in generate_observation: {e}")
            traceback.print_exc()
            # Return zeros with the fixed dimension
            return np.zeros(self.observation_dim, dtype=np.float32)
    
    def lock_observation_dimension(self, dim: int) -> None:
        """
        Explicitly lock the observation dimension to a fixed size.
        
        Args:
            dim: The fixed dimension to use for all observations
        """
        if self.observation_dim != dim:
            old_dim = self.observation_dim
            self.observation_dim = dim
            print(f"Observation dimension changed: {old_dim} -> {dim}")
        else:
            print(f"Observation dimension already locked at {dim}")
    
    def _calculate_expected_dimension(self) -> int:
        """
        Calculate the expected dimension of observations when no fixed dim is provided.
        
        Returns:
            Expected observation dimension
        """
        # Typical sizes for each feature group
        dimension_estimates = {
            'price_data': 20,              # Close prices for window
            'technical_indicators': 30,     # Common indicators (RSI, MACD, etc.)
            'position_and_account': 12      # Position and account features
        }
        
        # Calculate total expected dimension with some buffer
        total_dim = sum(dimension_estimates.values()) + 10  # Buffer for flexibility
        
        # Round to a clean number
        return ((total_dim // 5) + 1) * 5  # Round up to next multiple of 5