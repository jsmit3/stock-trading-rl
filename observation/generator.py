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
        fixed_dim: int = 325
    ):
        """
        Initialize the observation generator.
        
        Args:
            window_size: Number of past days to include in observation
            include_sentiment: Whether to include sentiment features
            use_pca: Whether to use PCA to reduce dimensionality
            max_features: Maximum number of features after PCA
            include_market_context: Whether to include market index context
            fixed_dim: Fixed dimension to use for all observations (default: 325)
        """
        self.window_size = window_size
        self.include_sentiment = include_sentiment
        self.use_pca = use_pca
        self.max_features = max_features
        self.include_market_context = include_market_context
        
        # Lock the observation dimension to a fixed size from the start
        self.observation_dim = fixed_dim
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
        Generate observation vector with forced fixed dimensions.
        
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
        # If observation_dim is not set, use default of 325
        if self.observation_dim is None:
            self.observation_dim = 325
            print(f"Setting default observation dimension to {self.observation_dim}")
        
        try:
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
            # Use the last 7 slots for position features
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
    
    def lock_observation_dimension(self, dim=325):
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
        Calculate a safe fixed dimension for all observations.
        
        Returns:
            Fixed observation dimension (325 by default)
        """
        # We're using a fixed dimension approach, so this just returns the configured value
        return self.observation_dim if self.observation_dim is not None else 325
    
    def _get_price_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract price-related features from market data.
        
        Args:
            market_data: Market data for observation window
            
        Returns:
            price_features: Array of price features
        """
        try:
            # Extract OHLC data
            features = []
            
            # Open prices
            if 'open' in market_data.columns:
                features.append(market_data['open'].values)
                
            # High prices
            if 'high' in market_data.columns:
                features.append(market_data['high'].values)
                
            # Low prices
            if 'low' in market_data.columns:
                features.append(market_data['low'].values)
                
            # Close prices
            if 'close' in market_data.columns:
                features.append(market_data['close'].values)
                
            # Stack features into a 2D array
            if features:
                price_features = np.vstack(features)
                
                # Normalize using the last close price
                last_close = market_data['close'].iloc[-1]
                if last_close > 0:
                    price_features = price_features / last_close - 1.0  # Convert to returns relative to last close
                    
                return price_features
            else:
                # Return empty array with correct shape if no features
                return np.zeros((4, min(len(market_data), self.window_size)))
        except Exception as e:
            print(f"Error in _get_price_features: {e}")
            return np.zeros((4, min(len(market_data), self.window_size)))
    
    def _get_volume_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract volume-related features from market data.
        
        Args:
            market_data: Market data for observation window
            
        Returns:
            volume_features: Array of volume features
        """
        try:
            if 'volume' in market_data.columns:
                # Extract volume
                volume = market_data['volume'].values
                
                # Handle zero or negative values
                volume = np.maximum(volume, 1.0)
                
                # Normalize by diving by the mean
                mean_volume = np.mean(volume)
                if mean_volume > 0:
                    normalized_volume = volume / mean_volume
                else:
                    normalized_volume = volume
                    
                return np.reshape(normalized_volume, (1, -1))
            else:
                # Return empty array with correct shape if no volume data
                return np.zeros((1, min(len(market_data), self.window_size)))
        except Exception as e:
            print(f"Error in _get_volume_features: {e}")
            return np.zeros((1, min(len(market_data), self.window_size)))
    
    def _get_trend_indicators(self, market_data: pd.DataFrame) -> List[np.ndarray]:
        """
        Extract trend indicators from market data.
        
        Args:
            market_data: Market data for observation window
            
        Returns:
            trend_features: List of trend indicator arrays
        """
        trend_features = []
        
        try:
            # SMA-based features
            for col in ['sma_10', 'sma_20', 'sma_50']:
                if col in market_data.columns:
                    # Calculate ratio to close price
                    sma_ratio = np.reshape(market_data[col].values / market_data['close'].values, (1, -1))
                    trend_features.append(sma_ratio)
            
            # EMA-based features
            for col in ['ema_10', 'ema_20', 'ema_50']:
                if col in market_data.columns:
                    # Calculate ratio to close price
                    ema_ratio = np.reshape(market_data[col].values / market_data['close'].values, (1, -1))
                    trend_features.append(ema_ratio)
                    
            # MACD related features
            if 'macd' in market_data.columns and 'macd_signal' in market_data.columns:
                # MACD line
                macd = np.reshape(market_data['macd'].values, (1, -1))
                # MACD signal line
                macd_signal = np.reshape(market_data['macd_signal'].values, (1, -1))
                # MACD histogram
                if 'macd_hist' in market_data.columns:
                    macd_hist = np.reshape(market_data['macd_hist'].values, (1, -1))
                    trend_features.append(macd_hist)
                else:
                    # Calculate histogram if not available
                    macd_hist = np.reshape(market_data['macd'].values - market_data['macd_signal'].values, (1, -1))
                    trend_features.append(macd_hist)
                
                # Add MACD and signal
                trend_features.append(macd)
                trend_features.append(macd_signal)
                
            # ADX (Average Directional Index)
            if 'adx' in market_data.columns:
                adx = np.reshape(market_data['adx'].values / 100.0, (1, -1))  # Normalize to [0, 1]
                trend_features.append(adx)
        except Exception as e:
            print(f"Error in _get_trend_indicators: {e}")
        
        # Ensure at least one feature is returned
        if not trend_features:
            trend_features.append(np.zeros((1, min(len(market_data), self.window_size))))
            
        return trend_features
    
    def _get_momentum_indicators(self, market_data: pd.DataFrame) -> List[np.ndarray]:
        """
        Extract momentum indicators from market data.
        
        Args:
            market_data: Market data for observation window
            
        Returns:
            momentum_features: List of momentum indicator arrays
        """
        momentum_features = []
        
        try:
            # RSI
            if 'rsi' in market_data.columns:
                rsi = np.reshape(market_data['rsi'].values / 100.0, (1, -1))  # Normalize to [0, 1]
                momentum_features.append(rsi)
                
            # Stochastic Oscillator
            if 'slowk' in market_data.columns and 'slowd' in market_data.columns:
                slowk = np.reshape(market_data['slowk'].values / 100.0, (1, -1))  # Normalize to [0, 1]
                slowd = np.reshape(market_data['slowd'].values / 100.0, (1, -1))  # Normalize to [0, 1]
                momentum_features.append(slowk)
                momentum_features.append(slowd)
                
            # CCI (Commodity Channel Index)
            if 'cci' in market_data.columns:
                # Normalize CCI to [-1, 1]
                cci = np.reshape(np.clip(market_data['cci'].values / 200.0, -1, 1), (1, -1))
                momentum_features.append(cci)
        except Exception as e:
            print(f"Error in _get_momentum_indicators: {e}")
        
        # Ensure at least one feature is returned
        if not momentum_features:
            momentum_features.append(np.zeros((1, min(len(market_data), self.window_size))))
            
        return momentum_features
    
    def _get_volatility_indicators(self, market_data: pd.DataFrame) -> List[np.ndarray]:
        """
        Extract volatility indicators from market data.
        
        Args:
            market_data: Market data for observation window
            
        Returns:
            volatility_features: List of volatility indicator arrays
        """
        volatility_features = []
        
        try:
            # Bollinger Bands
            if all(col in market_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                # Calculate Bollinger band width
                bb_width = (market_data['bb_upper'] - market_data['bb_lower']) / market_data['bb_middle']
                volatility_features.append(np.reshape(bb_width.values, (1, -1)))
                
                # Calculate price position within bands (0 = at lower band, 1 = at upper band)
                upper_diff = market_data['bb_upper'] - market_data['close']
                lower_diff = market_data['close'] - market_data['bb_lower']
                band_range = market_data['bb_upper'] - market_data['bb_lower']
                
                # Avoid division by zero
                band_range = np.maximum(band_range, 0.001)
                
                # Calculate normalized position (0 to 1)
                band_position = lower_diff / band_range
                volatility_features.append(np.reshape(band_position.values, (1, -1)))
                
            # ATR (Average True Range)
            if 'atr' in market_data.columns:
                # Normalize ATR by price
                normalized_atr = market_data['atr'] / market_data['close']
                volatility_features.append(np.reshape(normalized_atr.values, (1, -1)))
                
            # ATR percentage
            if 'atr_pct' in market_data.columns:
                volatility_features.append(np.reshape(market_data['atr_pct'].values, (1, -1)))
                
            # General volatility (standard deviation of returns)
            if 'volatility' in market_data.columns:
                volatility_features.append(np.reshape(market_data['volatility'].values, (1, -1)))
        except Exception as e:
            print(f"Error in _get_volatility_indicators: {e}")
        
        # Ensure at least one feature is returned
        if not volatility_features:
            volatility_features.append(np.zeros((1, min(len(market_data), self.window_size))))
            
        return volatility_features
    
    def _get_volume_indicators(self, market_data: pd.DataFrame) -> List[np.ndarray]:
        """
        Extract volume indicators from market data.
        
        Args:
            market_data: Market data for observation window
            
        Returns:
            volume_features: List of volume indicator arrays
        """
        volume_features = []
        
        try:
            # OBV (On-Balance Volume)
            if 'obv' in market_data.columns:
                # Normalize OBV by its standard deviation
                obv_std = np.std(market_data['obv'].values)
                if obv_std > 0:
                    normalized_obv = market_data['obv'].values / obv_std
                else:
                    normalized_obv = market_data['obv'].values
                    
                volume_features.append(np.reshape(normalized_obv, (1, -1)))
                
            # CMF (Chaikin Money Flow)
            if 'cmf' in market_data.columns:
                volume_features.append(np.reshape(market_data['cmf'].values, (1, -1)))
                
            # MFI (Money Flow Index)
            if 'mfi' in market_data.columns:
                normalized_mfi = market_data['mfi'].values / 100.0  # Normalize to [0, 1]
                volume_features.append(np.reshape(normalized_mfi, (1, -1)))
                
            # Volume ratio
            if 'volume_ratio' in market_data.columns:
                volume_features.append(np.reshape(market_data['volume_ratio'].values, (1, -1)))
                
            # Volume SMA ratio
            if 'volume' in market_data.columns and 'volume_sma_20' in market_data.columns:
                volume_sma_ratio = market_data['volume'].values / np.maximum(market_data['volume_sma_20'].values, 1)
                volume_features.append(np.reshape(volume_sma_ratio, (1, -1)))
        except Exception as e:
            print(f"Error in _get_volume_indicators: {e}")
        
        # Ensure at least one feature is returned
        if not volume_features:
            volume_features.append(np.zeros((1, min(len(market_data), self.window_size))))
            
        return volume_features
    
    def _get_market_context(
        self, 
        market_data: pd.DataFrame, 
        market_index_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Extract market context information from index data.
        
        Args:
            market_data: Stock market data for observation window
            market_index_data: Market index data for the same window
            
        Returns:
            market_context: Array of market context features
        """
        try:
            # Extract relative performance of stock vs index
            if len(market_index_data) > 0 and 'close' in market_index_data.columns:
                stock_returns = market_data['close'].pct_change().fillna(0).values
                index_returns = market_index_data['close'].pct_change().fillna(0).values
                
                # For vectors of different lengths, take the last N elements where N is the minimum length
                min_length = min(len(stock_returns), len(index_returns))
                stock_returns = stock_returns[-min_length:]
                index_returns = index_returns[-min_length:]
                
                # Calculate correlation
                correlation = np.corrcoef(stock_returns, index_returns)[0, 1] if min_length > 1 else 0
                if np.isnan(correlation):
                    correlation = 0
                    
                # Calculate relative strength (stock return - index return)
                relative_strength = np.mean(stock_returns - index_returns)
                if np.isnan(relative_strength):
                    relative_strength = 0
                    
                # Calculate beta (using covariance and variance)
                index_var = np.var(index_returns)
                if index_var > 0:
                    beta = np.cov(stock_returns, index_returns)[0, 1] / index_var
                else:
                    beta = 1.0
                    
                if np.isnan(beta):
                    beta = 1.0
                    
                # Return market context features
                return np.array([[correlation, relative_strength, beta]]).flatten()
            else:
                return np.zeros(3)
        except Exception as e:
            print(f"Error in _get_market_context: {e}")
            return np.zeros(3)
    
    def _get_position_features(
        self,
        current_position: float,
        position_entry_price: float,
        position_entry_step: int,
        current_step: int,
        stop_loss_pct: float,
        take_profit_pct: float,
        latest_data: pd.Series
    ) -> np.ndarray:
        """
        Generate position-related features.
        
        Args:
            current_position: Current position size
            position_entry_price: Entry price of current position
            position_entry_step: Step at which position was entered
            current_step: Current time step
            stop_loss_pct: Current stop loss percentage
            take_profit_pct: Current take profit percentage
            latest_data: Latest market data point
            
        Returns:
            position_features: Array of position features
        """
        features = np.zeros(7)
        
        try:
            # 1. Normalized position size (0 to 1)
            features[0] = min(current_position / 100, 1.0) if current_position > 0 else 0
            
            # 2. Position indicator (1 if has position, 0 otherwise)
            features[1] = 1.0 if current_position > 0 else 0.0
            
            if current_position > 0 and position_entry_price > 0:
                # Get current close price
                current_price = latest_data['close']
                
                # 3. Price change since entry (as %)
                price_change = (current_price - position_entry_price) / position_entry_price
                features[2] = price_change
                
                # 4. Days held (normalized)
                days_held = (current_step - position_entry_step) if current_step >= position_entry_step else 0
                features[3] = min(days_held / 20, 1.0)  # Normalize to [0, 1] with 20 days as max
                
                # 5. Stop loss information
                if stop_loss_pct > 0:
                    stop_price = position_entry_price * (1 - stop_loss_pct)
                    # Distance to stop loss as % of current price
                    stop_distance = (current_price - stop_price) / current_price
                    features[4] = max(min(stop_distance, 1.0), -1.0)  # Clip to [-1, 1]
                
                # 6. Take profit information
                if take_profit_pct > 0:
                    take_profit_price = position_entry_price * (1 + take_profit_pct)
                    # Distance to take profit as % of current price
                    tp_distance = (take_profit_price - current_price) / current_price
                    features[5] = max(min(tp_distance, 1.0), -1.0)  # Clip to [-1, 1]
                    
                # 7. Risk-reward ratio
                if stop_loss_pct > 0 and take_profit_pct > 0:
                    risk_reward_ratio = take_profit_pct / stop_loss_pct
                    features[6] = min(risk_reward_ratio / 3, 1.0)  # Normalize with 3 as "good" ratio
        except Exception as e:
            print(f"Error in _get_position_features: {e}")
            
        return features
    
    def _get_account_features(
        self,
        portfolio_value: float,
        initial_capital: float,
        cash_balance: float,
        max_portfolio_value: float,
        drawdown: float
    ) -> np.ndarray:
        """
        Generate account status features.
        
        Args:
            portfolio_value: Current portfolio value
            initial_capital: Initial capital
            cash_balance: Current cash balance
            max_portfolio_value: Maximum portfolio value
            drawdown: Current drawdown
            
        Returns:
            account_features: Array of account features
        """
        features = np.zeros(5)
        
        try:
            # 1. Total return vs initial capital
            if initial_capital > 0:
                total_return = (portfolio_value / initial_capital) - 1
                features[0] = max(min(total_return, 1.0), -1.0)  # Clip to [-1, 1]
                
            # 2. Cash ratio (proportion of portfolio in cash)
            if portfolio_value > 0:
                cash_ratio = cash_balance / portfolio_value
                features[1] = min(cash_ratio, 1.0)  # Clip to [0, 1]
                
            # 3. Drawdown
            features[2] = min(drawdown, 1.0)  # Clip to [0, 1]
            
            # 4. Proximity to peak (1 - drawdown)
            features[3] = 1.0 - features[2]
            
            # 5. Relative portfolio size compared to initial capital
            if initial_capital > 0:
                relative_size = portfolio_value / initial_capital
                features[4] = min(relative_size, 5.0) / 5.0  # Normalize to [0, 1] with 5x as max
        except Exception as e:
            print(f"Error in _get_account_features: {e}")
            
        return features
    
    def _get_time_features(self, timestamp: pd.Timestamp) -> np.ndarray:
        """
        Generate time-related features.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            time_features: Array of time features
        """
        features = np.zeros(7)
        
        try:
            # 1. Day of week (normalized, 0=Monday, 1=Sunday)
            features[0] = timestamp.dayofweek / 6.0
            
            # 2-6. One-hot encoding of day of week
            features[1 + timestamp.dayofweek] = 1.0
        except Exception as e:
            print(f"Error in _get_time_features: {e}")
            
        return features