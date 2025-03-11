"""
observation/generator.py

Generates observations for the trading environment.

This module processes market data and current state information
to create observation vectors for the agent, including:
- Price data features
- Volume information
- Technical indicators
- Volatility metrics
- Market context
- Position information
- Account status
- Time features

Author: [Your Name]
Date: March 10, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import talib
from sklearn.decomposition import PCA


class ObservationGenerator:
    """
    Generates comprehensive observation vectors for the RL agent.
    
    This class handles:
    - Feature engineering from raw market data
    - Technical indicator calculation
    - Position and account information integration
    - Data normalization and preprocessing
    - Dimensionality reduction with PCA
    """
    
    def __init__(
        self,
        window_size: int = 20,
        include_sentiment: bool = False,
        use_pca: bool = True,
        pca_variance_threshold: float = 0.95,
        include_market_context: bool = True
    ):
        """
        Initialize the observation generator.
        
        Args:
            window_size: Number of days in the lookback window
            include_sentiment: Whether to include sentiment features
            use_pca: Whether to apply PCA for dimensionality reduction
            pca_variance_threshold: Target explained variance for PCA
            include_market_context: Whether to include market index data
        """
        self.window_size = window_size
        self.include_sentiment = include_sentiment
        self.use_pca = use_pca
        self.pca_variance_threshold = pca_variance_threshold
        self.include_market_context = include_market_context
        
        # Initialize PCA model
        self.pca_model = None
        self.feature_names = []
        
        # Feature flags to control what gets included
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
        Generate the complete observation vector.
        
        Args:
            market_data: Market data for the observation window
            current_step: Current time step in the environment
            current_position: Current position size in shares
            position_entry_price: Entry price of current position
            position_entry_step: Step at which current position was entered
            stop_loss_pct: Current stop loss percentage
            take_profit_pct: Current take profit percentage
            portfolio_value: Current total portfolio value
            initial_capital: Initial capital amount
            cash_balance: Current cash balance
            max_portfolio_value: Maximum portfolio value achieved
            drawdown: Current drawdown from peak
            market_index_data: Market index data for market context
            
        Returns:
            observation: Complete observation vector
        """
        # List to hold all observation components
        observation_components = []
        
        # Get latest data point
        latest_data = market_data.iloc[-1]
        
        # 1. Price Data Features
        if self.feature_flags['price_data']:
            price_features = self._get_price_features(market_data)
            observation_components.append(price_features)
            
        # 2. Volume Information
        if self.feature_flags['volume_data']:
            volume_features = self._get_volume_features(market_data)
            observation_components.append(volume_features)
            
        # 3. Technical Indicators
        tech_indicator_features = []
        
        # 3.1 Trend Indicators
        if self.feature_flags['trend_indicators']:
            trend_features = self._get_trend_indicators(market_data)
            tech_indicator_features.extend(trend_features)
            
        # 3.2 Momentum Indicators
        if self.feature_flags['momentum_indicators']:
            momentum_features = self._get_momentum_indicators(market_data)
            tech_indicator_features.extend(momentum_features)
            
        # 3.3 Volatility Indicators
        if self.feature_flags['volatility_indicators']:
            volatility_features = self._get_volatility_indicators(market_data)
            tech_indicator_features.extend(volatility_features)
            
        # 3.4 Volume Indicators
        if self.feature_flags['volume_indicators']:
            volume_ind_features = self._get_volume_indicators(market_data)
            tech_indicator_features.extend(volume_ind_features)
            
        # Apply PCA to technical indicators if enabled
        if self.use_pca and len(tech_indicator_features) > 0:
            tech_indicators_array = np.array(tech_indicator_features)
            if self.pca_model is None:
                # First time, fit the PCA model
                self.pca_model = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
                self.pca_model.fit(tech_indicators_array.T)  # Transpose to get features as columns
            
            # Transform the data
            tech_indicators_pca = self.pca_model.transform(tech_indicators_array.T).T
            observation_components.append(tech_indicators_pca)
        else:
            # Add the raw technical indicators
            observation_components.append(np.array(tech_indicator_features))
        
        # 4. Market Context
        if self.feature_flags['market_context'] and self.include_market_context and market_index_data is not None:
            market_context_features = self._get_market_context(market_data, market_index_data)
            observation_components.append(market_context_features)
            
        # 5. Position Information
        if self.feature_flags['position_info']:
            position_features = self._get_position_features(
                current_position,
                position_entry_price,
                position_entry_step,
                current_step,
                stop_loss_pct,
                take_profit_pct,
                latest_data
            )
            observation_components.append(position_features)
            
        # 6. Account Status
        if self.feature_flags['account_status']:
            account_features = self._get_account_features(
                portfolio_value,
                initial_capital,
                cash_balance,
                max_portfolio_value,
                drawdown
            )
            observation_components.append(account_features)
            
        # 7. Time Features
        if self.feature_flags['time_features'] and isinstance(market_data.index[0], pd.Timestamp):
            time_features = self._get_time_features(market_data.index[-1])
            observation_components.append(time_features)
            
        # Flatten all components into a single array
        flattened_components = []
        for component in observation_components:
            if component.ndim == 1:
                flattened_components.append(component)
            else:
                # Flatten if it's 2D+
                flattened_components.append(component.flatten())
                
        observation = np.concatenate(flattened_components)
        return observation.astype(np.float32)
    
    def _get_price_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract price-related features.
        
        Args:
            market_data: Market data for the observation window
            
        Returns:
            price_features: Array of price-related features
        """
        # Extract price data
        close_prices = market_data['close'].values
        open_prices = market_data['open'].values if 'open' in market_data.columns else close_prices
        high_prices = market_data['high'].values if 'high' in market_data.columns else close_prices
        low_prices = market_data['low'].values if 'low' in market_data.columns else close_prices
        
        # Calculate returns
        returns = np.diff(close_prices) / close_prices[:-1]
        returns = np.append(0, returns)  # Add 0 for the first day
        
        # Calculate Z-score normalized OHLC
        mean_price = np.mean(close_prices)
        std_price = np.std(close_prices)
        z_open = (open_prices - mean_price) / std_price if std_price > 0 else open_prices
        z_high = (high_prices - mean_price) / std_price if std_price > 0 else high_prices
        z_low = (low_prices - mean_price) / std_price if std_price > 0 else low_prices
        z_close = (close_prices - mean_price) / std_price if std_price > 0 else close_prices
        
        # Calculate price relative to moving average
        sma_20 = talib.SMA(close_prices, timeperiod=min(20, len(close_prices)))
        price_to_sma = close_prices / sma_20 if np.any(sma_20 > 0) else np.ones_like(close_prices)
        
        # Replace NaN values
        price_to_sma = np.nan_to_num(price_to_sma, nan=1.0)
        
        # Create normalized price data using percentage changes
        # This helps with generalization across different price scales
        normalized_open = open_prices / close_prices[0] - 1 if close_prices[0] > 0 else open_prices
        normalized_high = high_prices / close_prices[0] - 1 if close_prices[0] > 0 else high_prices
        normalized_low = low_prices / close_prices[0] - 1 if close_prices[0] > 0 else low_prices
        normalized_close = close_prices / close_prices[0] - 1 if close_prices[0] > 0 else close_prices
        
        # Combine all price features
        price_features = np.column_stack([
            returns,
            z_open, z_high, z_low, z_close,
            price_to_sma,
            normalized_open, normalized_high, normalized_low, normalized_close
        ])
        
        return price_features
    
    def _get_volume_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Extract volume-related features.
        
        Args:
            market_data: Market data for the observation window
            
        Returns:
            volume_features: Array of volume-related features
        """
        # Check if volume data is available
        if 'volume' not in market_data.columns:
            return np.zeros((len(market_data), 3))  # Return zeros if no volume
            
        # Extract volume data
        volume = market_data['volume'].values
        
        # Normalize volume to 20-day average
        volume_sma_20 = talib.SMA(volume, timeperiod=min(20, len(volume)))
        normalized_volume = volume / volume_sma_20 if np.any(volume_sma_20 > 0) else np.ones_like(volume)
        
        # Calculate On-Balance Volume (OBV)
        close_prices = market_data['close'].values
        obv = talib.OBV(close_prices, volume)
        
        # Normalize OBV
        obv_norm = (obv - np.mean(obv)) / np.std(obv) if np.std(obv) > 0 else obv
        
        # Calculate Volume Rate of Change (5-day)
        volume_roc = talib.ROC(volume, timeperiod=min(5, len(volume)-1))
        
        # Replace NaN values
        normalized_volume = np.nan_to_num(normalized_volume, nan=1.0)
        obv_norm = np.nan_to_num(obv_norm, nan=0.0)
        volume_roc = np.nan_to_num(volume_roc, nan=0.0)
        
        # Combine all volume features
        volume_features = np.column_stack([
            normalized_volume,
            obv_norm,
            volume_roc
        ])
        
        return volume_features
    
    def _get_trend_indicators(self, market_data: pd.DataFrame) -> List[np.ndarray]:
        """
        Calculate trend indicators.
        
        Args:
            market_data: Market data for the observation window
            
        Returns:
            trend_features: List of trend indicator arrays
        """
        close_prices = market_data['close'].values
        high_prices = market_data['high'].values if 'high' in market_data.columns else close_prices
        low_prices = market_data['low'].values if 'low' in market_data.columns else close_prices
        
        # ADX (14) - Average Directional Index
        try:
            adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=min(14, len(close_prices)-1))
            adx = np.nan_to_num(adx, nan=0.0) / 100.0  # Normalize to [0,1]
        except Exception:
            adx = np.zeros_like(close_prices)
        
        # Moving Average Convergence (20/50)
        try:
            sma_20 = talib.SMA(close_prices, timeperiod=min(20, len(close_prices)))
            sma_50 = talib.SMA(close_prices, timeperiod=min(50, len(close_prices)))
            # Calculate the difference between the two
            ma_convergence = (sma_20 - sma_50) / close_prices if len(close_prices) > 0 else np.zeros_like(sma_20)
            ma_convergence = np.nan_to_num(ma_convergence, nan=0.0)
        except Exception:
            ma_convergence = np.zeros_like(close_prices)
        
        return [adx, ma_convergence]
    
    def _get_momentum_indicators(self, market_data: pd.DataFrame) -> List[np.ndarray]:
        """
        Calculate momentum indicators.
        
        Args:
            market_data: Market data for the observation window
            
        Returns:
            momentum_features: List of momentum indicator arrays
        """
        close_prices = market_data['close'].values
        
        # RSI (14)
        try:
            rsi = talib.RSI(close_prices, timeperiod=min(14, len(close_prices)-1))
            rsi = np.nan_to_num(rsi, nan=50.0) / 100.0  # Normalize to [0,1]
        except Exception:
            rsi = np.ones_like(close_prices) * 0.5  # Default to 0.5 (50%)
        
        # MACD (12,26,9)
        try:
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices, 
                fastperiod=min(12, len(close_prices)-1), 
                slowperiod=min(26, len(close_prices)-1), 
                signalperiod=min(9, len(close_prices)-1)
            )
            # Normalize MACD by price to make it scale-invariant
            macd = macd / close_prices if len(close_prices) > 0 else np.zeros_like(macd)
            macd_signal = macd_signal / close_prices if len(close_prices) > 0 else np.zeros_like(macd_signal)
            macd_hist = macd_hist / close_prices if len(close_prices) > 0 else np.zeros_like(macd_hist)
            
            # Replace NaN values
            macd = np.nan_to_num(macd, nan=0.0)
            macd_signal = np.nan_to_num(macd_signal, nan=0.0)
            macd_hist = np.nan_to_num(macd_hist, nan=0.0)
        except Exception:
            macd = np.zeros_like(close_prices)
            macd_signal = np.zeros_like(close_prices)
            macd_hist = np.zeros_like(close_prices)
        
        return [rsi, macd, macd_signal, macd_hist]
    
    def _get_volatility_indicators(self, market_data: pd.DataFrame) -> List[np.ndarray]:
        """
        Calculate volatility indicators.
        
        Args:
            market_data: Market data for the observation window
            
        Returns:
            volatility_features: List of volatility indicator arrays
        """
        close_prices = market_data['close'].values
        high_prices = market_data['high'].values if 'high' in market_data.columns else close_prices
        low_prices = market_data['low'].values if 'low' in market_data.columns else close_prices
        
        # ATR (14) - Average True Range
        try:
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=min(14, len(close_prices)-1))
            # Normalize ATR by price to get percentage volatility
            atr_pct = atr / close_prices if len(close_prices) > 0 else np.zeros_like(atr)
        except Exception:
            atr_pct = np.ones_like(close_prices) * 0.01  # Default to 1% volatility
        
        # Bollinger Band Width (20)
        try:
            upperband, middleband, lowerband = talib.BBANDS(
                close_prices, 
                timeperiod=min(20, len(close_prices)), 
                nbdevup=2, 
                nbdevdn=2, 
                matype=0
            )
            # Calculate Bollinger Band Width normalized by price
            bb_width = (upperband - lowerband) / middleband if np.all(middleband != 0) else np.zeros_like(upperband)
        except Exception:
            bb_width = np.ones_like(close_prices) * 0.1  # Default to 10% width
        
        # 20-day standard deviation of returns
        try:
            returns = np.diff(close_prices) / close_prices[:-1]
            returns = np.append(0, returns)  # Add 0 for the first day
            std_returns = np.array([
                np.std(returns[max(0, i-min(20, len(returns))):i+1]) 
                for i in range(len(returns))
            ])
        except Exception:
            std_returns = np.ones_like(close_prices) * 0.01  # Default to 1% std dev
        
        # Replace NaN values
        atr_pct = np.nan_to_num(atr_pct, nan=0.01)  # Default to 1% volatility
        bb_width = np.nan_to_num(bb_width, nan=0.1)  # Default to 10% width
        std_returns = np.nan_to_num(std_returns, nan=0.01)  # Default to 1% std dev
        
        return [atr_pct, bb_width, std_returns]
    
    def _get_volume_indicators(self, market_data: pd.DataFrame) -> List[np.ndarray]:
        """
        Calculate volume-based indicators.
        
        Args:
            market_data: Market data for the observation window
            
        Returns:
            volume_indicator_features: List of volume indicator arrays
        """
        close_prices = market_data['close'].values
        
        # Check if volume data is available
        if 'volume' not in market_data.columns:
            return [np.ones_like(close_prices) * 0.5]  # Default MFI to 50%
            
        high_prices = market_data['high'].values if 'high' in market_data.columns else close_prices
        low_prices = market_data['low'].values if 'low' in market_data.columns else close_prices
        volume = market_data['volume'].values
        
        # Money Flow Index (14)
        try:
            mfi = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=min(14, len(close_prices)-1))
            mfi = np.nan_to_num(mfi, nan=50.0) / 100.0  # Normalize to [0,1]
        except Exception:
            # In case of errors (e.g., insufficient data)
            mfi = np.full_like(close_prices, 0.5)
        
        return [mfi]
    
    def _get_market_context(
        self, 
        market_data: pd.DataFrame, 
        market_index_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate market context features.
        
        Args:
            market_data: Market data for the observation window
            market_index_data: Market index data (e.g., S&P 500)
            
        Returns:
            market_context_features: Array of market context features
        """
        # Calculate relative performance vs S&P 500
        # This helps the agent distinguish stock-specific movements from broader market trends
        stock_returns = market_data['close'].pct_change().fillna(0).values
        
        try:
            # Try to align the index data with stock data
            aligned_index_data = market_index_data.reindex(market_data.index, method='ffill')
            index_returns = aligned_index_data['close'].pct_change().fillna(0).values
            
            # Calculate relative performance (excess return)
            relative_performance = stock_returns - index_returns
        except (KeyError, ValueError, AttributeError):
            # If alignment fails, use zeros
            relative_performance = np.zeros_like(stock_returns)
        
        # Combine into a single array
        market_context_features = np.column_stack([relative_performance])
        
        return market_context_features
    
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
        Calculate position information features.
        
        Args:
            current_position: Current position size in shares
            position_entry_price: Entry price of current position
            position_entry_step: Step at which current position was entered
            current_step: Current time step
            stop_loss_pct: Current stop loss percentage
            take_profit_pct: Current take profit percentage
            latest_data: Latest market data
            
        Returns:
            position_features: Array of position information features
        """
        # Current position size (normalized)
        position_size_norm = 1.0 if current_position > 0 else 0.0
        
        # Current P&L of open position (%)
        current_price = latest_data['close']
        position_pnl_pct = ((current_price / position_entry_price) - 1.0) * 100 if position_entry_price > 0 else 0
        position_pnl_norm = position_pnl_pct / 100  # Normalize to [0,1] range approximately
        
        # Days position has been held
        days_held = current_step - position_entry_step if current_position > 0 else 0
        days_held_norm = min(days_held / 20, 1.0)  # Normalize by max holding period
        
        # Distance to stop loss (%)
        distance_to_sl = ((current_price - (position_entry_price * (1 - stop_loss_pct))) / current_price) * 100 \
            if position_entry_price > 0 and stop_loss_pct > 0 else 0
        distance_to_sl_norm = min(distance_to_sl / 10, 1.0)  # Normalize assuming 10% is a large distance
        
        # Distance to take profit (%)
        distance_to_tp = (((position_entry_price * (1 + take_profit_pct)) - current_price) / current_price) * 100 \
            if position_entry_price > 0 and take_profit_pct > 0 else 0
        distance_to_tp_norm = min(distance_to_tp / 20, 1.0)  # Normalize assuming 20% is a large distance
        
        # Combine features
        position_features = np.array([
            position_size_norm,
            position_pnl_norm,
            days_held_norm,
            distance_to_sl_norm,
            distance_to_tp_norm
        ])
        
        return position_features
    
    def _get_account_features(
        self,
        portfolio_value: float,
        initial_capital: float,
        cash_balance: float,
        max_portfolio_value: float,
        drawdown: float
    ) -> np.ndarray:
        """
        Calculate account status features.
        
        Args:
            portfolio_value: Current total portfolio value
            initial_capital: Initial capital amount
            cash_balance: Current cash balance
            max_portfolio_value: Maximum portfolio value achieved
            drawdown: Current drawdown from peak
            
        Returns:
            account_features: Array of account status features
        """
        # Available capital (% of starting capital)
        available_capital_pct = min(cash_balance / initial_capital, 1.0) if initial_capital > 0 else 1.0
        
        # Current portfolio value (normalized)
        portfolio_value_norm = portfolio_value / initial_capital if initial_capital > 0 else 1.0
        
        # Maximum drawdown experienced
        max_drawdown = min(drawdown, 1.0)
        
        # Combine features
        account_features = np.array([
            available_capital_pct,
            portfolio_value_norm,
            max_drawdown
        ])
        
        return account_features
    
    def _get_time_features(self, date: Union[pd.Timestamp, datetime]) -> np.ndarray:
        """
        Calculate time-based features.
        
        Args:
            date: Current date
            
        Returns:
            time_features: Array of time features
        """
        # Day of week (one-hot encoded)
        try:
            day_of_week = date.dayofweek
            day_of_week_onehot = np.zeros(5)  # Monday-Friday (0-4)
            if day_of_week < 5:  # Ensure it's a weekday
                day_of_week_onehot[day_of_week] = 1
            
            # Month (one-hot encoded)
            month = date.month - 1  # Convert to 0-11
            month_onehot = np.zeros(12)
            month_onehot[month] = 1
            
            # Combine features
            time_features = np.concatenate([day_of_week_onehot, month_onehot])
        except (AttributeError, TypeError):
            # If date doesn't have necessary attributes, return empty features
            time_features = np.zeros(17)  # 5 for days + 12 for months
        
        return time_features