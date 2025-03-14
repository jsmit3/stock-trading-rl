"""
data/processor.py

Processes and prepares market data for the trading environment.

This module:
- Cleans and validates raw price data
- Calculates technical indicators
- Handles missing values and outliers
- Prepares data for the observation space
- Provides normalization methods

Author: [Your Name]
Date: March 10, 2025
"""

import numpy as np
import pandas as pd
import talib
from typing import Optional, List, Dict, Union, Tuple
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """
    Processes and prepares market data for the trading environment.
    
    This class:
    - Cleans and validates raw price data
    - Calculates technical indicators
    - Handles missing values and outliers
    - Prepares data for the observation space
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        window_size: int = 20,
        handle_missing: bool = True,
        detect_anomalies: bool = True,
        calculate_indicators: bool = True,
        standard_indicators_only: bool = False
    ):
        """
        Initialize the data processor.
        
        Args:
            price_data: Raw price data (DataFrame with OHLCV columns)
            window_size: Lookback window size for indicators
            handle_missing: Whether to handle missing values
            detect_anomalies: Whether to detect and remove anomalies
            calculate_indicators: Whether to calculate technical indicators
            standard_indicators_only: Whether to use only standard indicators
        """
        self.raw_data = price_data.copy()
        self.window_size = window_size
        self.handle_missing = handle_missing
        self.detect_anomalies = detect_anomalies
        self.calculate_indicators = calculate_indicators
        self.standard_indicators_only = standard_indicators_only
        
        # Validate input data
        self._validate_input_data()
        
        # Scalers for normalization
        self.price_scaler = StandardScaler()
        self.volume_scaler = StandardScaler()
    
    def process_data(self) -> pd.DataFrame:
        """
        Process the raw price data for use in the trading environment.
        
        Returns:
            processed_data: Processed and enhanced price data
        """
        # Create a copy of the raw data
        processed_data = self.raw_data.copy()
        
        # Ensure the data has the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in processed_data.columns]
        
        # If missing columns, try to add them with reasonable approximations
        if missing_columns:
            processed_data = self._add_missing_columns(processed_data, missing_columns)
        
        # Handle missing values if enabled
        if self.handle_missing:
            processed_data = self._handle_missing_values(processed_data)
            
        # Detect and handle anomalies if enabled
        if self.detect_anomalies:
            processed_data = self._detect_and_handle_anomalies(processed_data)
            
        # Calculate technical indicators if enabled
        if self.calculate_indicators:
            processed_data = self._calculate_technical_indicators(processed_data)
            
        # Calculate price differences and returns
        processed_data['return'] = processed_data['close'].pct_change()
        processed_data['return_1d'] = processed_data['close'].pct_change(1)
        processed_data['return_5d'] = processed_data['close'].pct_change(5)
        processed_data['return_20d'] = processed_data['close'].pct_change(20)
        
        # Calculate price statistics
        processed_data['price_mean_20d'] = processed_data['close'].rolling(window=20).mean()
        processed_data['price_std_20d'] = processed_data['close'].rolling(window=20).std()
        processed_data['price_z_score'] = (
            (processed_data['close'] - processed_data['price_mean_20d']) / 
            processed_data['price_std_20d']
        ).replace([np.inf, -np.inf], 0)
        
        # Fill NaN values with appropriate defaults
        processed_data = processed_data.ffill().bfill().fillna(0)
        
        # Add day of week feature
        if isinstance(processed_data.index, pd.DatetimeIndex):
            processed_data['day_of_week'] = processed_data.index.dayofweek
            processed_data['month'] = processed_data.index.month
            
        return processed_data
    
    def _add_missing_columns(
        self, 
        data: pd.DataFrame, 
        missing_columns: List[str]
    ) -> pd.DataFrame:
        """
        Add missing columns with approximated values.
        
        Args:
            data: Input DataFrame
            missing_columns: List of missing columns to add
            
        Returns:
            data: DataFrame with added columns
        """
        if 'close' not in data.columns:
            raise ValueError("'close' column is required and cannot be approximated")
        
        # Create a copy of the input data
        enhanced_data = data.copy()
        
        # Add approximations for missing columns
        for col in missing_columns:
            if col == 'open':
                # Use previous day's close as an approximation for open
                enhanced_data['open'] = enhanced_data['close'].shift(1)
                # Use current close for the first day
                enhanced_data['open'].iloc[0] = enhanced_data['close'].iloc[0]
            
            elif col == 'high':
                # Approximate high as 0.5% above close if no open, or max of open and close plus 0.25%
                if 'open' in enhanced_data.columns:
                    enhanced_data['high'] = enhanced_data[['open', 'close']].max(axis=1) * 1.0025
                else:
                    enhanced_data['high'] = enhanced_data['close'] * 1.005
            
            elif col == 'low':
                # Approximate low as 0.5% below close if no open, or min of open and close minus 0.25%
                if 'open' in enhanced_data.columns:
                    enhanced_data['low'] = enhanced_data[['open', 'close']].min(axis=1) * 0.9975
                else:
                    enhanced_data['low'] = enhanced_data['close'] * 0.995
            
            elif col == 'volume':
                # Create synthetic volume based on daily return
                returns = enhanced_data['close'].pct_change().abs()
                mean_return = returns.mean()
                enhanced_data['volume'] = np.where(
                    returns > mean_return,
                    np.random.lognormal(15, 1, len(enhanced_data)),  # Higher volume on volatile days
                    np.random.lognormal(14, 0.5, len(enhanced_data))  # Lower volume on calm days
                )
        
        return enhanced_data
    
    def _validate_input_data(self) -> None:
        """
        Validate the input data structure and content.
        
        Raises:
            ValueError: If the data is invalid
        """
        # Check if the data is a DataFrame
        if not isinstance(self.raw_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
            
        # Check if the data has the minimum required columns
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check if the data has at least some rows
        if len(self.raw_data) < 2:
            raise ValueError("Input data must have at least 2 rows")
            
        # If index is datetime, check if it's sorted
        if isinstance(self.raw_data.index, pd.DatetimeIndex):
            if not self.raw_data.index.is_monotonic_increasing:
                raise ValueError("DatetimeIndex must be sorted in ascending order")
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: Input data with potential missing values
            
        Returns:
            cleaned_data: Data with missing values handled
        """
        # Make a copy of the input data
        cleaned_data = data.copy()
        
        # Check for missing values
        missing_values = cleaned_data.isnull().sum()
        
        # If there are missing values
        if missing_values.sum() > 0:
            # For price columns, use forward fill then backward fill
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in cleaned_data.columns:
                    # Count consecutive NaNs
                    na_groups = cleaned_data[col].isnull().astype(int).groupby(
                        cleaned_data[col].notnull().astype(int).cumsum()
                    ).sum()
                    
                    # If there are more than 3 consecutive NaNs, warn but still attempt to fill
                    if (na_groups > 3).any():
                        print(f"Warning: More than 3 consecutive missing values in column '{col}'")
                    
                    # Fill missing values
                    cleaned_data[col] = cleaned_data[col].fillna(method='ffill', limit=3).fillna(method='bfill')
                    
            # For volume, use the median of nearby values
            if 'volume' in cleaned_data.columns:
                cleaned_data['volume'] = cleaned_data['volume'].fillna(
                    cleaned_data['volume'].rolling(window=5, min_periods=1, center=True).median()
                )
                
        return cleaned_data
    
    def _detect_and_handle_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle anomalies in the data.
        
        Args:
            data: Input data with potential anomalies
            
        Returns:
            cleaned_data: Data with anomalies handled
        """
        # Make a copy of the input data
        cleaned_data = data.copy()
        
        # Check for price anomalies (extreme returns)
        if 'close' in cleaned_data.columns:
            # Calculate daily returns
            daily_returns = cleaned_data['close'].pct_change()
            
            # Define anomaly threshold (e.g., +/- 50% in a single day)
            anomaly_threshold = 0.5
            
            # Identify anomalies
            anomalies = (daily_returns.abs() > anomaly_threshold) & (daily_returns.abs() != np.inf)
            
            # Print information about detected anomalies
            if anomalies.sum() > 0:
                print(f"Detected {anomalies.sum()} price anomalies")
                
                # Handle anomalies (replace with previous valid value)
                for idx in cleaned_data.index[anomalies]:
                    prev_idx = cleaned_data.index.get_loc(idx) - 1
                    if prev_idx >= 0:
                        cleaned_data.loc[idx, 'close'] = cleaned_data.iloc[prev_idx]['close']
                        print(f"Corrected anomaly at {idx}")
        
        # Check for volume anomalies
        if 'volume' in cleaned_data.columns:
            # Calculate rolling statistics
            rolling_vol_mean = cleaned_data['volume'].rolling(window=20, min_periods=1).mean()
            rolling_vol_std = cleaned_data['volume'].rolling(window=20, min_periods=1).std()
            
            # Define anomaly threshold (e.g., 3 standard deviations)
            vol_anomalies = (
                (cleaned_data['volume'] > rolling_vol_mean + 3 * rolling_vol_std) | 
                (cleaned_data['volume'] < rolling_vol_mean - 3 * rolling_vol_std)
            )
            
            # Handle volume anomalies
            if vol_anomalies.sum() > 0:
                print(f"Detected {vol_anomalies.sum()} volume anomalies")
                
                # Replace anomalous volumes with the rolling median
                if vol_anomalies.sum() > 0:
                    valid_indices = ~vol_anomalies
                    x_valid = np.where(valid_indices)[0]
                    y_valid = cleaned_data.loc[valid_indices, 'volume'].values
                    # Use interpolation for a smoother result
                    anomaly_indices = np.where(vol_anomalies)[0]
                    if len(x_valid) > 0 and len(anomaly_indices) > 0:
                        interpolated = np.interp(anomaly_indices, x_valid, y_valid)
                        # Convert to integer to avoid dtype warnings
                        cleaned_data.loc[vol_anomalies, 'volume'] = interpolated.astype(np.int64)
                
        return cleaned_data
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the data.
        
        Args:
            data: Input price data
            
        Returns:
            enhanced_data: Data with technical indicators added
        """
        # Make a copy of the input data
        enhanced_data = data.copy()
        
        # Extract price and volume data
        close = enhanced_data['close'].values
        high = enhanced_data['high'].values if 'high' in enhanced_data.columns else close
        low = enhanced_data['low'].values if 'low' in enhanced_data.columns else close
        volume = enhanced_data['volume'].values if 'volume' in enhanced_data.columns else None
        
        try:
            # --- Trend Indicators ---
            
            # SMA - Simple Moving Averages
            enhanced_data['sma_10'] = self._safe_indicator(talib.SMA, close, timeperiod=10)
            enhanced_data['sma_20'] = self._safe_indicator(talib.SMA, close, timeperiod=20)
            enhanced_data['sma_50'] = self._safe_indicator(talib.SMA, close, timeperiod=50)
            
            # EMA - Exponential Moving Averages
            enhanced_data['ema_10'] = self._safe_indicator(talib.EMA, close, timeperiod=10)
            enhanced_data['ema_20'] = self._safe_indicator(talib.EMA, close, timeperiod=20)
            enhanced_data['ema_50'] = self._safe_indicator(talib.EMA, close, timeperiod=50)
            
            # MACD - Moving Average Convergence Divergence
            macd, macd_signal, macd_hist = self._safe_indicator(
                talib.MACD, close, fastperiod=12, slowperiod=26, signalperiod=9, return_tuple=True
            )
            enhanced_data['macd'] = macd
            enhanced_data['macd_signal'] = macd_signal
            enhanced_data['macd_hist'] = macd_hist
                
            # ADX - Average Directional Index
            enhanced_data['adx'] = self._safe_indicator(talib.ADX, high, low, close, timeperiod=14)
            
            # --- Momentum Indicators ---
            
            # RSI - Relative Strength Index
            enhanced_data['rsi'] = self._safe_indicator(talib.RSI, close, timeperiod=14)
            
            # Stochastic
            slowk, slowd = self._safe_indicator(
                talib.STOCH, high, low, close, 
                fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0,
                return_tuple=True
            )
            enhanced_data['slowk'] = slowk
            enhanced_data['slowd'] = slowd
            
            # CCI - Commodity Channel Index
            enhanced_data['cci'] = self._safe_indicator(talib.CCI, high, low, close, timeperiod=14)
            
            # --- Volatility Indicators ---
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._safe_indicator(
                talib.BBANDS, close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0, return_tuple=True
            )
            enhanced_data['bb_upper'] = bb_upper
            enhanced_data['bb_middle'] = bb_middle
            enhanced_data['bb_lower'] = bb_lower
                
            # ATR - Average True Range
            enhanced_data['atr'] = self._safe_indicator(talib.ATR, high, low, close, timeperiod=14)
            
            # Normalize ATR as percentage of price
            with np.errstate(divide='ignore', invalid='ignore'):
                enhanced_data['atr_pct'] = enhanced_data['atr'] / close
                enhanced_data['atr_pct'] = enhanced_data['atr_pct'].replace([np.inf, -np.inf], 0)
            
            # Standard deviation of returns
            enhanced_data['volatility'] = enhanced_data['close'].pct_change().rolling(window=20).std().fillna(0)
            
            # --- Volume Indicators ---
            
            if volume is not None:
                # OBV - On-Balance Volume
                enhanced_data['obv'] = self._safe_indicator(talib.OBV, close, volume)
                
                # CMF - Chaikin Money Flow
                enhanced_data['cmf'] = self._safe_indicator(
                    talib.ADOSC, high, low, close, volume, fastperiod=3, slowperiod=10
                )
                
                # MFI - Money Flow Index
                enhanced_data['mfi'] = self._safe_indicator(talib.MFI, high, low, close, volume, timeperiod=14)
                
                # Volume SMA
                enhanced_data['volume_sma_20'] = self._safe_indicator(talib.SMA, volume, timeperiod=20)
                
                # Volume ratio
                with np.errstate(divide='ignore', invalid='ignore'):
                    enhanced_data['volume_ratio'] = volume / enhanced_data['volume_sma_20'].values
                    enhanced_data['volume_ratio'] = enhanced_data['volume_ratio'].replace([np.inf, -np.inf], 1)
        
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            print("Continuing with basic indicators only")
        
        # Handle NaN values (from the indicator calculations)
        enhanced_data = enhanced_data.bfill().fillna(0)
        
        return enhanced_data
    
    def _safe_indicator(self, indicator_func, *args, return_tuple=False, **kwargs):
        """
        Safely calculate a technical indicator with error handling.
        
        Args:
            indicator_func: TALib indicator function
            *args: Arguments for the indicator function
            return_tuple: Whether the function returns multiple values
            **kwargs: Keyword arguments for the indicator function
            
        Returns:
            Indicator values or tuple of indicator values
        """
        try:
            # Calculate the indicator
            result = indicator_func(*args, **kwargs)
            
            # Return the result
            return result
        except Exception as e:
            # Handle errors
            if return_tuple:
                # Determine number of return values based on the indicator
                if indicator_func.__name__ == 'MACD':
                    return np.zeros_like(args[0]), np.zeros_like(args[0]), np.zeros_like(args[0])
                elif indicator_func.__name__ == 'BBANDS':
                    return np.zeros_like(args[0]), np.zeros_like(args[0]), np.zeros_like(args[0])
                elif indicator_func.__name__ == 'STOCH':
                    return np.zeros_like(args[0]), np.zeros_like(args[0])
                else:
                    return np.zeros_like(args[0]), np.zeros_like(args[0])
            else:
                # Return zeros for single-value indicators
                return np.zeros_like(args[0])
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data for neural network inputs.
        
        Args:
            data: Input data to normalize
            
        Returns:
            normalized_data: Data with normalized values
        """
        # Make a copy of the input data
        normalized_data = data.copy()
        
        # Group columns by type
        price_columns = ['open', 'high', 'low', 'close', 'sma_10', 'sma_20', 'sma_50', 
                        'ema_10', 'ema_20', 'ema_50', 'bb_upper', 'bb_middle', 'bb_lower']
        volume_columns = ['volume', 'obv', 'volume_sma_20']
        
        # Filter to only include columns that exist in the data
        price_columns = [col for col in price_columns if col in normalized_data.columns]
        volume_columns = [col for col in volume_columns if col in normalized_data.columns]
        
        # Normalize price data using z-score
        if price_columns:
            price_data = normalized_data[price_columns].values
            price_data_scaled = self.price_scaler.fit_transform(price_data)
            normalized_data[price_columns] = price_data_scaled
            
        # Normalize volume data using z-score
        if volume_columns:
            volume_data = normalized_data[volume_columns].values
            # Handle zeros and NaNs
            volume_data = np.where(volume_data <= 0, 0.1, volume_data)  # Replace zeros with small value
            volume_data_scaled = self.volume_scaler.fit_transform(volume_data)
            normalized_data[volume_columns] = volume_data_scaled
        
        # Indicators are often already normalized (e.g., RSI is 0-100)
        # but we can still cap extreme values
        indicator_columns = ['rsi', 'slowk', 'slowd', 'adx', 'cci', 'mfi']
        indicator_columns = [col for col in indicator_columns if col in normalized_data.columns]
        for col in indicator_columns:
            if col in ['rsi', 'slowk', 'slowd', 'mfi']:
                # Already 0-100 scale, normalize to 0-1
                normalized_data[col] = normalized_data[col] / 100.0
            elif col == 'cci':
                # CCI can be large, clip to reasonable range and normalize
                normalized_data[col] = np.clip(normalized_data[col], -200, 200) / 200.0
            elif col == 'adx':
                # ADX is 0-100, normalize to 0-1
                normalized_data[col] = normalized_data[col] / 100.0
        
        return normalized_data
    
    def get_price_feature_vector(self, data: pd.DataFrame, window_size: int) -> np.ndarray:
        """
        Extract a price feature vector from the data.
        
        Args:
            data: Price data
            window_size: Window size for the feature vector
            
        Returns:
            feature_vector: Price feature vector
        """
        # Select relevant features
        features = ['close', 'rsi', 'sma_20', 'ema_20', 'volatility', 'macd']
        features = [f for f in features if f in data.columns]
        
        # Get the most recent window of data
        window_data = data[features].iloc[-window_size:].values
        
        # Flatten the data into a feature vector
        feature_vector = window_data.flatten()
        
        return feature_vector
    
    def get_normalized_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get normalized price data with returns rather than absolute prices.
        
        Args:
            data: Price data
            
        Returns:
            normalized_data: Normalized price data
        """
        # Make a copy of the input data
        normalized_data = data.copy()
        
        # Calculate returns for OHLC data
        for col in ['open', 'high', 'low', 'close']:
            if col in normalized_data.columns:
                normalized_data[f'{col}_ret_1d'] = normalized_data[col].pct_change(1)
                normalized_data[f'{col}_ret_5d'] = normalized_data[col].pct_change(5)
                
        # Drop the raw price columns
        price_columns = ['open', 'high', 'low', 'close']
        normalized_data = normalized_data.drop([col for col in price_columns if col in normalized_data.columns], axis=1)
        
        # Fill NaN values
        normalized_data = normalized_data.fillna(0)
        
        return normalized_data