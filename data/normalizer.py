"""
data/normalizer.py

Handles data normalization strategies for the trading environment.

This module provides various normalization approaches to handle the challenge
of making market data suitable for neural network inputs while preserving
important information about price levels.

Author: [Your Name]
Date: March 10, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class DataNormalizer:
    """
    Handles various normalization strategies for market data.
    
    Provides methods for:
    - Z-score normalization
    - Min-max scaling
    - Percentage change normalization
    - Differential normalization
    - Log normalization
    - Hybrid approaches
    """
    
    def __init__(
        self,
        strategy: str = 'z_score',
        feature_groups: Optional[Dict[str, List[str]]] = None,
        preserve_scale: bool = False
    ):
        """
        Initialize the data normalizer.
        
        Args:
            strategy: Normalization strategy ('z_score', 'min_max', 'pct_change', 
                                           'differential', 'log', 'hybrid')
            feature_groups: Dictionary mapping group names to lists of column names
            preserve_scale: Whether to preserve the scale for trading calculations
        """
        self.strategy = strategy
        self.preserve_scale = preserve_scale
        
        # Initialize scalers
        self.scalers = {}
        
        # Set default feature groups if not provided
        if feature_groups is None:
            self.feature_groups = {
                'price': ['open', 'high', 'low', 'close'],
                'volume': ['volume'],
                'indicators': [
                    'rsi', 'macd', 'macd_signal', 'macd_hist', 'slowk', 'slowd',
                    'adx', 'cci', 'mfi', 'cmf'
                ],
                'moving_averages': [
                    'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20', 'ema_50',
                    'bb_upper', 'bb_middle', 'bb_lower'
                ],
                'volatility': ['atr', 'atr_pct', 'volatility']
            }
        else:
            self.feature_groups = feature_groups
            
        # Store original scale information
        self.price_scale_info = {}
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the normalizer to the data.
        
        Args:
            data: Input data to fit the normalizer
        """
        # Store scale information for price columns if preserving scale
        if self.preserve_scale:
            price_cols = self.feature_groups.get('price', [])
            for col in price_cols:
                if col in data.columns:
                    self.price_scale_info[col] = {
                        'mean': data[col].mean(),
                        'std': data[col].std(),
                        'min': data[col].min(),
                        'max': data[col].max(),
                        'last': data[col].iloc[-1]
                    }
        
        # Fit scalers based on strategy
        if self.strategy in ['z_score', 'min_max', 'robust', 'hybrid']:
            for group, columns in self.feature_groups.items():
                # Filter to columns that exist in the data
                group_cols = [col for col in columns if col in data.columns]
                
                if not group_cols:
                    continue
                    
                # Select appropriate scaler based on strategy and group
                if self.strategy == 'z_score' or (self.strategy == 'hybrid' and group != 'price'):
                    self.scalers[group] = StandardScaler()
                elif self.strategy == 'min_max':
                    self.scalers[group] = MinMaxScaler(feature_range=(-1, 1))
                elif self.strategy == 'robust':
                    self.scalers[group] = RobustScaler()
                elif self.strategy == 'hybrid' and group == 'price':
                    # For price in hybrid mode, we'll use percentage change
                    continue
                
                # Fit the scaler
                self.scalers[group].fit(data[group_cols])
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted normalizer.
        
        Args:
            data: Input data to normalize
            
        Returns:
            normalized_data: Normalized data
        """
        # Create a copy of the input data
        normalized_data = data.copy()
        
        # Apply normalization based on strategy
        if self.strategy == 'pct_change':
            normalized_data = self._apply_pct_change(normalized_data)
        elif self.strategy == 'differential':
            normalized_data = self._apply_differential(normalized_data)
        elif self.strategy == 'log':
            normalized_data = self._apply_log_transform(normalized_data)
        elif self.strategy == 'hybrid':
            normalized_data = self._apply_hybrid(normalized_data)
        else:  # 'z_score', 'min_max', 'robust'
            normalized_data = self._apply_scalers(normalized_data)
        
        # Store original price data if preserving scale
        if self.preserve_scale:
            price_cols = self.feature_groups.get('price', [])
            for col in price_cols:
                if col in data.columns:
                    normalized_data[f'original_{col}'] = data[col]
        
        return normalized_data
    
    def inverse_transform(
        self, 
        normalized_data: pd.DataFrame, 
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            normalized_data: Normalized data
            columns: Specific columns to inverse transform (default: all)
            
        Returns:
            original_scale_data: Data transformed back to original scale
        """
        # Create a copy of the input data
        original_scale_data = normalized_data.copy()
        
        # If no columns specified, use all columns that can be inverse transformed
        if columns is None:
            columns = []
            for group, cols in self.feature_groups.items():
                if group in self.scalers:
                    columns.extend([col for col in cols if col in normalized_data.columns])
        
        # Apply inverse transformation based on strategy
        if self.strategy in ['z_score', 'min_max', 'robust']:
            original_scale_data = self._inverse_apply_scalers(original_scale_data, columns)
        elif self.strategy == 'pct_change':
            # Percentage change can't be directly inverted without a reference
            if self.preserve_scale:
                original_scale_data = self._inverse_pct_change(original_scale_data)
        elif self.strategy == 'log':
            original_scale_data = self._inverse_log_transform(original_scale_data, columns)
        elif self.strategy == 'hybrid':
            original_scale_data = self._inverse_hybrid(original_scale_data, columns)
        
        # Restore original price data if available
        for col in columns:
            orig_col = f'original_{col}'
            if orig_col in normalized_data.columns:
                original_scale_data[col] = normalized_data[orig_col]
                
        return original_scale_data
    
    def _apply_scalers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted scalers to the data.
        
        Args:
            data: Input data
            
        Returns:
            normalized_data: Normalized data
        """
        normalized_data = data.copy()
        
        for group, scaler in self.scalers.items():
            # Get columns for this group that exist in the data
            group_cols = [col for col in self.feature_groups.get(group, []) if col in data.columns]
            
            if not group_cols:
                continue
                
            # Transform the columns
            try:
                normalized_values = scaler.transform(data[group_cols])
                normalized_data[group_cols] = normalized_values
            except Exception as e:
                print(f"Error scaling {group} columns: {e}")
                # Fallback to simpler normalization if scaler fails
                for col in group_cols:
                    normalized_data[col] = (data[col] - data[col].mean()) / (data[col].std() if data[col].std() > 0 else 1)
        
        return normalized_data
    
    def _inverse_apply_scalers(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply inverse transformation using fitted scalers.
        
        Args:
            data: Normalized data
            columns: Columns to inverse transform
            
        Returns:
            original_scale_data: Data in original scale
        """
        original_scale_data = data.copy()
        
        for group, scaler in self.scalers.items():
            # Get columns for this group that exist in both the data and the requested columns
            group_cols = [
                col for col in self.feature_groups.get(group, []) 
                if col in data.columns and col in columns
            ]
            
            if not group_cols:
                continue
                
            # Inverse transform the columns
            try:
                original_values = scaler.inverse_transform(data[group_cols])
                original_scale_data[group_cols] = original_values
            except Exception as e:
                print(f"Error inverse scaling {group} columns: {e}")
        
        return original_scale_data
    
    def _apply_pct_change(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply percentage change normalization.
        
        Args:
            data: Input data
            
        Returns:
            normalized_data: Normalized data
        """
        normalized_data = data.copy()
        
        # Apply to price columns
        price_cols = self.feature_groups.get('price', [])
        for col in price_cols:
            if col in data.columns:
                # Store last price before normalizing
                if self.preserve_scale:
                    self.price_scale_info[col]['last'] = data[col].iloc[-1]
                
                # Calculate percentage change
                normalized_data[col] = data[col].pct_change().fillna(0)
                
        # Apply to other numeric columns that need normalization
        for group in ['volume', 'moving_averages']:
            group_cols = self.feature_groups.get(group, [])
            for col in group_cols:
                if col in data.columns:
                    normalized_data[col] = data[col].pct_change().fillna(0)
        
        return normalized_data
    
    def _inverse_pct_change(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform percentage change normalization.
        
        Args:
            data: Normalized data
            
        Returns:
            original_scale_data: Data in original scale
        """
        original_scale_data = data.copy()
        
        # Inverse transform price columns
        price_cols = self.feature_groups.get('price', [])
        for col in price_cols:
            if col in data.columns and col in self.price_scale_info:
                # Get the last known price
                last_price = self.price_scale_info[col]['last']
                
                # Calculate cumulative product of (1 + percentage change)
                # working backwards from the last known price
                pct_changes = data[col].values
                prices = np.zeros_like(pct_changes)
                prices[-1] = last_price
                
                for i in range(len(prices) - 2, -1, -1):
                    prices[i] = prices[i + 1] / (1 + pct_changes[i + 1])
                
                original_scale_data[col] = prices
                
        return original_scale_data
    
    def _apply_differential(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply differential normalization (first differences).
        
        Args:
            data: Input data
            
        Returns:
            normalized_data: Normalized data
        """
        normalized_data = data.copy()
        
        # Apply to price columns
        price_cols = self.feature_groups.get('price', [])
        for col in price_cols:
            if col in data.columns:
                # Store last price before normalizing
                if self.preserve_scale:
                    self.price_scale_info[col]['last'] = data[col].iloc[-1]
                
                # Calculate differences
                normalized_data[col] = data[col].diff().fillna(0)
                
                # Scale by the average price to make it relative
                if data[col].mean() > 0:
                    normalized_data[col] = normalized_data[col] / data[col].mean()
        
        return normalized_data
    
    def _apply_log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log transformation.
        
        Args:
            data: Input data
            
        Returns:
            normalized_data: Normalized data
        """
        normalized_data = data.copy()
        
        # Apply to price and volume columns
        for group in ['price', 'volume']:
            group_cols = self.feature_groups.get(group, [])
            for col in group_cols:
                if col in data.columns:
                    # Ensure all values are positive
                    min_val = data[col].min()
                    offset = abs(min_val) + 1 if min_val <= 0 else 0
                    
                    # Store transformation info
                    if self.preserve_scale:
                        self.price_scale_info[col]['offset'] = offset
                    
                    # Apply log transform
                    normalized_data[col] = np.log(data[col] + offset)
        
        return normalized_data
    
    def _inverse_log_transform(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Inverse transform log transformation.
        
        Args:
            data: Normalized data
            columns: Columns to inverse transform
            
        Returns:
            original_scale_data: Data in original scale
        """
        original_scale_data = data.copy()
        
        # Inverse transform applicable columns
        for col in columns:
            if col in data.columns and col in self.price_scale_info:
                # Get the offset
                offset = self.price_scale_info[col].get('offset', 0)
                
                # Apply inverse transform
                original_scale_data[col] = np.exp(data[col]) - offset
        
        return original_scale_data
    
    def _apply_hybrid(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply hybrid normalization (percentage change for prices, z-score for others).
        
        Args:
            data: Input data
            
        Returns:
            normalized_data: Normalized data
        """
        normalized_data = data.copy()
        
        # Apply percentage change to price columns
        price_cols = self.feature_groups.get('price', [])
        for col in price_cols:
            if col in data.columns:
                # Store last price before normalizing
                if self.preserve_scale:
                    self.price_scale_info[col]['last'] = data[col].iloc[-1]
                    
                # Store the original close price for reference
                if col == 'close':
                    normalized_data['original_close'] = data['close']
                
                # Calculate percentage change
                normalized_data[col] = data[col].pct_change().fillna(0)
        
        # Apply scalers to other columns
        for group, columns in self.feature_groups.items():
            if group == 'price':
                continue  # Already handled above
                
            if group in self.scalers:
                # Get columns for this group that exist in the data
                group_cols = [col for col in columns if col in data.columns]
                
                if not group_cols:
                    continue
                    
                # Transform the columns
                try:
                    normalized_values = self.scalers[group].transform(data[group_cols])
                    normalized_data[group_cols] = normalized_values
                except Exception as e:
                    print(f"Error scaling {group} columns: {e}")
        
        return normalized_data
    
    def _inverse_hybrid(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Inverse transform hybrid normalization.
        
        Args:
            data: Normalized data
            columns: Columns to inverse transform
            
        Returns:
            original_scale_data: Data in original scale
        """
        original_scale_data = data.copy()
        
        # Separate price columns from others
        price_cols = [col for col in self.feature_groups.get('price', []) if col in columns]
        other_cols = [col for col in columns if col not in price_cols]
        
        # Inverse transform price columns using percentage change method
        if price_cols and self.preserve_scale:
            for col in price_cols:
                if col in data.columns and col in self.price_scale_info:
                    # Get the last known price
                    last_price = self.price_scale_info[col]['last']
                    
                    # Calculate cumulative product of (1 + percentage change)
                    # working backwards from the last known price
                    pct_changes = data[col].values
                    prices = np.zeros_like(pct_changes)
                    prices[-1] = last_price
                    
                    for i in range(len(prices) - 2, -1, -1):
                        prices[i] = prices[i + 1] / (1 + pct_changes[i + 1])
                    
                    original_scale_data[col] = prices
        
        # Inverse transform other columns using scalers
        for group, scaler in self.scalers.items():
            if group == 'price':
                continue  # Already handled above
                
            # Get columns for this group that exist in both the data and the requested columns
            group_cols = [
                col for col in self.feature_groups.get(group, []) 
                if col in data.columns and col in other_cols
            ]
            
            if not group_cols:
                continue
                
            # Inverse transform the columns
            try:
                original_values = scaler.inverse_transform(data[group_cols])
                original_scale_data[group_cols] = original_values
            except Exception as e:
                print(f"Error inverse scaling {group} columns: {e}")
        
        return original_scale_data