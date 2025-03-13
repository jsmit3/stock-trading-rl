"""
This is a replacement for the relevant parts of the ObservationGenerator class.
Add these methods to your observation/generator.py file.
"""

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
        # List to hold all observation components
        observation_components = []
        
        # Get latest data point
        latest_data = market_data.iloc[-1]
        
        # 1. Price Data Features
        if self.feature_flags['price_data']:
            price_features = self._get_price_features(market_data)
            observation_components.append(price_features.flatten())
            
        # 2. Volume Features
        if self.feature_flags['volume_data']:
            volume_features = self._get_volume_features(market_data)
            observation_components.append(volume_features.flatten())
            
        # 3. Technical Indicators
        tech_indicator_features = []
        
        # 3.1 Trend Indicators
        if self.feature_flags['trend_indicators']:
            trend_features = self._get_trend_indicators(market_data)
            for feat in trend_features:
                tech_indicator_features.append(feat.flatten())
            
        # 3.2 Momentum Indicators
        if self.feature_flags['momentum_indicators']:
            momentum_features = self._get_momentum_indicators(market_data)
            for feat in momentum_features:
                tech_indicator_features.append(feat.flatten())
            
        # 3.3 Volatility Indicators
        if self.feature_flags['volatility_indicators']:
            volatility_features = self._get_volatility_indicators(market_data)
            for feat in volatility_features:
                tech_indicator_features.append(feat.flatten())
            
        # 3.4 Volume Indicators
        if self.feature_flags['volume_indicators']:
            volume_ind_features = self._get_volume_indicators(market_data)
            for feat in volume_ind_features:
                tech_indicator_features.append(feat.flatten())
            
        # Combine all technical indicators
        if tech_indicator_features:
            for feat in tech_indicator_features:
                observation_components.append(feat)
        
        # 4. Market Context
        if self.feature_flags['market_context'] and self.include_market_context and market_index_data is not None:
            market_context_features = self._get_market_context(market_data, market_index_data)
            observation_components.append(market_context_features.flatten())
            
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
            
        # Combine all components into a flat vector
        flat_observation = np.concatenate([comp for comp in observation_components])
        
        # Store the dimension for consistency
        if not hasattr(self, 'observation_dim'):
            self.observation_dim = flat_observation.shape[0]
            print(f"Setting observation_dim to {self.observation_dim}")
        
        # CRITICAL: Force consistent dimensions by either padding or truncating
        if flat_observation.shape[0] != self.observation_dim:
            print(f"WARNING: Observation dimension mismatch: actual={flat_observation.shape[0]}, expected={self.observation_dim}")
            if flat_observation.shape[0] < self.observation_dim:
                # Pad with zeros
                padded = np.zeros(self.observation_dim)
                padded[:flat_observation.shape[0]] = flat_observation
                flat_observation = padded
            else:
                # Truncate
                flat_observation = flat_observation[:self.observation_dim]
            
        return flat_observation.astype(np.float32)
        
    except Exception as e:
        print(f"Error in generate_observation: {e}")
        # Return zeros with the stored dimension or a default if not set yet
        dim = getattr(self, 'observation_dim', 325)
        return np.zeros(dim, dtype=np.float32)

def lock_observation_dimension(self, dim=325):
    """
    Explicitly lock the observation dimension to a fixed size.
    
    Args:
        dim: The fixed dimension to use for all observations
    """
    self.observation_dim = dim
    print(f"Locked observation dimension to {dim}")