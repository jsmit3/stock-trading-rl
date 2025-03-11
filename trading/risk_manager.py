"""
trading/risk_manager.py

Manages risk for trading operations in the environment.

This module handles:
- Position sizing based on risk parameters
- Risk adjustment based on market conditions
- Trade result tracking and adaptive risk management
- Volatility-based position sizing
- Gap protection for overnight and weekend positions
- Maximum consecutive loss protection
- VIX-based position scaling

Author: [Your Name]
Date: March 10, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union


class RiskManager:
    """
    Manages trading risk in the environment.
    
    Adjusts position sizes based on risk parameters, market conditions,
    and past trading performance.
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.25,
        min_position_pct: float = 0.05,
        max_risk_per_trade: float = 0.02,
        volatility_scaling: bool = True,
        adaptive_sizing: bool = True,
        max_drawdown_limit: float = 0.25,
        enable_weekend_adjustment: bool = True,
        enable_earnings_adjustment: bool = True,
        enable_consecutive_loss_protection: bool = True,
        enable_vix_scaling: bool = True
    ):
        """
        Initialize the risk manager.
        
        Args:
            max_position_pct: Maximum position size as % of capital
            min_position_pct: Minimum position size as % of capital
            max_risk_per_trade: Maximum risk per trade as % of capital
            volatility_scaling: Whether to scale position size by volatility
            adaptive_sizing: Whether to adapt position size based on performance
            max_drawdown_limit: Maximum drawdown limit
            enable_weekend_adjustment: Whether to reduce size before weekends
            enable_earnings_adjustment: Whether to reduce size before earnings
            enable_consecutive_loss_protection: Whether to reduce size after losses
            enable_vix_scaling: Whether to scale positions based on VIX
        """
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_risk_per_trade = max_risk_per_trade
        self.volatility_scaling = volatility_scaling
        self.adaptive_sizing = adaptive_sizing
        self.max_drawdown_limit = max_drawdown_limit
        self.enable_weekend_adjustment = enable_weekend_adjustment
        self.enable_earnings_adjustment = enable_earnings_adjustment
        self.enable_consecutive_loss_protection = enable_consecutive_loss_protection
        self.enable_vix_scaling = enable_vix_scaling
        
        # Performance tracking
        self.trade_results: List[float] = []
        self.win_rate = 0.5
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.current_drawdown = 0.0
        
        # Internal state
        self.consecutive_losses = 0
        self.last_trade_profit = 0.0
        self.daily_loss_occurred = False
        self.daily_loss_cooldown = 0
        self.is_earnings_season = False
    
    def adjust_position_size(
        self,
        position_size_pct: float,
        current_data: pd.Series,
        current_step: int,
        price_data: pd.DataFrame,
        day_of_week: Optional[int] = None,
        vix_value: Optional[float] = None,
        near_earnings: bool = False
    ) -> float:
        """
        Adjust position size based on risk parameters and market conditions.
        
        Args:
            position_size_pct: Requested position size as percentage of capital
            current_data: Current market data
            current_step: Current step in the environment
            price_data: Full price data
            day_of_week: Day of week (0=Monday, 6=Sunday)
            vix_value: Current VIX index value
            near_earnings: Whether near earnings announcement
            
        Returns:
            adjusted_position_size: Adjusted position size as percentage of capital
        """
        # Ensure position size is within basic limits
        adjusted_position_size = np.clip(position_size_pct, 0, self.max_position_pct)
        
        # Get volatility information
        volatility = current_data.get('atr_pct', current_data.get('volatility', 0.01))
        
        # Apply volatility scaling if enabled
        if self.volatility_scaling:
            vol_scaling_factor = self._calculate_volatility_factor(volatility)
            adjusted_position_size *= vol_scaling_factor
        
        # Apply weekend adjustment (Friday = day 4)
        if self.enable_weekend_adjustment and day_of_week == 4:
            adjusted_position_size *= 0.8  # Reduce by 20% on Fridays
            
        # Apply earnings adjustment
        if self.enable_earnings_adjustment and near_earnings:
            adjusted_position_size *= 0.5  # Reduce by 50% near earnings
            
        # Apply consecutive loss protection
        if self.enable_consecutive_loss_protection:
            consecutive_loss_factor = self._get_consecutive_loss_factor()
            adjusted_position_size *= consecutive_loss_factor
            
        # Apply daily loss cooldown
        if self.daily_loss_cooldown > 0:
            adjusted_position_size *= 0.5  # Reduce by 50% during cooldown
            
        # Apply VIX-based scaling
        if self.enable_vix_scaling and vix_value is not None:
            vix_factor = self._calculate_vix_factor(vix_value)
            adjusted_position_size *= vix_factor
        
        # Apply risk-based sizing
        if 'atr' in current_data and current_data['atr'] > 0:
            # Calculate position size based on ATR for consistent dollar risk
            price = current_data['close']
            atr = current_data['atr']
            
            # Use ATR-multiple as stop distance (e.g., 2 * ATR)
            stop_distance_pct = (2 * atr) / price
            
            # Calculate position size based on risk per trade
            risk_based_size = self.max_risk_per_trade / stop_distance_pct
            
            # Use the minimum of risk-based size and original adjusted size
            adjusted_position_size = min(adjusted_position_size, risk_based_size)
        
        # Apply adaptive sizing based on performance if enabled
        if self.adaptive_sizing and len(self.trade_results) >= 5:
            # If win rate is poor, reduce position size
            if self.win_rate < 0.4:
                adjusted_position_size *= 0.8
            # If win rate is very poor, reduce more dramatically
            if self.win_rate < 0.3:
                adjusted_position_size *= 0.5
            
            # If drawdown is high, reduce position size
            if self.current_drawdown > 0.1:
                drawdown_factor = 1.0 - (self.current_drawdown / self.max_drawdown_limit)
                drawdown_factor = np.clip(drawdown_factor, 0.25, 1.0)
                adjusted_position_size *= drawdown_factor
            
        # Ensure position size doesn't go below minimum if it was non-zero to begin with
        if position_size_pct > 0:
            adjusted_position_size = max(adjusted_position_size, self.min_position_pct)
        
        return adjusted_position_size
    
    def calculate_position_size(
        self,
        capital: float,
        price: float,
        stop_loss_pct: float,
        volatility: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate position size based on fixed-risk principle.
        
        Uses the formula: Position Size = (Capital × Risk) ÷ (Price × Stop Distance)
        
        Args:
            capital: Available capital
            price: Current price
            stop_loss_pct: Stop loss percentage
            volatility: Current market volatility
            
        Returns:
            num_shares: Number of shares to buy
            position_size_pct: Position size as percentage of capital
        """
        # Ensure input values are valid
        if capital <= 0 or price <= 0 or stop_loss_pct <= 0:
            return 0.0, 0.0
        
        # Ensure stop loss is not too small
        stop_loss_pct = max(stop_loss_pct, 0.005)  # Minimum 0.5% stop loss
        
        # Calculate dollar risk amount based on max risk per trade
        dollar_risk = capital * self.max_risk_per_trade
        
        # Calculate dollar stop loss amount
        stop_loss_amount = price * stop_loss_pct
        
        # Calculate position size in shares
        num_shares = dollar_risk / stop_loss_amount if stop_loss_amount > 0 else 0
        
        # Calculate position size as percentage of capital
        position_size_pct = (num_shares * price) / capital if capital > 0 else 0
        
        # Apply volatility adjustment if provided
        if volatility is not None:
            volatility_factor = self._calculate_volatility_factor(volatility)
            position_size_pct *= volatility_factor
            num_shares = (position_size_pct * capital) / price if price > 0 else 0
        
        # Enforce position size limits
        if position_size_pct > self.max_position_pct:
            position_size_pct = self.max_position_pct
            num_shares = (position_size_pct * capital) / price if price > 0 else 0
        elif position_size_pct < self.min_position_pct and position_size_pct > 0:
            position_size_pct = self.min_position_pct
            num_shares = (position_size_pct * capital) / price if price > 0 else 0
            
        return num_shares, position_size_pct
    
    def update_trade_result(self, profit_loss: float) -> None:
        """
        Update internal state based on trade result.
        
        Args:
            profit_loss: Profit/loss from the trade as a decimal (e.g., 0.05 for 5%)
        """
        # Add to trade history
        self.trade_results.append(profit_loss)
        
        # Keep only the last 20 trades for calculations
        if len(self.trade_results) > 20:
            self.trade_results = self.trade_results[-20:]
        
        # Update win rate
        wins = sum(1 for r in self.trade_results if r > 0)
        self.win_rate = wins / len(self.trade_results) if self.trade_results else 0.5
        
        # Update average win and loss
        win_returns = [r for r in self.trade_results if r > 0]
        loss_returns = [r for r in self.trade_results if r <= 0]
        
        self.avg_win = np.mean(win_returns) if win_returns else 0.0
        self.avg_loss = np.mean(loss_returns) if loss_returns else 0.0
        
        # Update drawdown calculation (simplified)
        if len(self.trade_results) >= 5:
            recent_returns = self.trade_results[-5:]
            cum_return = np.prod(1 + np.array(recent_returns)) - 1
            self.current_drawdown = max(0, -cum_return)
        
        # Track consecutive losses
        if profit_loss < 0:
            self.consecutive_losses += 1
            
            # Check for daily loss threshold (-5%)
            if profit_loss < -0.05:
                self.daily_loss_occurred = True
                self.daily_loss_cooldown = 3  # Cooldown for 3 days
        else:
            self.consecutive_losses = 0  # Reset on a winning trade
            
        # Store last trade result
        self.last_trade_profit = profit_loss
    
    def update_daily_state(self) -> None:
        """
        Update internal state for the next day.
        """
        # Decrease cooldown counter if active
        if self.daily_loss_cooldown > 0:
            self.daily_loss_cooldown -= 1
            
        # Reset daily loss flag
        self.daily_loss_occurred = False
    
    def get_stop_loss_for_gap_risk(
        self,
        base_stop_loss_pct: float,
        volatility: float,
        is_weekend: bool = False
    ) -> float:
        """
        Calculate adjusted stop loss to account for gap risk.
        
        Args:
            base_stop_loss_pct: Base stop loss percentage
            volatility: Current market volatility
            is_weekend: Whether position held over weekend
            
        Returns:
            adjusted_stop_loss: Adjusted stop loss percentage
        """
        # Ensure base stop loss is valid
        if base_stop_loss_pct <= 0:
            return 0.01  # Default to 1% stop loss
        
        # Add a buffer based on volatility to account for gaps
        volatility_buffer = volatility * 1.5
        
        # Add extra buffer for weekend
        if is_weekend:
            weekend_buffer = volatility * 0.5
        else:
            weekend_buffer = 0
            
        # Calculate adjusted stop loss
        adjusted_stop_loss = base_stop_loss_pct + volatility_buffer + weekend_buffer
        
        # Cap at reasonable levels
        adjusted_stop_loss = min(adjusted_stop_loss, 0.20)  # Maximum 20% stop loss
        
        return adjusted_stop_loss
    
    def get_max_loss_override(self) -> float:
        """
        Get the maximum open loss limit regardless of stop loss.
        
        Returns:
            max_loss_pct: Maximum loss percentage allowed
        """
        return 0.15  # Force exit at -15% regardless of stop loss
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get current risk metrics.
        
        Returns:
            metrics: Dictionary of risk metrics
        """
        return {
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'risk_reward_ratio': abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else 0,
            'expectancy': (self.win_rate * self.avg_win) + ((1 - self.win_rate) * self.avg_loss),
            'current_drawdown': self.current_drawdown,
            'consecutive_losses': self.consecutive_losses
        }
    
    def _calculate_volatility_factor(self, volatility: float) -> float:
        """
        Calculate position sizing factor based on volatility.
        
        Args:
            volatility: Current market volatility (e.g., ATR as percentage)
            
        Returns:
            volatility_factor: Position sizing adjustment factor
        """
        # Base volatility is considered to be 1% daily, scale accordingly
        # Higher volatility = smaller position size
        base_volatility = 0.01  # 1% daily volatility as baseline
        
        if volatility <= 0:
            return 1.0  # Avoid division by zero
            
        volatility_factor = base_volatility / volatility
        
        # Limit the range of the volatility factor
        volatility_factor = max(min(volatility_factor, 2.0), 0.2)
        
        return volatility_factor
    
    def _calculate_vix_factor(self, vix_value: float) -> float:
        """
        Calculate position sizing factor based on VIX.
        
        Args:
            vix_value: Current VIX index value
            
        Returns:
            vix_factor: Position sizing adjustment factor
        """
        # Base VIX level is considered to be 20, scale accordingly
        # Higher VIX = smaller position size
        base_vix = 20.0
        
        # Formula: Adjusted Size = Original Size * (20 / max(20, VIX))
        vix_factor = base_vix / max(base_vix, vix_value)
        
        return vix_factor
    
    def _get_consecutive_loss_factor(self) -> float:
        """
        Get position sizing factor based on consecutive losses.
        
        Returns:
            consecutive_loss_factor: Position sizing adjustment factor
        """
        if self.consecutive_losses >= 3:
            return 0.5  # Reduce position size by 50% after 3 consecutive losses
        elif self.consecutive_losses >= 2:
            return 0.7  # Reduce position size by 30% after 2 consecutive losses
        elif self.consecutive_losses >= 1:
            return 0.9  # Reduce position size by 10% after 1 consecutive loss
        else:
            return 1.0  # No adjustment