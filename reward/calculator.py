"""
reward/calculator.py

Calculates rewards for the trading environment.

The reward function formula is:
Reward = Daily_Return - λ₁ * Daily_Volatility - λ₂ * max(0, (Drawdown - Threshold))
         - λ₃ * (1 - CapitalUtilization) + Trade_Bonus + Position_Bonus - Trading_Cost_Penalty

Author: [Your Name]
Date: March 10, 2025
"""

import numpy as np
from typing import Optional, List


class RewardCalculator:
    """
    Calculates rewards for the trading agent based on multiple factors.
    
    This class implements a comprehensive reward function that considers:
    - Portfolio returns
    - Risk-adjusted performance
    - Drawdown penalties
    - Capital utilization incentives
    - Trading cost penalties
    - Position and trade bonuses
    """
    
    def __init__(
        self,
        scaling: float = 2.0,
        risk_aversion: float = 0.5,
        drawdown_penalty: float = 1.0,
        opportunity_cost: float = 0.05,
        drawdown_threshold: float = 0.05,
        use_smoothing: bool = False,
        smoothing_factor: float = 0.7,
        profitable_trade_bonus: float = 0.2,
        position_bonus: float = 0.0001,
        trading_cost_penalty: float = 0.0003
    ):
        """
        Initialize the reward calculator.
        
        Args:
            scaling: Scaling factor for reward normalization
            risk_aversion: Coefficient for volatility penalty (λ₁)
            drawdown_penalty: Coefficient for drawdown penalty (λ₂)
            opportunity_cost: Coefficient for unused capital penalty (λ₃)
            drawdown_threshold: Acceptable drawdown threshold
            use_smoothing: Whether to apply reward smoothing
            smoothing_factor: Exponential smoothing factor (higher = more smoothing)
            profitable_trade_bonus: Bonus reward for profitable trades
            position_bonus: Small bonus for maintaining positions
            trading_cost_penalty: Penalty for changing positions
        """
        self.scaling = scaling
        self.risk_aversion = risk_aversion
        self.drawdown_penalty = drawdown_penalty
        self.opportunity_cost = opportunity_cost
        self.drawdown_threshold = drawdown_threshold
        self.use_smoothing = use_smoothing
        self.smoothing_factor = smoothing_factor
        self.profitable_trade_bonus = profitable_trade_bonus
        self.position_bonus = position_bonus
        self.trading_cost_penalty = trading_cost_penalty
        
        # For reward smoothing
        self.last_reward = 0.0
        
        # For tracking historical context
        self.portfolio_values = []
        self.cumulative_return = 0.0
        self.recent_trades = []
    
    def calculate_reward(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float,
        prev_position: float,
        current_position: float,
        drawdown: float,
        daily_volatility: float,
        capital_utilization: float,
        trade_completed: bool = False,
        trade_profit: float = 0.0,
        current_step: Optional[int] = None
    ) -> float:
        """
        Calculate the reward based on the defined formula.
        
        Args:
            prev_portfolio_value: Portfolio value at previous step
            current_portfolio_value: Portfolio value at current step
            prev_position: Position size at previous step
            current_position: Position size at current step
            drawdown: Current drawdown from peak
            daily_volatility: Daily price volatility
            capital_utilization: Proportion of capital currently utilized
            trade_completed: Whether a trade was completed this step
            trade_profit: Profit/loss from completed trade (if any)
            current_step: Current step in the episode
            
        Returns:
            reward: Calculated reward value
        """
        # Track portfolio values
        self.portfolio_values.append(current_portfolio_value)
        
        # Calculate daily return (as a percentage)
        if prev_portfolio_value > 0:
            daily_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            self.cumulative_return += daily_return
        else:
            daily_return = 0.0
            
        # Position change penalty - discourage excessive trading
        position_changed = current_position != prev_position
        trading_cost_penalty = self.trading_cost_penalty if position_changed else 0.0
        
        # Adding a position bonus to encourage the agent to take positions
        position_bonus = self.position_bonus if current_position > 0 else 0.0
            
        # Calculate drawdown penalty
        # Apply penalty only if drawdown exceeds threshold
        drawdown_excess = max(0, drawdown - self.drawdown_threshold)
        drawdown_penalty = self.drawdown_penalty * drawdown_excess
        
        # Calculate capital utilization penalty
        # This incentivizes using available capital appropriately
        utilization_penalty = self.opportunity_cost * (1 - capital_utilization)
        
        # Calculate volatility penalty
        volatility_penalty = self.risk_aversion * daily_volatility
        
        # Add bonus for profitable trades
        trade_bonus = 0.0
        if trade_completed and trade_profit > 0:
            # Scale by portfolio value for percentage
            trade_bonus = self.profitable_trade_bonus * (trade_profit / prev_portfolio_value if prev_portfolio_value > 0 else 0)
            
            # Track trade for analytics
            if current_step is not None:
                self.recent_trades.append((current_step, trade_profit))
                # Keep only the last 20 trades
                if len(self.recent_trades) > 20:
                    self.recent_trades = self.recent_trades[-20:]
        
        # Combine all components for the final reward
        reward = (daily_return + trade_bonus + position_bonus - 
                  volatility_penalty - drawdown_penalty - 
                  utilization_penalty - trading_cost_penalty)
        
        # Scale the reward
        reward = reward * self.scaling
        
        # Apply reward smoothing if enabled
        if self.use_smoothing:
            reward = self.smooth_reward(reward)
            
        # Apply tanh transformation to keep rewards in a reasonable range
        # Using tanh with a scaling factor to not overly compress the reward
        reward = np.tanh(reward * 2)
        
        return reward
    
    def smooth_reward(self, raw_reward: float) -> float:
        """
        Apply exponential smoothing to the reward.
        
        Args:
            raw_reward: The calculated raw reward
            
        Returns:
            smoothed_reward: The smoothed reward
        """
        smoothed_reward = (self.smoothing_factor * self.last_reward) + \
                          ((1 - self.smoothing_factor) * raw_reward)
        self.last_reward = smoothed_reward
        return smoothed_reward
    
    def reset(self):
        """
        Reset the reward calculator state.
        """
        self.last_reward = 0.0
        self.portfolio_values = []
        self.cumulative_return = 0.0
        self.recent_trades = []
    
    def get_trade_statistics(self) -> dict:
        """
        Get statistics about recent trades.
        
        Returns:
            dict: Statistics about recent trades
        """
        if not self.recent_trades:
            return {
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'profit_loss_ratio': 0.0,
                'expectancy': 0.0
            }
        
        # Extract profit/loss from trades
        profits_losses = [trade[1] for trade in self.recent_trades]
        
        # Count winning and losing trades
        winning_trades = [p for p in profits_losses if p > 0]
        losing_trades = [p for p in profits_losses if p <= 0]
        
        # Calculate statistics
        win_rate = len(winning_trades) / len(profits_losses) if profits_losses else 0.0
        avg_profit = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0.0
        expectancy = (win_rate * avg_profit) + ((1 - win_rate) * avg_loss) if profits_losses else 0.0
        
        return {
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'expectancy': expectancy
        }