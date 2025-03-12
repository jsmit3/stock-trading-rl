"""
trading_env.py

This module implements a custom Gymnasium environment for stock trading
with realistic market dynamics, designed for reinforcement learning applications.

Fixed to handle edge cases in action parsing and prevent negative position sizes.

Author: [Your Name]
Date: March 2, 2025
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List, Union
import matplotlib.pyplot as plt
import random

from observation_space import ObservationGenerator
from action_space import ActionInterpreter
from reward_function import RewardCalculator
from market_simulator import MarketSimulator
from archive.data_processor import DataProcessor
from utils.risk_management import RiskManager


class StockTradingEnv(gym.Env):
    """
    A Gymnasium environment for stock trading using the SAC algorithm.
    
    This environment simulates daily stock trading with realistic constraints:
    - Single stock trading
    - Long-only positions
    - Risk-adjusted position sizing
    - Stop-loss and take-profit mechanisms
    - Handling of market gaps and transaction costs
    - Maximum holding periods
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        initial_capital: float = 100000.0,
        max_holding_period: int = 20,
        transaction_cost_pct: float = 0.0015,
        window_size: int = 20,
        reward_scaling: float = 2.0,
        risk_aversion: float = 0.5,
        drawdown_penalty: float = 1.0,
        opportunity_cost: float = 0.05,
        drawdown_threshold: float = 0.05,
        max_drawdown_pct: float = 0.25,
        include_sentiment: bool = False,
        max_position_pct: float = 0.3,
        min_position_pct: float = 0.05,
        max_episodes: Optional[int] = None,
        curriculum_level: int = 1,
        debug_mode: bool = False,
        min_episode_length: int = 20  # Added minimum episode length
    ):
        """
        Initialize the stock trading environment.
        
        Args:
            price_data: DataFrame with OHLCV data and dates as index
            initial_capital: Starting capital amount
            max_holding_period: Maximum number of days to hold a position
            transaction_cost_pct: Combined spread and slippage costs
            window_size: Number of past days to include in observation
            reward_scaling: Scaling factor for reward normalization
            risk_aversion: Coefficient for volatility penalty (λ₁)
            drawdown_penalty: Coefficient for drawdown penalty (λ₂)
            opportunity_cost: Coefficient for unused capital penalty (λ₃)
            drawdown_threshold: Acceptable drawdown threshold
            max_drawdown_pct: Maximum drawdown before early termination
            include_sentiment: Whether to include sentiment features
            max_position_pct: Maximum position size as % of capital
            min_position_pct: Minimum position size as % of capital
            max_episodes: Maximum number of episodes (for curriculum learning)
            curriculum_level: Current curriculum learning level (1-3)
            debug_mode: Whether to print debug information
            min_episode_length: Minimum number of steps before allowing early termination
        """
        super(StockTradingEnv, self).__init__()
        
        # Store configuration parameters
        self.initial_capital = initial_capital
        self.max_holding_period = max_holding_period
        self.transaction_cost_pct = transaction_cost_pct
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.risk_aversion = risk_aversion
        self.drawdown_penalty = drawdown_penalty
        self.opportunity_cost = opportunity_cost
        self.drawdown_threshold = drawdown_threshold
        self.max_drawdown_pct = max_drawdown_pct
        self.include_sentiment = include_sentiment
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_episodes = max_episodes
        self.curriculum_level = curriculum_level
        self.debug_mode = debug_mode
        self.min_episode_length = min_episode_length
        
        # Process and validate input data
        self.data_processor = DataProcessor(price_data, window_size=window_size)
        self.price_data = self.data_processor.process_data()
        self.dates = self.price_data.index.tolist()
        
        # Apply curriculum learning parameters
        self._apply_curriculum_settings()
        
        # Initialize component modules
        self.observation_generator = ObservationGenerator(
            window_size=window_size, 
            include_sentiment=include_sentiment
        )
        
        self.action_interpreter = ActionInterpreter(
            max_position_pct=self.max_position_pct,
            min_position_pct=self.min_position_pct
        )
        
        self.reward_calculator = RewardCalculator(
            scaling=self.reward_scaling,
            risk_aversion=self.risk_aversion,
            drawdown_penalty=self.drawdown_penalty,
            opportunity_cost=self.opportunity_cost,
            drawdown_threshold=self.drawdown_threshold,
            profitable_trade_bonus=0.2  # Added bonus for profitable trades
        )
        
        self.risk_manager = RiskManager(
            max_position_pct=self.max_position_pct,
            min_position_pct=self.min_position_pct,
            max_risk_per_trade=self.max_risk_per_trade
        )
        
        self.market_simulator = MarketSimulator(
            transaction_cost_pct=self.transaction_cost_pct
        )
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),  # Position size, stop-loss, take-profit, exit signal
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.current_step = self.window_size
        self.current_position = 0.0
        self.current_price = self.price_data.iloc[self.window_size]['close']
        self.position_entry_price = 0.0
        self.position_entry_step = 0
        self.stop_loss_pct = 0.0
        self.take_profit_pct = 0.0
        self.cash_balance = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        self.drawdown = 0.0
        self.position_pnl = 0.0
        self.total_pnl = 0.0
        
        # Generate a dummy observation to determine observation space shape
        dummy_observation = self._get_observation()
        
        # Define the observation space now that we know the shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_observation.shape,
            dtype=np.float32
        )
        
        # Initialize tracking variables for performance metrics
        self.returns_history = []
        self.positions_history = []
        self.actions_history = []
        self.rewards_history = []
        self.trade_history = []
        
        # Diagnostic counters for monitoring
        self.stop_loss_count = 0
        self.take_profit_count = 0
        self.max_holding_count = 0
        self.exit_signal_count = 0
        self.early_termination_count = 0
        
        # Episode step counter for minimum episode length enforcement
        self.episode_step_count = 0
        
        # Reset the environment state
        self.episode_count = 0
        self._reset_called = False
    
    def _apply_curriculum_settings(self):
        """Apply settings based on curriculum level"""
        # Set risk parameters based on curriculum level
        if self.curriculum_level == 1:
            # Beginner level - Lower risk, clearer patterns
            self.max_risk_per_trade = 0.01  # 1% max risk per trade
            self.transaction_cost_pct = 0.0005  # Lower transaction costs
            self.max_drawdown_pct = 0.35  # More conservative early stopping (increased from 0.15)
            
        elif self.curriculum_level == 2:
            # Intermediate level
            self.max_risk_per_trade = 0.015  # 1.5% max risk per trade
            self.transaction_cost_pct = 0.001  # Medium transaction costs
            self.max_drawdown_pct = 0.40  # Medium drawdown tolerance (increased from 0.20)
            
        else:  # Level 3 or higher
            # Advanced level - Full complexity
            self.max_risk_per_trade = 0.02  # 2% max risk per trade
            self.transaction_cost_pct = 0.0015  # Full transaction costs
            self.max_drawdown_pct = 0.45  # Higher drawdown tolerance (increased from 0.25)
            
        # Make sure the risk manager has the updated risk parameters
        if hasattr(self, 'risk_manager'):
            self.risk_manager.max_risk_per_trade = self.max_risk_per_trade
    
    def advance_curriculum(self):
        """Advance to the next curriculum level if not at max"""
        if self.curriculum_level < 3:
            self.curriculum_level += 1
            self._apply_curriculum_settings()
            return True
        return False
        
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset behavior
            
        Returns:
            observation: The initial observation
            info: Additional information about the environment state
        """
        super().reset(seed=seed)
        
        # Initialize episode starting point
        # Default to beginning of data after window_size
        start_idx = self.window_size
        
        # If options provided and contains custom start index
        if options and 'start_idx' in options and options['start_idx'] is not None:
            start_idx = max(self.window_size, min(options['start_idx'], len(self.price_data) - 1))
        else:
            # Randomly select a start point between window_size and 3/4 of the data
            # This helps avoid always starting at the same point
            max_start = int(len(self.price_data) * 0.75)
            start_idx = random.randint(self.window_size, max_start)
        
        # Start somewhere in the data, after the required window_size
        # but with enough data left to have a meaningful episode
        self.current_step = start_idx
        
        # Reset financial state
        self.cash_balance = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        self.drawdown = 0.0
        
        # Reset position state
        self.current_position = 0.0
        self.position_entry_price = 0.0
        self.position_entry_step = 0
        self.stop_loss_pct = 0.0
        self.take_profit_pct = 0.0
        self.position_pnl = 0.0
        self.total_pnl = 0.0
        
        # Reset tracking variables for this episode
        self.returns_history = []
        self.positions_history = []
        self.actions_history = []
        self.rewards_history = []
        self.trade_history = []
        
        # Reset diagnostic counters
        self.stop_loss_count = 0
        self.take_profit_count = 0
        self.max_holding_count = 0
        self.exit_signal_count = 0
        
        # Reset episode step counter
        self.episode_step_count = 0
        
        # Get current market data
        self.current_price = self.price_data.iloc[self.current_step]['close']
        
        # Reset reward calculator
        self.reward_calculator.reset()
        
        # Generate initial observation
        observation = self._get_observation()
        
        # Mark reset as called
        self._reset_called = True
        self.episode_count += 1
        
        # Return initial observation and info
        info = {
            'step': self.current_step,
            'date': self.dates[self.current_step],
            'price': self.current_price,
            'portfolio_value': self.portfolio_value,
            'cash_balance': self.cash_balance,
            'current_position': self.current_position,
            'position_pnl': self.position_pnl,
            'total_pnl': self.total_pnl,
            'drawdown': self.drawdown,
            'initial_capital': self.initial_capital,
            'curriculum_level': self.curriculum_level
        }
        
        if self.debug_mode:
            print(f"Episode {self.episode_count} started at step {self.current_step}, date {self.dates[self.current_step]}")
            print(f"Data goes from {self.dates[0]} to {self.dates[-1]} ({len(self.dates)} days)")
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Array with values between 0 and 1 representing:
                   [position_size, stop_loss, take_profit, exit_signal]
                   
        Returns:
            observation: The new observation after action
            reward: The reward for taking the action
            terminated: Whether the episode has ended
            truncated: Whether the episode was truncated (e.g., max steps)
            info: Additional information about the step
        """
        if not self._reset_called:
            raise RuntimeError("Call reset before using step method.")
            
        # Increment step counter
        self.episode_step_count += 1
            
        # Record the action
        self.actions_history.append(action.copy())
        
        # Get current market state
        current_data = self.price_data.iloc[self.current_step]
        prev_data = self.price_data.iloc[self.current_step - 1]
        
        # Store previous state for reward calculation
        prev_portfolio_value = self.portfolio_value
        prev_position = self.current_position
        prev_price = prev_data['close']
        
        # Ensure action is valid (no NaN or infinite values)
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            if self.debug_mode:
                print(f"WARNING: Invalid action detected: {action}, using zeros instead")
            action = np.zeros_like(action)
        
        # Clip action to valid range [0, 1]
        action = np.clip(action, 0, 1)
        
        # Interpret the action into trading decision
        position_size_pct, stop_loss_pct, take_profit_pct, exit_signal = \
            self.action_interpreter.interpret_action(action, current_data)
        
        # Double check position size is non-negative 
        if position_size_pct < 0:
            if self.debug_mode:
                print(f"WARNING: Negative position size after interpreter: {position_size_pct}, setting to 0")
            position_size_pct = 0.0
            
        # Force minimum values for stop-loss and take-profit to ensure they're valid
        stop_loss_pct = max(stop_loss_pct, 0.005)  # At least 0.5%
        take_profit_pct = max(take_profit_pct, 0.005)  # At least 0.5%
            
        # Apply risk management adjustments
        try:
            position_size_pct = self.risk_manager.adjust_position_size(
                position_size_pct, 
                current_data, 
                self.current_step,
                self.price_data,
                day_of_week=self.dates[self.current_step].dayofweek
            )
            # Triple check position size is non-negative 
            if position_size_pct < 0:
                if self.debug_mode:
                    print(f"WARNING: Negative position size after risk manager: {position_size_pct}, setting to 0")
                position_size_pct = 0.0
        except Exception as e:
            if self.debug_mode:
                print(f"Error in risk manager: {e}, setting position_size_pct to 0")
            position_size_pct = 0.0
        
        # Execute trading logic
        trade_info = self._execute_trading_logic(
            position_size_pct, 
            stop_loss_pct, 
            take_profit_pct, 
            exit_signal,
            current_data
        )
        
        # Track if a trade was completed
        trade_completed = trade_info.get('trade_completed', False) if trade_info else False
        trade_profit = trade_info.get('trade_profit', 0.0) if trade_info else 0.0
        
        # Move to the next day
        self.current_step += 1
        truncated = self.current_step >= len(self.price_data) - 1
        
        # Update current price
        if not truncated:
            self.current_price = self.price_data.iloc[self.current_step]['close']
            
            # Check for stop-loss or take-profit hits
            if self.current_position > 0:
                # Update position days held
                days_held = self.current_step - self.position_entry_step
                
                # Calculate current position P&L
                self.position_pnl = ((self.current_price / self.position_entry_price) - 1.0) * 100
                
                # Check for stop-loss hit (using low price to be realistic)
                low_price = self.price_data.iloc[self.current_step]['low']
                if low_price <= self.position_entry_price * (1 - self.stop_loss_pct) and self.stop_loss_pct > 0:
                    # Stop-loss hit
                    trade_info = self._close_position(reason="stop_loss")
                    self.stop_loss_count += 1
                    trade_completed = True
                    trade_profit = trade_info.get('trade_profit', 0.0) if trade_info else 0.0
                
                # Check for take-profit hit (using high price to be realistic)
                high_price = self.price_data.iloc[self.current_step]['high']
                if high_price >= self.position_entry_price * (1 + self.take_profit_pct) and self.take_profit_pct > 0:
                    # Take-profit hit
                    trade_info = self._close_position(reason="take_profit")
                    self.take_profit_count += 1
                    trade_completed = True
                    trade_profit = trade_info.get('trade_profit', 0.0) if trade_info else 0.0
                    
                # Check for maximum holding period
                if days_held >= self.max_holding_period:
                    # Max holding period reached
                    trade_info = self._close_position(reason="max_holding_period")
                    self.max_holding_count += 1
                    trade_completed = True
                    trade_profit = trade_info.get('trade_profit', 0.0) if trade_info else 0.0
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            prev_portfolio_value=prev_portfolio_value,
            current_portfolio_value=self.portfolio_value,
            prev_position=prev_position,
            current_position=self.current_position,
            drawdown=self.drawdown,
            daily_volatility=current_data.get('volatility', 0.01),
            capital_utilization=self.current_position * self.current_price / self.portfolio_value if self.portfolio_value > 0 else 0,
            trade_completed=trade_completed,
            trade_profit=trade_profit
        )
        
        # Record the reward
        self.rewards_history.append(reward)
        
        # Get the next observation
        observation = self._get_observation()
        
        # Check for bankruptcy or max drawdown termination conditions
        terminated = False
        termination_reason = None
        
        # Only allow early termination if we've gone past the minimum episode length
        if self.episode_step_count >= self.min_episode_length:
            if self.portfolio_value <= 0:
                terminated = True
                termination_reason = "bankruptcy"
                self.early_termination_count += 1
                if self.debug_mode:
                    print(f"Episode terminated due to bankruptcy at step {self.current_step}")
                    
            elif self.drawdown >= self.max_drawdown_pct:
                terminated = True
                termination_reason = f"max_drawdown_exceeded (drawdown: {self.drawdown:.2%}, threshold: {self.max_drawdown_pct:.2%})"
                self.early_termination_count += 1
                if self.debug_mode:
                    print(f"Episode terminated due to max drawdown exceeded at step {self.current_step}")
                    print(f"Drawdown: {self.drawdown:.2%}, Threshold: {self.max_drawdown_pct:.2%}")
        
        # Prepare info dictionary
        info = {
            'step': self.current_step,
            'date': self.dates[self.current_step] if not truncated and self.current_step < len(self.dates) else self.dates[-1],
            'price': self.current_price,
            'portfolio_value': self.portfolio_value,
            'cash_balance': self.cash_balance,
            'current_position': self.current_position,
            'position_pnl': self.position_pnl,
            'total_pnl': self.total_pnl,
            'drawdown': self.drawdown,
            'initial_capital': self.initial_capital,
            'curriculum_level': self.curriculum_level,
            'action': {
                'position_size': position_size_pct,
                'stop_loss': stop_loss_pct,
                'take_profit': take_profit_pct,
                'exit_signal': exit_signal
            },
            'episode_steps': self.episode_step_count,
            'min_episode_length': self.min_episode_length
        }
        
        # Add trade info if a trade occurred
        if trade_info:
            info.update(trade_info)
            
        # Add termination reason if applicable
        if termination_reason:
            info['termination_reason'] = termination_reason
            
        # Add diagnostic information
        info['diagnostics'] = {
            'stop_loss_count': self.stop_loss_count,
            'take_profit_count': self.take_profit_count,
            'max_holding_count': self.max_holding_count,
            'exit_signal_count': self.exit_signal_count,
            'early_termination_count': self.early_termination_count
        }
        
        # Add warning if the episode is too short
        if (terminated or truncated) and self.episode_step_count < 10:
            info['warning'] = f"Episode ended after only {self.episode_step_count} steps"
            if self.debug_mode:
                print(f"WARNING: Episode ended after only {self.episode_step_count} steps")
                
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: The rendering mode
            
        Returns:
            Rendered image depending on the mode
        """
        if mode == 'human':
            # Simple console output with current state
            print(f"Step: {self.current_step}, Date: {self.dates[self.current_step if self.current_step < len(self.dates) else -1]}")
            print(f"Price: ${self.current_price:.2f}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Cash Balance: ${self.cash_balance:.2f}")
            print(f"Position: {self.current_position:.2f} shares")
            print(f"Position P&L: {self.position_pnl:.2f}%")
            print(f"Total P&L: {((self.portfolio_value / self.initial_capital) - 1) * 100:.2f}%")
            print(f"Drawdown: {self.drawdown * 100:.2f}%")
            print(f"Curriculum Level: {self.curriculum_level}")
            print("-" * 50)
            return None
            
        elif mode == 'rgb_array':
            # Generate a visualization of the trading activity
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            
            # Define the visible window (last 100 steps or all available steps)
            visible_start = max(0, self.current_step - 100)
            visible_end = min(self.current_step + 1, len(self.dates))
            
            # Extract relevant data
            dates = self.dates[visible_start:visible_end]
            prices = self.price_data.iloc[visible_start:visible_end]['close'].values
            portfolio_values = np.array(self.returns_history)[-len(dates):] if len(self.returns_history) > 0 else []
            
            # Plot price
            ax1.plot(dates, prices, color='blue')
            ax1.set_title('Stock Price')
            ax1.set_ylabel('Price ($)')
            
            # Plot portfolio value
            if len(portfolio_values) > 0:
                ax2.plot(dates[-len(portfolio_values):], portfolio_values, color='green')
            ax2.set_title('Portfolio Value')
            ax2.set_ylabel('Value ($)')
            
            # Plot position sizes
            if len(self.positions_history) > 0:
                position_sizes = np.array(self.positions_history)[-len(dates):] if len(self.positions_history) > 0 else []
                if len(position_sizes) > 0:
                    ax3.bar(dates[-len(position_sizes):], position_sizes, color='orange')
            ax3.set_title('Position Size')
            ax3.set_ylabel('Shares')
            ax3.set_xlabel('Date')
            
            # Format the figure
            plt.tight_layout()
            
            try:
                # Convert the figure to an image
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                plt.close(fig)
                return image
            except Exception as e:
                print(f"Error rendering: {e}")
                plt.close(fig)
                return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def close(self):
        """
        Clean up resources.
        """
        plt.close('all')
    
    def seed(self, seed=None):
        """
        Set the random seed.
        
        Args:
            seed: The random seed
            
        Returns:
            List with the seed
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)  # Also seed Python's random
        return [seed]
    
    def _get_observation(self) -> np.ndarray:
        """
        Generate the observation vector for the current state.
        
        Returns:
            Observation vector
        """
        try:
            # Get market data for observation window
            if self.current_step < self.window_size:
                raise ValueError(f"Current step {self.current_step} is less than window size {self.window_size}")
                
            market_data = self.price_data.iloc[self.current_step - self.window_size:self.current_step]
            
            # Generate observation vector
            observation = self.observation_generator.generate_observation(
                market_data=market_data,
                current_step=self.current_step,
                current_position=self.current_position,
                position_entry_price=self.position_entry_price,
                position_entry_step=self.position_entry_step,
                stop_loss_pct=self.stop_loss_pct,
                take_profit_pct=self.take_profit_pct,
                portfolio_value=self.portfolio_value,
                initial_capital=self.initial_capital,
                cash_balance=self.cash_balance,
                max_portfolio_value=self.max_portfolio_value,
                drawdown=self.drawdown
            )
            
            return observation
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error generating observation: {e}")
            # Return a zero vector of the correct shape as a fallback
            if hasattr(self, 'observation_space'):
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                # If observation_space not defined yet, return a placeholder
                return np.zeros(100, dtype=np.float32)
    
    def _execute_trading_logic(
        self, 
        position_size_pct: float, 
        stop_loss_pct: float, 
        take_profit_pct: float, 
        exit_signal: float,
        current_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Execute trading logic based on the provided parameters.
        
        Args:
            position_size_pct: Desired position size as percentage of available capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            exit_signal: Signal to exit position (>0.5 means exit)
            current_data: Current market data
            
        Returns:
            trade_info: Information about any trades executed
        """
        trade_info = None
        
        # Check if we should exit current position
        if self.current_position > 0 and exit_signal > 0.5:
            trade_info = self._close_position(reason="exit_signal")
            self.exit_signal_count += 1
            
        # If we don't have a position, consider entering one
        elif self.current_position == 0 and position_size_pct > 0:
            trade_info = self._open_position(position_size_pct, stop_loss_pct, take_profit_pct, current_data)
            
        return trade_info
    
    def _open_position(
        self, 
        position_size_pct: float, 
        stop_loss_pct: float, 
        take_profit_pct: float, 
        current_data: pd.Series
    ) -> Dict[str, Any]:
        """
        Open a new trading position.
        
        Args:
            position_size_pct: Position size as percentage of available capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            current_data: Current market data
            
        Returns:
            trade_info: Information about the trade
        """
        # Ensure positive position size and valid price
        if position_size_pct <= 0 or current_data['close'] <= 0:
            return None
            
        # Calculate position size in shares
        capital_to_use = self.cash_balance * position_size_pct
        
        # Ensure we have enough capital to open a position
        if capital_to_use <= 0:
            if self.debug_mode:
                print(f"Not enough capital to open position: {capital_to_use}")
            return None
            
        shares_to_buy = capital_to_use / current_data['close']
        
        # Ensure shares to buy is positive
        if shares_to_buy <= 0:
            if self.debug_mode:
                print(f"Invalid shares to buy: {shares_to_buy}")
            return None
            
        # Apply transaction costs
        transaction_cost = capital_to_use * self.transaction_cost_pct
        
        # Execute the trade through market simulator
        try:
            executed_shares, executed_price, actual_cost = self.market_simulator.execute_buy_order(
                shares=shares_to_buy,
                current_price=current_data['close'],
                capital=self.cash_balance,
                transaction_cost_pct=self.transaction_cost_pct
            )
        except Exception as e:
            if self.debug_mode:
                print(f"Error executing buy order: {e}")
            return None
        
        # Ensure we actually executed shares
        if executed_shares <= 0:
            if self.debug_mode:
                print(f"No shares executed in buy order")
            return None
            
        # Update state
        self.cash_balance -= actual_cost
        self.current_position = executed_shares
        self.position_entry_price = executed_price
        self.position_entry_step = self.current_step
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Record the position
        self.positions_history.append(executed_shares)
        
        # Prepare trade information
        trade_info = {
            'trade_executed': True,
            'trade_type': 'buy',
            'trade_shares': executed_shares,
            'trade_price': executed_price,
            'trade_cost': actual_cost,
            'trade_date': self.dates[self.current_step] if self.current_step < len(self.dates) else self.dates[-1],
            'trade_completed': False,
            'stop_loss': self.position_entry_price * (1 - stop_loss_pct),
            'take_profit': self.position_entry_price * (1 + take_profit_pct)
        }
        
        # Add to trade history
        self.trade_history.append(trade_info)
        
        if self.debug_mode:
            print(f"Opened position at step {self.current_step}, date {self.dates[self.current_step if self.current_step < len(self.dates) else -1]}")
            print(f"Shares: {executed_shares:.2f}, Entry price: ${executed_price:.2f}")
            print(f"Stop loss: ${self.position_entry_price * (1 - stop_loss_pct):.2f} ({stop_loss_pct:.2%})")
            print(f"Take profit: ${self.position_entry_price * (1 + take_profit_pct):.2f} ({take_profit_pct:.2%})")
        
        return trade_info
    
    def _close_position(self, reason: str) -> Dict[str, Any]:
        """
        Close the current trading position.
        
        Args:
            reason: Reason for closing the position
            
        Returns:
            trade_info: Information about the trade
        """
        if self.current_position == 0:
            return None
            
        # Execute the trade through market simulator
        try:
            executed_shares, executed_price, sale_value = self.market_simulator.execute_sell_order(
                shares=self.current_position,
                current_price=self.current_price,
                transaction_cost_pct=self.transaction_cost_pct
            )
        except Exception as e:
            if self.debug_mode:
                print(f"Error executing sell order: {e}")
            # Force position closure even if market simulator fails
            executed_shares = self.current_position
            executed_price = self.current_price
            sale_value = executed_shares * executed_price * (1 - self.transaction_cost_pct)
        
        # Calculate P&L
        cost_basis = self.position_entry_price * executed_shares
        profit_loss = sale_value - cost_basis
        self.position_pnl = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
        self.total_pnl += profit_loss
        
        # Update state
        self.cash_balance += sale_value
        previous_position = self.current_position
        entry_price = self.position_entry_price
        entry_date = self.dates[self.position_entry_step] if self.position_entry_step < len(self.dates) else self.dates[-1]
        
        self.current_position = 0
        self.position_entry_price = 0
        self.stop_loss_pct = 0
        self.take_profit_pct = 0
        
        # Record the position closure
        self.positions_history.append(0)
        
        # Prepare trade information
        trade_info = {
            'trade_executed': True,
            'trade_type': 'sell',
            'trade_shares': executed_shares,
            'trade_price': executed_price,
            'trade_value': sale_value,
            'trade_date': self.dates[self.current_step] if self.current_step < len(self.dates) else self.dates[-1],
            'trade_profit': profit_loss,
            'trade_profit_pct': self.position_pnl,
            'trade_completed': True,
            'trade_reason': reason,
            'trade_duration': self.current_step - self.position_entry_step,
            'entry_date': entry_date,
            'entry_price': entry_price
        }
        
        # Add to trade history
        self.trade_history.append(trade_info)
        
        # Update risk manager with trade result
        self.risk_manager.update_trade_result(self.position_pnl / 100)  # Convert percentage to decimal
        
        if self.debug_mode:
            print(f"Closed position at step {self.current_step}, date {self.dates[self.current_step if self.current_step < len(self.dates) else -1]}")
            print(f"Reason: {reason}, Profit/Loss: ${profit_loss:.2f} ({self.position_pnl:.2%})")
            print(f"Duration: {self.current_step - self.position_entry_step} days")
        
        return trade_info
    
    def _update_portfolio_value(self) -> None:
        """
        Update the portfolio value and related metrics.
        """
        # Calculate total portfolio value
        holdings_value = self.current_position * self.current_price
        self.portfolio_value = self.cash_balance + holdings_value
        
        # Update maximum portfolio value
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
            
        # Update drawdown
        if self.max_portfolio_value > 0:
            self.drawdown = 1 - (self.portfolio_value / self.max_portfolio_value)
            
        # Record portfolio value
        self.returns_history.append(self.portfolio_value)