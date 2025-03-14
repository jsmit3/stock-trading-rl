"""
env/core.py

Core environment class for stock trading reinforcement learning.

This module implements the main Gymnasium environment interface,
delegating specific functionality to specialized components.

Author: [Your Name]
Date: March 10, 2025
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import random

from env.state import EnvironmentState
from env.renderer import EnvironmentRenderer
from observation.generator import ObservationGenerator
from action.interpreter import ActionInterpreter
from reward.calculator import RewardCalculator
from market.simulator import MarketSimulator
from trading.position_manager import PositionManager
from trading.risk_manager import RiskManager
from data.processor import DataProcessor


class StockTradingEnv(gym.Env):
    """
    A Gymnasium environment for stock trading using reinforcement learning.
    
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
        min_episode_length: int = 20,
        observation_generator: Optional[ObservationGenerator] = None,  # Custom observation generator
        observation_dim: Optional[int] = None,  # Fixed observation dimension
        symbol_feature_dim: int = 0  # Symbol feature dimension
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
            observation_generator: Custom observation generator (optional)
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
        self.observation_dim = observation_dim
        self.symbol_feature_dim = symbol_feature_dim
        
        # Process and validate input data
        self.data_processor = DataProcessor(price_data, window_size=window_size)
        self.price_data = self.data_processor.process_data()
        self.dates = self.price_data.index.tolist()
        
        # Apply curriculum learning parameters
        self._apply_curriculum_settings()
        
        # Initialize environment state
        self.state = EnvironmentState(
            initial_capital=initial_capital,
            window_size=window_size
        )
        
        # Initialize component modules
        # Use provided observation generator or create default
        if observation_generator is not None:
            self.observation_generator = observation_generator
        else:
            # Create observation generator with fixed dimension if specified
            fixed_dim = observation_dim
            if fixed_dim is not None and symbol_feature_dim > 0:
                # Adjust fixed_dim to account for symbol features
                fixed_dim = fixed_dim - symbol_feature_dim
            
            self.observation_generator = ObservationGenerator(
                window_size=window_size, 
                include_sentiment=include_sentiment,
                fixed_dim=fixed_dim
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
            profitable_trade_bonus=0.2
        )
        
        self.risk_manager = RiskManager(
            max_position_pct=self.max_position_pct,
            min_position_pct=self.min_position_pct,
            max_risk_per_trade=self.max_risk_per_trade
        )
        
        self.market_simulator = MarketSimulator(
            transaction_cost_pct=self.transaction_cost_pct
        )
        
        self.position_manager = PositionManager(
            market_simulator=self.market_simulator,
            risk_manager=self.risk_manager,
            debug_mode=self.debug_mode
        )
        
        self.renderer = EnvironmentRenderer()
        
        # Define action space
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),  # Position size, stop-loss, take-profit, exit signal
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Generate a dummy observation to determine observation space shape
        dummy_observation = self._get_observation()
        
        # Define the observation space with the correct shape
        if self.observation_dim is not None:
            # Use fixed observation dimension if specified
            obs_shape = (self.observation_dim,)
            
            # If dummy observation doesn't match, pad or trim it
            if dummy_observation.shape[0] != self.observation_dim:
                print(f"Warning: Observation shape mismatch - expected {self.observation_dim}, got {dummy_observation.shape[0]}")
                print(f"Using fixed observation dimension: {self.observation_dim}")
        else:
            # Use the actual shape from the dummy observation
            obs_shape = dummy_observation.shape
            
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # Initialize step tracking
        self.current_step = window_size  # Initialize to window_size to ensure it exists before first observation
        self.episode_step_count = 0
        self.episode_count = 0
        self._reset_called = False
        
        # Initialize diagnostic counters
        self.stop_loss_count = 0
        self.take_profit_count = 0
        self.max_holding_count = 0
        self.exit_signal_count = 0
        self.early_termination_count = 0
    
    def set_observation_generator(self, observation_generator):
        """
        Replace the observation generator and update observation space.
        
        Args:
            observation_generator: New observation generator to use
        """
        # Store the old generator temporarily
        old_generator = self.observation_generator
        
        # Set the new generator
        self.observation_generator = observation_generator
        
        # Generate a dummy observation with the new generator
        try:
            dummy_observation = self._get_observation()
            
            # Update the observation space
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=dummy_observation.shape,
                dtype=np.float32
            )
            
            if self.debug_mode:
                print(f"Updated observation space to shape: {dummy_observation.shape}")
        except Exception as e:
            # Restore the old generator if there's a problem
            self.observation_generator = old_generator
            if self.debug_mode:
                print(f"Error updating observation generator: {e}")
                print("Reverted to previous observation generator")
            raise e
    
    def _apply_curriculum_settings(self):
        """Apply settings based on curriculum level"""
        # Set risk parameters based on curriculum level
        if self.curriculum_level == 1:
            # Beginner level - Lower risk, clearer patterns
            self.max_risk_per_trade = 0.01
            self.transaction_cost_pct = 0.0005
            self.max_drawdown_pct = 0.35
            
        elif self.curriculum_level == 2:
            # Intermediate level
            self.max_risk_per_trade = 0.015
            self.transaction_cost_pct = 0.001
            self.max_drawdown_pct = 0.40
            
        else:  # Level 3 or higher
            # Advanced level - Full complexity
            self.max_risk_per_trade = 0.02
            self.transaction_cost_pct = 0.0015
            self.max_drawdown_pct = 0.45
            
        # Update risk manager if it exists
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
            max_start = int(len(self.price_data) * 0.75)
            start_idx = random.randint(self.window_size, max_start)
        
        # Reset state
        self.state.reset(
            start_idx=start_idx,
            initial_capital=self.initial_capital
        )
        
        # Reset current step
        self.current_step = start_idx
        
        # Get current market data
        self.current_price = self.price_data.iloc[self.current_step]['close']
        
        # Reset episode counters
        self.episode_step_count = 0
        self._reset_called = True
        self.episode_count += 1
        
        # Reset diagnostic counters
        self.stop_loss_count = 0
        self.take_profit_count = 0
        self.max_holding_count = 0
        self.exit_signal_count = 0
        
        # Reset reward calculator
        self.reward_calculator.reset()
        
        # Generate initial observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'step': self.current_step,
            'date': self.dates[self.current_step],
            'price': self.current_price,
            'portfolio_value': self.state.portfolio_value,
            'cash_balance': self.state.cash_balance,
            'current_position': self.state.current_position,
            'position_pnl': self.state.position_pnl,
            'total_pnl': self.state.total_pnl,
            'drawdown': self.state.drawdown,
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
        self.state.actions_history.append(action.copy())
        
        # Get current market state
        current_data = self.price_data.iloc[self.current_step]
        prev_data = self.price_data.iloc[self.current_step - 1]
        
        # Store previous state for reward calculation
        prev_portfolio_value = self.state.portfolio_value
        prev_position = self.state.current_position
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
        
        # Apply risk management adjustments
        try:
            position_size_pct = self.risk_manager.adjust_position_size(
                position_size_pct, 
                current_data, 
                self.current_step,
                self.price_data,
                day_of_week=self.dates[self.current_step].dayofweek
            )
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
        
        # Update current price and check for stop-loss/take-profit if not truncated
        if not truncated:
            self.current_price = self.price_data.iloc[self.current_step]['close']
            
            # Check for stop-loss or take-profit hits
            if self.state.current_position > 0:
                # Update position days held
                days_held = self.current_step - self.state.position_entry_step
                
                # Calculate current position P&L
                self.state.position_pnl = ((self.current_price / self.state.position_entry_price) - 1.0) * 100
                
                # Check for stop-loss hit
                low_price = self.price_data.iloc[self.current_step]['low']
                if low_price <= self.state.position_entry_price * (1 - self.state.stop_loss_pct) and self.state.stop_loss_pct > 0:
                    trade_info = self.position_manager.close_position(
                        self.state, self.current_price, "stop_loss", self.current_step, self.dates
                    )
                    self.stop_loss_count += 1
                    trade_completed = True
                    trade_profit = trade_info.get('trade_profit', 0.0) if trade_info else 0.0
                
                # Check for take-profit hit
                high_price = self.price_data.iloc[self.current_step]['high']
                if high_price >= self.state.position_entry_price * (1 + self.state.take_profit_pct) and self.state.take_profit_pct > 0:
                    trade_info = self.position_manager.close_position(
                        self.state, self.current_price, "take_profit", self.current_step, self.dates
                    )
                    self.take_profit_count += 1
                    trade_completed = True
                    trade_profit = trade_info.get('trade_profit', 0.0) if trade_info else 0.0
                    
                # Check for maximum holding period
                if days_held >= self.max_holding_period:
                    trade_info = self.position_manager.close_position(
                        self.state, self.current_price, "max_holding_period", self.current_step, self.dates
                    )
                    self.max_holding_count += 1
                    trade_completed = True
                    trade_profit = trade_info.get('trade_profit', 0.0) if trade_info else 0.0
        
        # Update portfolio value
        self.state.update_portfolio_value(self.current_price)
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            prev_portfolio_value=prev_portfolio_value,
            current_portfolio_value=self.state.portfolio_value,
            prev_position=prev_position,
            current_position=self.state.current_position,
            drawdown=self.state.drawdown,
            daily_volatility=current_data.get('volatility', 0.01),
            capital_utilization=self.state.current_position * self.current_price / self.state.portfolio_value 
                if self.state.portfolio_value > 0 else 0,
            trade_completed=trade_completed,
            trade_profit=trade_profit
        )
        
        # Record the reward
        self.state.rewards_history.append(reward)
        
        # Get the next observation
        observation = self._get_observation()
        
        # Check for bankruptcy or max drawdown termination conditions
        terminated = False
        termination_reason = None
        
        # Only allow early termination if we've gone past the minimum episode length
        if self.episode_step_count >= self.min_episode_length:
            if self.state.portfolio_value <= 0:
                terminated = True
                termination_reason = "bankruptcy"
                self.early_termination_count += 1
                if self.debug_mode:
                    print(f"Episode terminated due to bankruptcy at step {self.current_step}")
                    
            elif self.state.drawdown >= self.max_drawdown_pct:
                terminated = True
                termination_reason = f"max_drawdown_exceeded (drawdown: {self.state.drawdown:.2%}, threshold: {self.max_drawdown_pct:.2%})"
                self.early_termination_count += 1
                if self.debug_mode:
                    print(f"Episode terminated due to max drawdown exceeded at step {self.current_step}")
                    print(f"Drawdown: {self.state.drawdown:.2%}, Threshold: {self.max_drawdown_pct:.2%}")
        
        # Prepare info dictionary
        info = {
            'step': self.current_step,
            'date': self.dates[self.current_step] if not truncated and self.current_step < len(self.dates) else self.dates[-1],
            'price': self.current_price,
            'portfolio_value': self.state.portfolio_value,
            'cash_balance': self.state.cash_balance,
            'current_position': self.state.current_position,
            'position_pnl': self.state.position_pnl,
            'total_pnl': self.state.total_pnl,
            'drawdown': self.state.drawdown,
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
        return self.renderer.render(
            self.state, 
            self.current_step, 
            self.dates, 
            self.price_data,
            mode
        )
    
    def close(self):
        """Clean up resources."""
        self.renderer.close()
    
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
                current_position=self.state.current_position,
                position_entry_price=self.state.position_entry_price,
                position_entry_step=self.state.position_entry_step,
                stop_loss_pct=self.state.stop_loss_pct,
                take_profit_pct=self.state.take_profit_pct,
                portfolio_value=self.state.portfolio_value,
                initial_capital=self.initial_capital,
                cash_balance=self.state.cash_balance,
                max_portfolio_value=self.state.max_portfolio_value,
                drawdown=self.state.drawdown
            )
            
            # If symbol features are enabled, add empty placeholder for compatibility with multi-stock env
            if self.symbol_feature_dim > 0 and self.observation_dim is not None:
                # Create a combined observation with base observation and empty symbol features
                base_obs_dim = self.observation_dim - self.symbol_feature_dim
                
                # Create output observation of the correct full size
                combined_observation = np.zeros(self.observation_dim, dtype=np.float32)
                
                # Copy base observation (up to the appropriate size)
                base_obs_size = min(len(observation), base_obs_dim)
                combined_observation[:base_obs_size] = observation[:base_obs_size]
                
                # Symbol features are left as zeros (will be populated in multi-stock env)
                return combined_observation
            else:
                # Return the regular observation
                return observation
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error generating observation: {e}")
            
            # Return a zero vector of the CORRECT shape as a fallback
            if hasattr(self, 'observation_space'):
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                # If observation_space not defined yet, use the specified dimension or default
                if self.observation_dim is not None:
                    return np.zeros(self.observation_dim, dtype=np.float32)
                else:
                    # Use a reasonable default that matches your typical observation size
                    return np.zeros(325, dtype=np.float32)
    
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
        if self.state.current_position > 0 and exit_signal > 0.5:
            trade_info = self.position_manager.close_position(
                self.state, self.current_price, "exit_signal", self.current_step, self.dates
            )
            self.exit_signal_count += 1
            
        # If we don't have a position, consider entering one
        elif self.state.current_position == 0 and position_size_pct > 0:
            trade_info = self.position_manager.open_position(
                self.state,
                position_size_pct,
                stop_loss_pct,
                take_profit_pct,
                current_data['close'],
                self.current_step,
                self.dates
            )
            
        return trade_info