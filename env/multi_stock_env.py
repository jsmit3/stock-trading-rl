"""
env/multi_stock_env.py

Multi-stock trading environment for reinforcement learning.

This module extends the standard StockTradingEnv to handle multiple
stocks simultaneously, allowing the agent to learn general trading
strategies that work across different securities.

Author: [Your Name]
Date: March 13, 2025
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
import random

from env.core import StockTradingEnv
from env.state import EnvironmentState
from env.renderer import EnvironmentRenderer
from observation.generator import ObservationGenerator
from action.interpreter import ActionInterpreter
from reward.calculator import RewardCalculator
from market.simulator import MarketSimulator
from trading.position_manager import PositionManager
from trading.risk_manager import RiskManager
from data.processor import DataProcessor


class MultiStockTradingEnv(gym.Env):
    """
    A Gymnasium environment for multi-stock trading.
    
    This environment allows training a single agent across multiple stocks,
    enabling it to learn generalizable trading strategies rather than
    overfitting to the patterns of a single security.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        stock_data: Dict[str, pd.DataFrame],
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
        observation_generator: Optional[ObservationGenerator] = None,
        observation_dim: int = 325,
        symbol_feature_dim: int = 20
    ):
        """
        Initialize the multi-stock trading environment.
        
        Args:
            stock_data: Dictionary mapping symbols to DataFrames with OHLCV data
            initial_capital: Starting capital amount
            max_holding_period: Maximum number of days to hold a position
            transaction_cost_pct: Transaction costs (%)
            window_size: Number of past days to include in observation
            reward_scaling: Scaling factor for reward normalization
            risk_aversion: Coefficient for volatility penalty (λ₁)
            drawdown_penalty: Coefficient for drawdown penalty (λ₂)
            opportunity_cost: Coefficient for unused capital penalty (λ₃)
            drawdown_threshold: Acceptable drawdown threshold
            max_drawdown_pct: Maximum drawdown before early termination
            include_sentiment: Whether to include sentiment features
            max_position_pct: Maximum position size as percentage of capital
            min_position_pct: Minimum position size as percentage of capital
            max_episodes: Maximum number of episodes (for curriculum learning)
            curriculum_level: Current curriculum learning level (1-3)
            debug_mode: Whether to print debug information
            min_episode_length: Minimum number of steps before allowing early termination
            observation_generator: Custom observation generator (optional)
            observation_dim: Fixed dimension for observations
            symbol_feature_dim: Dimension for symbol embedding features
        """
        super(MultiStockTradingEnv, self).__init__()
        
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
        
        # Store all stock data
        self.stock_data = stock_data
        self.symbols = list(stock_data.keys())
        
        if len(self.symbols) == 0:
            raise ValueError("No stock data provided")
            
        # Process and validate input data for all stocks
        self.processed_data = {}
        self.data_processor = {}
        self.dates = {}
        
        for symbol, data in self.stock_data.items():
            self.data_processor[symbol] = DataProcessor(data, window_size=window_size)
            self.processed_data[symbol] = self.data_processor[symbol].process_data()
            self.dates[symbol] = self.processed_data[symbol].index.tolist()
        
        # Current stock being traded
        self.current_symbol = None
        
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
            self.observation_generator = ObservationGenerator(
                window_size=window_size, 
                include_sentiment=include_sentiment,
                fixed_dim=observation_dim - symbol_feature_dim  # Reserve space for symbol embedding
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
        
        # Define action space - same as the base env
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),  # Position size, stop-loss, take-profit, exit signal
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Generate a dummy observation to determine observation space shape
        self._select_random_symbol()
        dummy_observation = self._get_observation()
        
        # Define the observation space with the fixed dimension
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
        
        # Initialize step tracking
        self.current_step = self.window_size  # Initialize to window_size to ensure it exists before first observation
        self.episode_step_count = 0
        self.episode_count = 0
        self._reset_called = False
        
        # Initialize diagnostic counters
        self.stop_loss_count = 0
        self.take_profit_count = 0
        self.max_holding_count = 0
        self.exit_signal_count = 0
        self.early_termination_count = 0
        
        # Create symbol encodings
        self._create_symbol_encodings()
    
    def _create_symbol_encodings(self):
        """Create unique vector representations for each symbol."""
        num_symbols = len(self.symbols)
        
        # Symbol to index mapping
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        
        # Create unique embedding for each symbol
        # Option 1: One-hot encoding
        if num_symbols <= self.symbol_feature_dim:
            # One-hot encoding
            self.symbol_embeddings = {}
            for symbol, idx in self.symbol_to_idx.items():
                embedding = np.zeros(self.symbol_feature_dim, dtype=np.float32)
                embedding[idx] = 1.0
                self.symbol_embeddings[symbol] = embedding
        else:
            # Option 2: Random but consistent embeddings
            np.random.seed(42)  # For consistent embeddings
            self.symbol_embeddings = {}
            for symbol in self.symbols:
                embedding = np.random.normal(0, 1, self.symbol_feature_dim)
                # Normalize to unit length
                embedding = embedding / np.linalg.norm(embedding)
                self.symbol_embeddings[symbol] = embedding.astype(np.float32)
    
    def _select_random_symbol(self):
        """Select a random symbol for the episode."""
        self.current_symbol = random.choice(self.symbols)
        self.price_data = self.processed_data[self.current_symbol]
        return self.current_symbol
    
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
        
        # Select stock for this episode
        if options and 'symbol' in options:
            self.current_symbol = options['symbol']
            if self.current_symbol not in self.symbols:
                raise ValueError(f"Symbol {self.current_symbol} not found in available stocks")
        else:
            self._select_random_symbol()
        
        self.price_data = self.processed_data[self.current_symbol]
        
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
            'date': self.dates[self.current_symbol][self.current_step],
            'price': self.current_price,
            'portfolio_value': self.state.portfolio_value,
            'cash_balance': self.state.cash_balance,
            'current_position': self.state.current_position,
            'position_pnl': self.state.position_pnl,
            'total_pnl': self.state.total_pnl,
            'drawdown': self.state.drawdown,
            'initial_capital': self.initial_capital,
            'curriculum_level': self.curriculum_level,
            'symbol': self.current_symbol
        }
        
        if self.debug_mode:
            print(f"Episode {self.episode_count} started with symbol {self.current_symbol} at step {self.current_step}")
            print(f"Date: {self.dates[self.current_symbol][self.current_step]}")
        
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
            date_obj = self.dates[self.current_symbol][self.current_step]
            day_of_week = date_obj.dayofweek if hasattr(date_obj, 'dayofweek') else 0
            
            position_size_pct = self.risk_manager.adjust_position_size(
                position_size_pct, 
                current_data, 
                self.current_step,
                self.price_data,
                day_of_week=day_of_week
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
                        self.state, self.current_price, "stop_loss", self.current_step, self.dates[self.current_symbol]
                    )
                    self.stop_loss_count += 1
                    trade_completed = True
                    trade_profit = trade_info.get('trade_profit', 0.0) if trade_info else 0.0
                
                # Check for take-profit hit
                high_price = self.price_data.iloc[self.current_step]['high']
                if high_price >= self.state.position_entry_price * (1 + self.state.take_profit_pct) and self.state.take_profit_pct > 0:
                    trade_info = self.position_manager.close_position(
                        self.state, self.current_price, "take_profit", self.current_step, self.dates[self.current_symbol]
                    )
                    self.take_profit_count += 1
                    trade_completed = True
                    trade_profit = trade_info.get('trade_profit', 0.0) if trade_info else 0.0
                    
                # Check for maximum holding period
                if days_held >= self.max_holding_period:
                    trade_info = self.position_manager.close_position(
                        self.state, self.current_price, "max_holding_period", self.current_step, self.dates[self.current_symbol]
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
            'date': self.dates[self.current_symbol][self.current_step] if not truncated and self.current_step < len(self.dates[self.current_symbol]) else self.dates[self.current_symbol][-1],
            'price': self.current_price,
            'portfolio_value': self.state.portfolio_value,
            'cash_balance': self.state.cash_balance,
            'current_position': self.state.current_position,
            'position_pnl': self.state.position_pnl,
            'total_pnl': self.state.total_pnl,
            'drawdown': self.state.drawdown,
            'initial_capital': self.initial_capital,
            'curriculum_level': self.curriculum_level,
            'symbol': self.current_symbol,
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
            self.dates[self.current_symbol], 
            self.price_data,
            mode
        )
    
    def close(self):
        """Clean up resources."""
        self.renderer.close()
    
    def _get_observation(self) -> np.ndarray:
        """
        Generate the observation vector for the current state, including symbol encoding.
        
        Returns:
            Observation vector
        """
        try:
            # Get market data for observation window
            if self.current_step < self.window_size:
                raise ValueError(f"Current step {self.current_step} is less than window size {self.window_size}")
                
            market_data = self.price_data.iloc[self.current_step - self.window_size:self.current_step]
            
            # Generate base observation vector from the market data
            base_observation = self.observation_generator.generate_observation(
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
            
            # Get symbol encoding for current symbol
            symbol_encoding = self.symbol_embeddings[self.current_symbol]
            
            # Combine base observation and symbol encoding
            combined_observation = np.zeros(self.observation_dim, dtype=np.float32)
            
            # Copy base observation
            base_obs_dim = min(len(base_observation), self.observation_dim - self.symbol_feature_dim)
            combined_observation[:base_obs_dim] = base_observation[:base_obs_dim]
            
            # Copy symbol encoding to the end
            symbol_offset = self.observation_dim - self.symbol_feature_dim
            combined_observation[symbol_offset:] = symbol_encoding
            
            return combined_observation
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error generating observation: {e}")
            
            # Return a zero vector of the CORRECT shape as a fallback
            return np.zeros(self.observation_dim, dtype=np.float32)
    
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
                self.state, self.current_price, "exit_signal", self.current_step, self.dates[self.current_symbol]
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
                self.dates[self.current_symbol]
            )
            
        return trade_info