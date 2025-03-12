"""
main.py

This script demonstrates how to use the stock trading reinforcement learning environment
with improved settings to address early termination and GPU utilization issues.

Example usage:
```
python main.py --ticker AAPL --start_date 2020-01-01 --end_date 2023-01-01 --train --curriculum --use_gpu
```

Author: [Your Name]
Date: March 2, 2025
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any, List
import talib  # Make sure to install this
import torch  # Added import for torch to enable GPU
import random

# Import the trading environment
from trading_env import StockTradingEnv
from archive.data_processor import DataProcessor

# Create a custom callback for Curriculum Learning that properly inherits from BaseCallback
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumLearningCallback(BaseCallback):
    def __init__(self, env, target_reward=0.5, window_size=20, verbose=0):
        super(CurriculumLearningCallback, self).__init__(verbose)
        self.env = env
        self.target_reward = target_reward
        self.window_size = window_size
        self.rewards = []
    
    def _init_callback(self) -> None:
        # Initialize callback
        self.rewards = []
    
    def _on_step(self) -> bool:
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            # Get episode rewards from the info buffer
            if 'episode' in self.locals.get('infos', [{}])[0]:
                episode_reward = self.locals['infos'][0]['episode']['r']
                self.rewards.append(episode_reward)
                
                # Print episode length for debugging
                episode_length = self.locals['infos'][0]['episode']['l']
                if self.verbose > 0:
                    print(f"Episode length: {episode_length}, reward: {episode_reward:.3f}")
                
                # If we have enough rewards, check if we should advance
                if len(self.rewards) >= self.window_size:
                    avg_reward = sum(self.rewards[-self.window_size:]) / self.window_size
                    
                    # If average reward exceeds target, advance curriculum
                    if avg_reward >= self.target_reward:
                        # Get env from wrapper (DummyVecEnv)
                        vec_env = self.training_env
                        # Access the underlying env
                        env = vec_env.envs[0].env.env  # Unwrap Monitor -> StockTradingEnv
                        
                        advanced = env.advance_curriculum()
                        if advanced:
                            level = env.curriculum_level
                            print(f"\n=== Advanced to Curriculum Level {level} ===\n")
                            # Reset the rewards for the new level
                            self.rewards = []
        
        return True


# Custom logging callback to track training progress and diagnostics
class MetricsLoggerCallback(BaseCallback):
    def __init__(self, eval_env, log_path, log_freq=5000, verbose=1):
        super(MetricsLoggerCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.log_path = log_path
        self.log_freq = log_freq
        self.metrics = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
    def _init_callback(self) -> None:
        # Create the log file if it doesn't exist
        with open(self.log_path, "w") as f:
            f.write("timestep,avg_reward,avg_length,win_rate,loss_rate,sl_count,tp_count,max_hold_count,exit_count\n")
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Evaluate the agent
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=5,
                deterministic=True,
                return_episode_rewards=False
            )
            
            # Get diagnostics from the environment
            env = self.eval_env.envs[0].env.env  # Unwrap Monitor -> StockTradingEnv
            diagnostics = {
                'sl_count': env.stop_loss_count,
                'tp_count': env.take_profit_count,
                'max_hold_count': env.max_holding_count,
                'exit_count': env.exit_signal_count
            }
            
            # Calculate win rate
            trades = diagnostics['sl_count'] + diagnostics['tp_count'] + diagnostics['max_hold_count'] + diagnostics['exit_count']
            win_rate = (diagnostics['tp_count'] + diagnostics['exit_count']) / max(trades, 1)
            loss_rate = diagnostics['sl_count'] / max(trades, 1)
            
            # Reset diagnostics after evaluation
            env.stop_loss_count = 0
            env.take_profit_count = 0
            env.max_holding_count = 0
            env.exit_signal_count = 0
            
            # Get average episode length
            ep_len = 0
            if hasattr(self.model, 'ep_len_mean'):
                ep_len = self.model.ep_len_mean
            
            # Log to csv
            with open(self.log_path, "a") as f:
                f.write(f"{self.n_calls},{mean_reward},{ep_len},{win_rate},{loss_rate}," +
                        f"{diagnostics['sl_count']},{diagnostics['tp_count']},{diagnostics['max_hold_count']},{diagnostics['exit_count']}\n")
            
            if self.verbose > 0:
                print(f"Timestep {self.n_calls}: Mean Reward = {mean_reward:.2f}, Win Rate = {win_rate:.2f}, Loss Rate = {loss_rate:.2f}")
                print(f"SL: {diagnostics['sl_count']}, TP: {diagnostics['tp_count']}, Max Hold: {diagnostics['max_hold_count']}, Exit: {diagnostics['exit_count']}")
                
        return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Stock Trading RL Environment Demo')
    
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--start_date', type=str, default='2018-01-01',
                        help='Start date for data (default: 2018-01-01)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for data (default: today)')
    parser.add_argument('--initial_capital', type=float, default=100000.0,
                        help='Initial capital (default: 100000.0)')
    parser.add_argument('--train_test_split', type=float, default=0.8,
                        help='Train/test split ratio (default: 0.8)')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Directory to save models (default: ./models)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs (default: ./logs)')
    parser.add_argument('--train', action='store_true',
                        help='Train the agent (default: False)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the agent (default: False)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment (default: False)')
    parser.add_argument('--total_timesteps', type=int, default=250000,
                        help='Total timesteps for training (default: 250000)')
    parser.add_argument('--curriculum', action='store_true',
                        help='Use curriculum learning (default: False)')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU for training if available (default: True)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode (default: False)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None)')
    
    return parser.parse_args()


def load_stock_data(ticker: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load stock data from Yahoo Finance with caching support.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        stock_data: DataFrame with stock data
    """
    # If end_date is not provided, use today's date
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create cache directory if it doesn't exist
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename
    start_year = start_date.split('-')[0]
    end_year = end_date.split('-')[0]
    cache_file = os.path.join(cache_dir, f"{ticker}_{start_year}_to_{end_year}.pkl")
    
    # Check if cache file exists
    if os.path.exists(cache_file):
        print(f"Loading cached data for {ticker} from {start_date} to {end_date}")
        try:
            stock_data = pd.read_pickle(cache_file)
            
            # Verify the data covers the requested date range
            if stock_data.index[0].strftime('%Y-%m-%d') <= start_date and stock_data.index[-1].strftime('%Y-%m-%d') >= end_date:
                print(f"Loaded {len(stock_data)} days of data from cache")
                
                # Slice the data to match the requested date range
                stock_data = stock_data[start_date:end_date]
                
                # Add additional columns for date features
                stock_data['date'] = stock_data.index
                stock_data['day_of_week'] = stock_data.index.dayofweek
                stock_data['month'] = stock_data.index.month
                
                return stock_data
            else:
                print("Cached data doesn't cover requested date range, downloading fresh data...")
        except Exception as e:
            print(f"Error loading cached data: {e}")
    
    # If not in cache or cache invalid, download from Yahoo Finance
    print(f"Downloading data for {ticker} from {start_date} to {end_date}")
    
    try:
        # Try to download data from Yahoo Finance
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if data was successfully downloaded
        if not stock_data.empty:
            # Ensure the data has the required OHLCV columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in stock_data.columns:
                    raise ValueError(f"Column {col} not found in data")
            
            # Rename columns to lowercase for consistency with our environment
            stock_data.columns = [col.lower() for col in stock_data.columns]
            
            # Save to cache
            print(f"Saving data to cache: {cache_file}")
            stock_data.to_pickle(cache_file)
            
            # Add additional columns for date features
            stock_data['date'] = stock_data.index
            stock_data['day_of_week'] = stock_data.index.dayofweek
            stock_data['month'] = stock_data.index.month
            
            print(f"Loaded {len(stock_data)} days of data")
            
            return stock_data
    except Exception as e:
        print(f"Error downloading data from Yahoo Finance: {e}")
        print("Trying alternative approach...")
    
    # If we couldn't get real data, generate synthetic data for testing
    print("Generating synthetic data for testing purposes")
    
    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate date range
    date_range = pd.date_range(start=start, end=end, freq='B')  # Business days
    
    # Generate synthetic price data with reasonable properties
    np.random.seed(42)  # For reproducibility
    
    # Start with a base price
    base_price = 150.0  # For AAPL-like stock
    
    # Generate daily returns with drift
    n_days = len(date_range)
    daily_returns = np.random.normal(0.0005, 0.015, n_days)  # Mean positive return, realistic volatility
    
    # Calculate price series
    price_series = base_price * (1 + daily_returns).cumprod()
    
    # Generate OHLC based on close prices with realistic relationships
    close_prices = price_series
    high_prices = close_prices * (1 + np.random.uniform(0, 0.02, n_days))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.02, n_days))
    open_prices = low_prices + np.random.uniform(0, 1, n_days) * (high_prices - low_prices)
    
    # Generate volume
    volume = np.random.lognormal(16, 0.5, n_days)  # Realistic volume for a major stock
    
    # Create DataFrame
    synthetic_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
        'date': date_range,
        'day_of_week': date_range.dayofweek,
        'month': date_range.month
    })
    
    # Set date as index
    synthetic_data.set_index('date', inplace=True)
    
    print(f"Generated {len(synthetic_data)} days of synthetic data")
    
    # Save synthetic data to cache for consistency in future runs
    synthetic_data.to_pickle(os.path.join(cache_dir, f"{ticker}_{start_year}_to_{end_year}_synthetic.pkl"))
    
    return synthetic_data


def create_training_environment(
    train_data: pd.DataFrame,
    initial_capital: float = 100000.0,
    window_size: int = 20,
    render_mode: Optional[str] = None,
    curriculum_level: int = 1,
    debug_mode: bool = False
) -> gym.Env:
    """
    Create and configure the training environment.
    
    Args:
        train_data: Stock data for training
        initial_capital: Initial capital amount
        window_size: Observation window size
        render_mode: Rendering mode (None, 'human', or 'rgb_array')
        curriculum_level: Starting curriculum level
        debug_mode: Whether to print debug information
        
    Returns:
        env: Configured gym environment
    """
    # Normalize data
    data_processor = DataProcessor(train_data, window_size=window_size)
    normalized_data = data_processor.normalize_data(train_data)
    
    # Create the environment
    env = StockTradingEnv(
        price_data=normalized_data,
        initial_capital=initial_capital,
        max_holding_period=20,
        transaction_cost_pct=0.001,  # Lower for initial training
        window_size=window_size,
        reward_scaling=3.0,         # Increased from 2.0 to provide stronger signals
        risk_aversion=0.2,          # Decreased from 0.3 to reduce penalty further
        drawdown_penalty=0.3,       # Decreased from 0.5 to reduce penalty further
        opportunity_cost=0.1,       # Keep at 0.1 to encourage position taking
        drawdown_threshold=0.15,    # Increased from 0.08 to allow more drawdown
        max_drawdown_pct=0.40,      # Increased from 0.30 for less early termination
        include_sentiment=False,
        max_position_pct=0.5,       # Increased from 0.4 to allow larger positions
        min_position_pct=0.05,
        curriculum_level=curriculum_level,
        debug_mode=debug_mode,      # Enable debug logging
        min_episode_length=20       # Minimum episode length before early termination
    )
    
    # Wrap the environment with Monitor for logging
    env = Monitor(env)
    
    return env


def create_evaluation_environment(
    test_data: pd.DataFrame,
    initial_capital: float = 100000.0,
    window_size: int = 20,
    render_mode: Optional[str] = None,
    debug_mode: bool = False
) -> gym.Env:
    """
    Create and configure the evaluation environment.
    
    Args:
        test_data: Stock data for evaluation
        initial_capital: Initial capital amount
        window_size: Observation window size
        render_mode: Rendering mode (None, 'human', or 'rgb_array')
        debug_mode: Whether to print debug information
        
    Returns:
        env: Configured gym environment
    """
    # Normalize data
    data_processor = DataProcessor(test_data, window_size=window_size)
    normalized_data = data_processor.normalize_data(test_data)
    
    # Create the environment, similar to training but with test data
    env = StockTradingEnv(
        price_data=normalized_data,
        initial_capital=initial_capital,
        max_holding_period=20,
        transaction_cost_pct=0.0015,
        window_size=window_size,
        reward_scaling=3.0,         # Increased from 2.0
        risk_aversion=0.2,          # Decreased from 0.3
        drawdown_penalty=0.3,       # Decreased from 0.5
        opportunity_cost=0.1,       # Increased from 0.05
        drawdown_threshold=0.15,    # Increased from 0.08
        max_drawdown_pct=0.40,      # Increased from 0.30
        include_sentiment=False,
        max_position_pct=0.5,       # Increased from 0.4
        min_position_pct=0.05,
        curriculum_level=3,         # Always use highest level for evaluation
        debug_mode=debug_mode,      # Enable debug logging
        min_episode_length=20       # Minimum episode length before early termination
    )
    
    # Wrap the environment with Monitor for logging
    env = Monitor(env)
    
    return env


def train_agent(
    env: gym.Env,
    model_dir: str,
    log_dir: str,
    total_timesteps: int = 250000,
    eval_env: Optional[gym.Env] = None,
    use_curriculum: bool = False,
    use_gpu: bool = True,
    seed: Optional[int] = None
) -> SAC:
    """
    Train the SAC agent.
    
    Args:
        env: Training environment
        model_dir: Directory to save models
        log_dir: Directory to save logs
        total_timesteps: Total timesteps for training
        eval_env: Evaluation environment
        use_curriculum: Whether to use curriculum learning
        use_gpu: Whether to use GPU for training
        seed: Random seed for reproducibility
        
    Returns:
        model: Trained SAC agent
    """
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Determine device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Training on device: {device}")
    
    # Create the SAC agent with optimized hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=5e-5,      # Reduced from 1e-4 for more stable learning
        gamma=0.99,
        buffer_size=500000,      # Keep large buffer
        batch_size=512,          # Keep large batch size
        tau=0.01,                # Soft update coefficient
        ent_coef='auto_0.1',     # Higher minimum entropy for better exploration
        target_update_interval=1,
        gradient_steps=1,        # Stick with 1 to avoid overtraining
        learning_starts=5000,    # Reasonable amount before starting updates
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 128],  # Actor network
                qf=[512, 256, 128]   # Critic network
            ),
            # Added optimization for GPU if available
            optimizer_kwargs=dict(
                eps=1e-5,
                weight_decay=0.0001  # Small weight decay to reduce overfitting
            )
        ),
        tensorboard_log=log_dir,
        verbose=1,
        device=device,           # Explicitly set the device
        seed=seed                # Set random seed if provided
    )
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="sac_stock_model"
    )
    
    callbacks = [checkpoint_callback]
    
    # Add evaluation callback if eval_env is provided
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=5000,  # Evaluate every 5000 steps
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)
        
        # Add metrics logger callback
        metrics_logger = MetricsLoggerCallback(
            eval_env=eval_env,
            log_path=os.path.join(log_dir, "training_metrics.csv"),
            log_freq=5000  # Log every 5000 steps
        )
        callbacks.append(metrics_logger)
        
    # Add curriculum learning callback if enabled
    if use_curriculum:
        curriculum_callback = CurriculumLearningCallback(
            env, 
            target_reward=0.3,  # Lowered from 0.5 to make advancement easier
            window_size=20,     # Number of episodes to average over
            verbose=1
        )
        callbacks.append(curriculum_callback)
    
    # Train the agent
    print(f"Training agent for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=100
    )
    
    # Save the final model
    model_path = os.path.join(model_dir, "sac_stock_final_model")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Instructions for TensorBoard
    print("\nTo visualize training progress, run:")
    print(f"tensorboard --logdir={log_dir}")
    
    return model


def evaluate_agent(
    model: SAC,
    env: gym.Env,
    n_episodes: int = 5,
    render: bool = False
) -> Dict[str, float]:
    """
    Evaluate the trained agent.
    
    Args:
        model: Trained SAC agent
        env: Evaluation environment
        n_episodes: Number of evaluation episodes
        render: Whether to render the environment
        
    Returns:
        metrics: Evaluation metrics
    """
    print(f"Evaluating agent for {n_episodes} episodes...")
    
    # Initialize metrics
    metrics = {
        'total_return': [],
        'sharpe_ratio': [],
        'max_drawdown': [],
        'win_rate': [],
        'episode_length': []
    }
    
    # Access the unwrapped environment for diagnostics
    unwrapped_env = env.envs[0].env.env if hasattr(env, 'envs') else env
    
    # Reset diagnostic counters
    unwrapped_env.stop_loss_count = 0
    unwrapped_env.take_profit_count = 0
    unwrapped_env.max_holding_count = 0
    unwrapped_env.exit_signal_count = 0
    
    # Run evaluation episodes
    for episode in range(n_episodes):
        print(f"Episode {episode+1}/{n_episodes}")
        
        # Reset environment
        observation, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        returns = [1.0]  # Start with initial portfolio value normalized to 1
        daily_returns = []
        wins = 0
        trades = 0
        steps = 0
        
        # Run episode
        while not (done or truncated):
            # Get action from model
            action, _ = model.predict(observation, deterministic=True)
            
            # Take action in environment
            next_observation, reward, done, truncated, info = env.step(action)
            
            # Render if requested
            if render:
                env.render()
                
            # Update metrics
            total_reward += reward
            returns.append(info['portfolio_value'] / info['initial_capital'])
            
            # Calculate daily return
            if len(returns) > 1:
                daily_return = returns[-1] / returns[-2] - 1
                daily_returns.append(daily_return)
            
            # Update trade tracking
            if info.get('trade_completed', False):
                trades += 1
                if info.get('trade_profit', 0) > 0:
                    wins += 1
            
            # Update observation
            observation = next_observation
            steps += 1
        
        # Calculate episode metrics
        episode_return = returns[-1] - 1  # Final return
        
        # Calculate Sharpe ratio
        if len(daily_returns) > 1:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0
            
        # Calculate maximum drawdown
        max_drawdown = 0
        peak = returns[0]
        for r in returns:
            if r > peak:
                peak = r
            dd = (peak - r) / peak
            if dd > max_drawdown:
                max_drawdown = dd
                
        # Calculate win rate
        win_rate = wins / trades if trades > 0 else 0
        
        # Store metrics
        metrics['total_return'].append(episode_return)
        metrics['sharpe_ratio'].append(sharpe)
        metrics['max_drawdown'].append(max_drawdown)
        metrics['win_rate'].append(win_rate)
        metrics['episode_length'].append(steps)
        
        # Print episode summary
        print(f"Episode {episode+1} summary:")
        print(f"  Total return: {episode_return:.2%}")
        print(f"  Sharpe ratio: {sharpe:.2f}")
        print(f"  Max drawdown: {max_drawdown:.2%}")
        print(f"  Win rate: {win_rate:.2%} ({wins}/{trades} trades)")
        print(f"  Episode length: {steps} steps")
    
    # Calculate average metrics
    avg_metrics = {
        'avg_return': np.mean(metrics['total_return']),
        'avg_sharpe': np.mean(metrics['sharpe_ratio']),
        'avg_max_drawdown': np.mean(metrics['max_drawdown']),
        'avg_win_rate': np.mean(metrics['win_rate']),
        'avg_episode_length': np.mean(metrics['episode_length'])
    }
    
    # Get diagnostic counters from environment
    diagnostics = {
        'stop_loss_count': unwrapped_env.stop_loss_count,
        'take_profit_count': unwrapped_env.take_profit_count,
        'max_holding_count': unwrapped_env.max_holding_count,
        'exit_signal_count': unwrapped_env.exit_signal_count
    }
    
    # Print average metrics
    print("\nAverage evaluation metrics:")
    print(f"  Average return: {avg_metrics['avg_return']:.2%}")
    print(f"  Average Sharpe ratio: {avg_metrics['avg_sharpe']:.2f}")
    print(f"  Average max drawdown: {avg_metrics['avg_max_drawdown']:.2%}")
    print(f"  Average win rate: {avg_metrics['avg_win_rate']:.2%}")
    print(f"  Average episode length: {avg_metrics['avg_episode_length']:.1f} steps")
    
    # Print diagnostics
    print("\nDiagnostic counters:")
    print(f"  Stop-loss exits: {diagnostics['stop_loss_count']}")
    print(f"  Take-profit exits: {diagnostics['take_profit_count']}")
    print(f"  Max holding period exits: {diagnostics['max_holding_count']}")
    print(f"  Manual exit signal exits: {diagnostics['exit_signal_count']}")
    
    # Calculate trade outcomes
    total_trades = sum(diagnostics.values())
    if total_trades > 0:
        print(f"  Total trades: {total_trades}")
        print(f"  Stop-loss rate: {diagnostics['stop_loss_count']/total_trades:.2%}")
        print(f"  Take-profit rate: {diagnostics['take_profit_count']/total_trades:.2%}")
        print(f"  Max holding rate: {diagnostics['max_holding_count']/total_trades:.2%}")
        print(f"  Exit signal rate: {diagnostics['exit_signal_count']/total_trades:.2%}")
    
    return avg_metrics


def run_baseline_strategy(
    test_data: pd.DataFrame,
    strategy: str = 'buy_and_hold',
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.0015
) -> Dict[str, float]:
    """
    Run a baseline trading strategy for comparison.
    
    Args:
        test_data: Stock data for evaluation
        strategy: Baseline strategy ('buy_and_hold', 'moving_average', or 'rsi')
        initial_capital: Initial capital amount
        transaction_cost_pct: Transaction cost percentage
        
    Returns:
        metrics: Strategy performance metrics
    """
    print(f"Running baseline strategy: {strategy}")
    
    # Initialize metrics
    portfolio_value = initial_capital
    portfolio_values = [portfolio_value]
    daily_returns = []
    current_position = 0
    entry_price = 0
    trades = 0
    wins = 0
    
    if strategy == 'buy_and_hold':
        # Buy and hold strategy - buy on first day, hold until end
        # Buy as many shares as possible on first day
        price = test_data.iloc[0]['close']
        max_shares = portfolio_value / (price * (1 + transaction_cost_pct))
        shares = int(max_shares)  # Whole shares only
        cost = shares * price * (1 + transaction_cost_pct)
        portfolio_value -= cost
        current_position = shares
        entry_price = price
        
        # Track portfolio value for each day
        for i in range(1, len(test_data)):
            price = test_data.iloc[i]['close']
            stock_value = current_position * price
            portfolio_value = portfolio_value + stock_value
            portfolio_values.append(portfolio_value)
            daily_return = portfolio_values[-1] / portfolio_values[-2] - 1 if len(portfolio_values) > 1 else 0
            daily_returns.append(daily_return)
            
        # Sell at the end
        if current_position > 0:
            price = test_data.iloc[-1]['close']
            proceeds = current_position * price * (1 - transaction_cost_pct)
            if proceeds > current_position * entry_price:
                wins += 1
            trades += 1
    
    elif strategy == 'moving_average':
        # Moving average crossover strategy (50/200 day)
        # Calculate moving averages
        test_data['sma_50'] = test_data['close'].rolling(window=50).mean()
        test_data['sma_200'] = test_data['close'].rolling(window=200).mean()
        
        # Wait for enough data to calculate moving averages
        start_idx = 200
        portfolio_values = [initial_capital] * start_idx
        
        # Initial portfolio value
        portfolio_value = initial_capital
        
        # Implement strategy
        for i in range(start_idx, len(test_data)):
            price = test_data.iloc[i]['close']
            sma_50 = test_data.iloc[i]['sma_50']
            sma_200 = test_data.iloc[i]['sma_200']
            prev_sma_50 = test_data.iloc[i-1]['sma_50']
            prev_sma_200 = test_data.iloc[i-1]['sma_200']
            
            # Buy signal: 50-day MA crosses above 200-day MA
            if prev_sma_50 <= prev_sma_200 and sma_50 > sma_200 and current_position == 0:
                # Buy with 10% of capital
                position_size = portfolio_value * 0.1
                shares = int(position_size / (price * (1 + transaction_cost_pct)))
                cost = shares * price * (1 + transaction_cost_pct)
                portfolio_value -= cost
                current_position = shares
                entry_price = price
                
            # Sell signal: 50-day MA crosses below 200-day MA
            elif prev_sma_50 >= prev_sma_200 and sma_50 < sma_200 and current_position > 0:
                # Sell all shares
                proceeds = current_position * price * (1 - transaction_cost_pct)
                portfolio_value += proceeds
                if proceeds > current_position * entry_price:
                    wins += 1
                trades += 1
                current_position = 0
                entry_price = 0
                
            # Update portfolio value
            stock_value = current_position * price
            current_portfolio_value = portfolio_value + stock_value
            portfolio_values.append(current_portfolio_value)
            daily_return = portfolio_values[-1] / portfolio_values[-2] - 1
            daily_returns.append(daily_return)
            
        # Sell at the end if still holding
        if current_position > 0:
            price = test_data.iloc[-1]['close']
            proceeds = current_position * price * (1 - transaction_cost_pct)
            portfolio_value += proceeds
            if proceeds > current_position * entry_price:
                wins += 1
            trades += 1
    
    elif strategy == 'rsi':
        # RSI Mean-Reversion strategy
        # Calculate RSI
        test_data['rsi'] = talib.RSI(test_data['close'].values, timeperiod=14)
        
        # Wait for enough data to calculate RSI
        start_idx = 14
        portfolio_values = [initial_capital] * start_idx
        
        # Initial portfolio value
        portfolio_value = initial_capital
        
        # Implement strategy
        for i in range(start_idx, len(test_data)):
            price = test_data.iloc[i]['close']
            rsi = test_data.iloc[i]['rsi']
            prev_rsi = test_data.iloc[i-1]['rsi']
            
            # Buy signal: RSI crosses above 30 from below
            if prev_rsi < 30 and rsi >= 30 and current_position == 0:
                # Buy with 10% of capital
                position_size = portfolio_value * 0.1
                shares = int(position_size / (price * (1 + transaction_cost_pct)))
                cost = shares * price * (1 + transaction_cost_pct)
                portfolio_value -= cost
                current_position = shares
                entry_price = price
                
            # Sell signal: RSI crosses above 70
            elif rsi >= 70 and current_position > 0:
                # Sell all shares
                proceeds = current_position * price * (1 - transaction_cost_pct)
                portfolio_value += proceeds
                if proceeds > current_position * entry_price:
                    wins += 1
                trades += 1
                current_position = 0
                entry_price = 0
                
            # Update portfolio value
            stock_value = current_position * price
            current_portfolio_value = portfolio_value + stock_value
            portfolio_values.append(current_portfolio_value)
            daily_return = portfolio_values[-1] / portfolio_values[-2] - 1
            daily_returns.append(daily_return)
            
        # Sell at the end if still holding
        if current_position > 0:
            price = test_data.iloc[-1]['close']
            proceeds = current_position * price * (1 - transaction_cost_pct)
            portfolio_value += proceeds
            if proceeds > current_position * entry_price:
                wins += 1
            trades += 1
    
    # Calculate metrics
    returns = np.array(portfolio_values) / initial_capital
    total_return = returns[-1] - 1
    
    # Calculate Sharpe ratio
    if len(daily_returns) > 1:
        mean_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)
        sharpe_ratio = mean_daily_return / std_daily_return * np.sqrt(252) if std_daily_return > 0 else 0
    else:
        sharpe_ratio = 0
        
    # Calculate maximum drawdown
    max_drawdown = 0
    peak = returns[0]
    for r in returns:
        if r > peak:
            peak = r
        dd = (peak - r) / peak
        if dd > max_drawdown:
            max_drawdown = dd
            
    # Calculate win rate
    win_rate = wins / trades if trades > 0 else 0
    
    # Store metrics
    metrics = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trades': trades
    }
    
    # Print metrics
    print(f"\n{strategy.capitalize()} strategy performance:")
    print(f"  Total return: {metrics['total_return']:.2%}")
    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Win rate: {metrics['win_rate']:.2%} ({wins}/{trades} trades)")
    
    return metrics, returns, daily_returns


def compare_performance(
    rl_metrics: Dict[str, float],
    baseline_metrics: Dict[str, Dict[str, float]],
    test_data: pd.DataFrame
):
    """
    Compare RL agent performance with baseline strategies.
    
    Args:
        rl_metrics: RL agent performance metrics
        baseline_metrics: Baseline strategies performance metrics
        test_data: Test data used for evaluation
    """
    print("\n===== Performance Comparison =====")
    
    # Create comparison table
    metrics_names = {
        'total_return': 'Total Return',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown',
        'win_rate': 'Win Rate'
    }
    
    # Print header
    print(f"{'Metric':<15} {'RL Agent':<12}", end='')
    for strategy in baseline_metrics:
        print(f" {strategy.capitalize():<12}", end='')
    print()
    
    # Print metrics
    for metric_key, metric_name in metrics_names.items():
        print(f"{metric_name:<15}", end='')
        
        # RL Agent
        if metric_key == 'total_return' or metric_key == 'max_drawdown' or metric_key == 'win_rate':
            print(f" {rl_metrics.get(f'avg_{metric_key}', 0):.2%}", end='')
        else:
            print(f" {rl_metrics.get(f'avg_{metric_key}', 0):.2f}", end='')
            
        # Baseline strategies
        for strategy in baseline_metrics:
            if metric_key == 'total_return' or metric_key == 'max_drawdown' or metric_key == 'win_rate':
                print(f" {baseline_metrics[strategy][0].get(metric_key, 0):.2%}", end='')
            else:
                print(f" {baseline_metrics[strategy][0].get(metric_key, 0):.2f}", end='')
        
        print()


def main():
    """Main function to run the demonstration."""
    # Parse command line arguments
    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        print(f"Random seed set to {args.seed}")
    
    # Check if CUDA is available and print status
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} device(s):")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU for training.")
    
    # Load stock data
    stock_data = load_stock_data(args.ticker, args.start_date, args.end_date)
    
    # Split data into train and test sets
    split_idx = int(len(stock_data) * args.train_test_split)
    train_data = stock_data.iloc[:split_idx].copy()
    test_data = stock_data.iloc[split_idx:].copy()
    
    print(f"Training data: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
    print(f"Testing data: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} days)")
    
    # Create environments
    train_env = create_training_environment(
        train_data,
        initial_capital=args.initial_capital,
        window_size=20,
        render_mode=None,
        curriculum_level=1 if args.curriculum else 3,  # Start at level 1 for curriculum, max for regular
        debug_mode=args.debug
    )
    
    eval_env = create_evaluation_environment(
        test_data,
        initial_capital=args.initial_capital,
        window_size=20,
        render_mode='rgb_array' if args.render else None,
        debug_mode=args.debug
    )
    
    # Train or load agent
    if args.train:
        # Train agent
        model = train_agent(
            env=train_env,
            model_dir=args.model_dir,
            log_dir=args.log_dir,
            total_timesteps=args.total_timesteps,
            eval_env=eval_env,
            use_curriculum=args.curriculum,
            use_gpu=args.use_gpu,
            seed=args.seed
        )
    else:
        # Load trained agent if available
        model_path = os.path.join(args.model_dir, "sac_stock_final_model")
        if os.path.exists(model_path + ".zip"):
            print(f"Loading model from {model_path}.zip")
            # Specify device for loading
            device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
            model = SAC.load(model_path, env=train_env, device=device)
            print(f"Model loaded to device: {model.device}")
        else:
            print(f"Model not found at {model_path}.zip. Training required.")
            return
    
    # Evaluate if requested
    if args.evaluate:
        # Evaluate RL agent
        rl_metrics = evaluate_agent(
            model=model,
            env=eval_env,
            n_episodes=5,
            render=args.render
        )
        
        # Run baseline strategies
        baseline_metrics = {}
        baseline_metrics['buy_and_hold'] = run_baseline_strategy(
            test_data=test_data,
            strategy='buy_and_hold',
            initial_capital=args.initial_capital,
            transaction_cost_pct=0.0015
        )
        
        baseline_metrics['moving_average'] = run_baseline_strategy(
            test_data=test_data,
            strategy='moving_average',
            initial_capital=args.initial_capital,
            transaction_cost_pct=0.0015
        )
        
        baseline_metrics['rsi'] = run_baseline_strategy(
            test_data=test_data,
            strategy='rsi',
            initial_capital=args.initial_capital,
            transaction_cost_pct=0.0015
        )
        
        # Compare performance
        compare_performance(rl_metrics, baseline_metrics, test_data)


if __name__ == "__main__":
    main()