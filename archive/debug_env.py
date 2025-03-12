"""
debug_env.py

This script helps diagnose early termination issues in the trading environment.
It runs a few episodes with better random action generation to avoid negative share requests.

Usage:
python debug_env.py
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import SAC
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Import the trading environment
from trading_env import StockTradingEnv
from archive.data_processor import DataProcessor

def load_stock_data(ticker='AAPL', start_date='2018-01-01', end_date='2023-01-01'):
    """Load stock data from cache or generate synthetic data."""
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    start_year = start_date.split('-')[0]
    end_year = end_date.split('-')[0]
    cache_file = os.path.join(cache_dir, f"{ticker}_{start_year}_to_{end_year}.pkl")
    
    if os.path.exists(cache_file):
        print(f"Loading cached data for {ticker}")
        try:
            stock_data = pd.read_pickle(cache_file)
            
            # Add additional columns for date features
            stock_data['date'] = stock_data.index
            stock_data['day_of_week'] = stock_data.index.dayofweek
            stock_data['month'] = stock_data.index.month
            
            return stock_data
        except Exception as e:
            print(f"Error loading cached data: {e}")
    
    # If no cached data, generate synthetic data
    print("Generating synthetic data")
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = pd.date_range(start=start, end=end, freq='B')
    
    random.seed(42)
    np.random.seed(42)
    
    base_price = 150.0
    n_days = len(date_range)
    daily_returns = np.random.normal(0.0005, 0.015, n_days)
    price_series = base_price * (1 + daily_returns).cumprod()
    
    close_prices = price_series
    high_prices = close_prices * (1 + np.random.uniform(0, 0.02, n_days))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.02, n_days))
    open_prices = low_prices + np.random.uniform(0, 1, n_days) * (high_prices - low_prices)
    
    volume = np.random.lognormal(16, 0.5, n_days)
    
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
    
    synthetic_data.set_index('date', inplace=True)
    return synthetic_data

def create_debug_environment(data, initial_capital=100000.0, window_size=20, curriculum_level=3):
    """Create environment with debug mode enabled."""
    data_processor = DataProcessor(data, window_size=window_size)
    normalized_data = data_processor.normalize_data(data)
    
    env = StockTradingEnv(
        price_data=normalized_data,
        initial_capital=initial_capital,
        max_holding_period=20,
        transaction_cost_pct=0.001,
        window_size=window_size,
        reward_scaling=3.0,
        risk_aversion=0.2,          # Further reduced from 0.3
        drawdown_penalty=0.3,       # Further reduced from 0.5
        opportunity_cost=0.1,
        drawdown_threshold=0.15,    # Significantly increased from 0.08
        max_drawdown_pct=0.40,      # Significantly increased from 0.30
        include_sentiment=False,
        max_position_pct=0.5,       # Increased from 0.4
        min_position_pct=0.05,
        curriculum_level=curriculum_level,
        debug_mode=True,            # Enable debug mode
        min_episode_length=20       # Minimum episode length
    )
    
    return env

def generate_good_action():
    """Generate random actions that are more likely to result in valid trades."""
    # Modified random action that biases toward taking positions
    # Action: [position_size, stop_loss, take_profit, exit_signal]
    
    # Position size (biased toward taking positions)
    position_size = random.uniform(0.3, 1.0) if random.random() > 0.5 else 0.0
    
    # Stop loss and take profit (sensible ranges)
    stop_loss = random.uniform(0.01, 0.1) 
    take_profit = random.uniform(0.01, 0.2)
    
    # Exit signal (mostly stay in positions)
    exit_signal = 0.0 if random.random() > 0.3 else 1.0
    
    return np.array([position_size, stop_loss, take_profit, exit_signal], dtype=np.float32)

def run_random_agent(env, n_episodes=3):
    """Run random agent to test environment behavior."""
    print("\n=== Testing with Random Agent ===")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        print(f"\nEpisode {episode+1} start info:")
        print(f"  Date: {info['date']}")
        print(f"  Initial portfolio: ${info['portfolio_value']:.2f}")
        
        terminated = False
        truncated = False
        step_count = 0
        actions_taken = []
        
        # Track trading activity
        positions_opened = 0
        trades_completed = 0
        
        while not (terminated or truncated):
            # Generate better random actions
            action = generate_good_action()
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Record action
            action_info = {
                'step': step_count,
                'position_size': action[0],
                'stop_loss': action[1],
                'take_profit': action[2],
                'exit_signal': action[3],
                'reward': reward,
                'portfolio_value': info['portfolio_value'],
                'current_position': info['current_position'],
                'drawdown': info['drawdown']
            }
            actions_taken.append(action_info)
            
            # Track when a position is opened
            if info.get('trade_executed', False) and info.get('trade_type') == 'buy':
                positions_opened += 1
                print(f"Opened position at step {step_count}, price: ${info['trade_price']:.2f}, shares: {info['trade_shares']:.2f}")
                
            # Track when a trade is completed
            if info.get('trade_completed', False):
                trades_completed += 1
                profit_loss = info.get('trade_profit', 0)
                profit_pct = info.get('trade_profit_pct', 0)
                reason = info.get('trade_reason', 'unknown')
                print(f"Closed position at step {step_count}, profit/loss: ${profit_loss:.2f} ({profit_pct:.2f}%), reason: {reason}")
            
            # Print step info
            if step_count % 10 == 0 or terminated or truncated:
                print(f"  Step {step_count}: Reward = {reward:.4f}, Portfolio = ${info['portfolio_value']:.2f}, Drawdown = {info['drawdown']:.2%}")
            
            # Check if we're holding a position
            if info['current_position'] > 0:
                print(f"    Holding position: {info['current_position']:.2f} shares, P&L: {info['position_pnl']:.2%}")
                
            # Print diagnostics if available
            if 'diagnostics' in info and (terminated or truncated or step_count % 50 == 0):
                diagnostics = info['diagnostics']
                print(f"    Diagnostics: SL={diagnostics['stop_loss_count']}, TP={diagnostics['take_profit_count']}, MH={diagnostics['max_holding_count']}, EX={diagnostics['exit_signal_count']}")
        
        # Print episode summary
        print(f"\nEpisode {episode+1} ended after {step_count} steps")
        print(f"  Final portfolio: ${info['portfolio_value']:.2f} ({(info['portfolio_value']/info['initial_capital']-1)*100:.2f}%)")
        print(f"  Positions opened: {positions_opened}, Trades completed: {trades_completed}")
        
        # Print termination reason if available
        if 'termination_reason' in info:
            print(f"  Termination reason: {info['termination_reason']}")
        elif truncated:
            print("  Episode was truncated (reached end of data)")
        else:
            print("  Episode ended normally")
        
        # Print action statistics
        if actions_taken:
            actions_df = pd.DataFrame(actions_taken)
            print("\nAction statistics:")
            print(f"  Position size: min={actions_df['position_size'].min():.2f}, max={actions_df['position_size'].max():.2f}, mean={actions_df['position_size'].mean():.2f}")
            print(f"  Stop loss: min={actions_df['stop_loss'].min():.2f}, max={actions_df['stop_loss'].max():.2f}, mean={actions_df['stop_loss'].mean():.2f}")
            print(f"  Take profit: min={actions_df['take_profit'].min():.2f}, max={actions_df['take_profit'].max():.2f}, mean={actions_df['take_profit'].mean():.2f}")
            
    print("\nRandom agent testing complete.")

def main():
    # Load data
    print("Loading stock data...")
    stock_data = load_stock_data()
    
    # Create debug environment
    print("Creating debug environment...")
    env = create_debug_environment(stock_data)
    
    # Run random agent
    run_random_agent(env, n_episodes=3)
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()