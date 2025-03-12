
"""
agent/callbacks.py

Implements custom callbacks for the reinforcement learning training process.

This module provides specialized callbacks for:
- Curriculum learning progression
- Performance metrics logging
- Early stopping
- Trading-specific visualizations

Author: [Your Name]
Date: March 10, 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Union, Tuple, Tuple
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv




class VisualizeTradesCallback(BaseCallback):
    """
    Creates visualizations of trading activity during training.
    
    This callback:
    - Plots position sizes, P&L, and portfolio value
    - Visualizes trade entry and exit points
    - Creates equity curve and drawdown charts
    """
    
    def __init__(
        self,
        eval_env,
        log_dir: str,
        plot_freq: int = 20000,
        n_eval_episodes: int = 1,
        verbose: int = 1
    ):
        """
        Initialize the visualization callback.
        
        Args:
            eval_env: Environment to use for evaluation
            log_dir: Directory to save visualizations
            plot_freq: Frequency of plotting (in timesteps)
            n_eval_episodes: Number of episodes to visualize
            verbose: Verbosity level
        """
        super(VisualizeTradesCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.plot_freq = plot_freq
        self.n_eval_episodes = n_eval_episodes
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            continue_training: Whether to continue training
        """
        if self.n_calls % self.plot_freq == 0:
            self._generate_trade_visualizations()
        return True
    
    def _generate_trade_visualizations(self) -> None:
        """Generate trading visualizations."""
        # Run evaluation episodes to collect data
        episodes_data = []
        
        for episode in range(self.n_eval_episodes):
            # Reset environment
            obs, _ = self.eval_env.reset()
            done = False
            
            # Collect data for this episode
            episode_data = {
                'prices': [],
                'positions': [],
                'portfolio_values': [],
                'trades': [],
                'returns': [],
                'drawdowns': []
            }
            
            # Run episode
            while not done:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Take action in environment
                next_obs, reward, done, _, info = self.eval_env.step(action)
                
                # Record data
                episode_data['prices'].append(info['price'])
                episode_data['positions'].append(info['current_position'])
                episode_data['portfolio_values'].append(info['portfolio_value'])
                episode_data['drawdowns'].append(info['drawdown'])
                
                # Record trade if completed
                if info.get('trade_completed', False):
                    trade_info = {
                        'step': info['step'],
                        'price': info['trade_price'],
                        'type': info['trade_type'],
                        'profit': info.get('trade_profit', 0),
                        'reason': info.get('trade_reason', 'unknown')
                    }
                    episode_data['trades'].append(trade_info)
                
                # Calculate returns
                if len(episode_data['portfolio_values']) > 1:
                    ret = (episode_data['portfolio_values'][-1] / episode_data['portfolio_values'][-2]) - 1
                    episode_data['returns'].append(ret)
                else:
                    episode_data['returns'].append(0)
                
                # Update observation
                obs = next_obs
            
            episodes_data.append(episode_data)
        
        # Create visualizations for each episode
        for i, episode_data in enumerate(episodes_data):
            # Plot 1: Equity curve and drawdown
            self._plot_equity_curve(episode_data, i)
            
            # Plot 2: Price and position size
            self._plot_trades(episode_data, i)
    
    def _plot_equity_curve(self, episode_data: Dict, episode_num: int) -> None:
        """
        Plot equity curve and drawdown.
        
        Args:
            episode_data: Data for the episode
            episode_num: Episode number
        """
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(episode_data['portfolio_values'], label='Portfolio Value')
        plt.title(f'Equity Curve - Episode {episode_num+1}')
        plt.xlabel('Step')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        plt.plot(np.array(episode_data['drawdowns']) * 100, label='Drawdown', color='red')
        plt.fill_between(range(len(episode_data['drawdowns'])), 
                         np.array(episode_data['drawdowns']) * 100, 
                         0, 
                         color='red', 
                         alpha=0.3)
        plt.title(f'Drawdown - Episode {episode_num+1}')
        plt.xlabel('Step')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'equity_curve_ep{episode_num+1}_step{self.n_calls}.png'), dpi=100)
        plt.close()
    
    def _plot_trades(self, episode_data: Dict, episode_num: int) -> None:
        """
        Plot price, trades, and position size.
        
        Args:
            episode_data: Data for the episode
            episode_num: Episode number
        """
        plt.figure(figsize=(12, 8))
        
        # Plot price and trades
        plt.subplot(2, 1, 1)
        plt.plot(episode_data['prices'], label='Price')
        
        # Mark buy trades
        buy_steps = [t['step'] for t in episode_data['trades'] if t['type'] == 'buy']
        buy_prices = [episode_data['prices'][min(s, len(episode_data['prices'])-1)] for s in buy_steps]
        plt.scatter(buy_steps, buy_prices, color='green', marker='^', s=100, label='Buy')
        
        # Mark sell trades
        sell_steps = [t['step'] for t in episode_data['trades'] if t['type'] == 'sell']
        sell_prices = [episode_data['prices'][min(s, len(episode_data['prices'])-1)] for s in sell_steps]
        plt.scatter(sell_steps, sell_prices, color='red', marker='v', s=100, label='Sell')
        
        plt.title(f'Price and Trades - Episode {episode_num+1}')
        plt.xlabel('Step')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot position size
        plt.subplot(2, 1, 2)
        plt.bar(range(len(episode_data['positions'])), episode_data['positions'], label='Position Size')
        plt.title(f'Position Size - Episode {episode_num+1}')
        plt.xlabel('Step')
        plt.ylabel('Shares')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'trades_ep{episode_num+1}_step{self.n_calls}.png'), dpi=100)
        plt.close()