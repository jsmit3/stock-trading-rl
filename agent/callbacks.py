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


class CurriculumLearningCallback(BaseCallback):
    """
    Implements curriculum learning by advancing difficulty levels.
    
    This callback:
    - Monitors agent performance over a window of episodes
    - Advances curriculum level when performance threshold is met
    - Resets metrics when advancing to a new level
    """
    
    def __init__(
        self,
        env,
        target_reward: float = 0.5,
        window_size: int = 20,
        verbose: int = 0
    ):
        """
        Initialize the curriculum learning callback.
        
        Args:
            env: Training environment
            target_reward: Target reward to trigger curriculum advancement
            window_size: Number of episodes to average over
            verbose: Verbosity level
        """
        super(CurriculumLearningCallback, self).__init__(verbose)
        self.env = env
        self.target_reward = target_reward
        self.window_size = window_size
        self.rewards = []
    
    def _init_callback(self) -> None:
        """Initialize the callback."""
        self.rewards = []
    
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            continue_training: Whether to continue training
        """
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


class MetricsLoggerCallback(BaseCallback):
    """
    Logs detailed metrics during training.
    
    This callback:
    - Tracks trading performance metrics (win rate, P&L, etc.)
    - Records environment diagnostics
    - Saves metrics to CSV for later analysis
    """
    
    def __init__(
        self,
        eval_env,
        log_path: str,
        log_freq: int = 5000,
        verbose: int = 1
    ):
        """
        Initialize the metrics logger callback.
        
        Args:
            eval_env: Environment to use for evaluation
            log_path: Path to save logs
            log_freq: Frequency of logging (in timesteps)
            verbose: Verbosity level
        """
        super(MetricsLoggerCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.log_path = log_path
        self.log_freq = log_freq
        self.metrics = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
    def _init_callback(self) -> None:
        """Initialize the callback."""
        # Create the log file if it doesn't exist
        with open(self.log_path, "w") as f:
            f.write("timestep,avg_reward,avg_length,win_rate,loss_rate,sl_count,tp_count,max_hold_count,exit_count\n")
        
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            continue_training: Whether to continue training
        """
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


class TradingEarlyStoppingCallback(EvalCallback):
    """
    Early stopping based on trading-specific metrics.
    
    This callback extends the standard EvalCallback to consider:
    - Win rate
    - Sharpe ratio
    - Maximum drawdown
    - Consistent profitability
    """
    
    def __init__(
        self,
        eval_env,
        log_dir: str,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        warning_threshold: int = 3,
        min_evals: int = 5,
        check_win_rate: bool = True,
        target_win_rate: float = 0.5,
        check_sharpe: bool = True,
        target_sharpe: float = 1.0,
        verbose: int = 1
    ):
        """
        Initialize the early stopping callback.
        
        Args:
            eval_env: Environment to use for evaluation
            log_dir: Directory to save logs
            n_eval_episodes: Number of episodes for each evaluation
            eval_freq: Frequency of evaluation
            warning_threshold: Number of consecutive degradations before stopping
            min_evals: Minimum number of evaluations before early stopping
            check_win_rate: Whether to check win rate
            target_win_rate: Target win rate for early stopping
            check_sharpe: Whether to check Sharpe ratio
            target_sharpe: Target Sharpe ratio for early stopping
            verbose: Verbosity level
        """
        super(TradingEarlyStoppingCallback, self).__init__(
            eval_env=eval_env,
            log_path=log_dir,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=n_eval_episodes,
            verbose=verbose
        )
        
        self.warning_count = 0
        self.warning_threshold = warning_threshold
        self.eval_count = 0
        self.min_evals = min_evals
        self.check_win_rate = check_win_rate
        self.target_win_rate = target_win_rate
        self.check_sharpe = check_sharpe
        self.target_sharpe = target_sharpe
        
        # Storage for metrics
        self.win_rates = []
        self.sharpe_ratios = []
        self.drawdowns = []
        self.pnl_history = []
        
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        
        Returns:
            continue_training: Whether to continue training
        """
        # Call the parent class method to evaluate the model
        continue_training = super()._on_step()
        
        # Check if an evaluation was just performed
        if self.num_timesteps % self.eval_freq == 0:
            self.eval_count += 1
            
            # Evaluate trading-specific metrics
            wins, losses, sharpe, max_dd = self._evaluate_trading_metrics()
            
            # Calculate win rate
            total_trades = wins + losses
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            # Store metrics
            self.win_rates.append(win_rate)
            self.sharpe_ratios.append(sharpe)
            self.drawdowns.append(max_dd)
            
            # Check for early stopping conditions if we have enough evaluations
            if self.eval_count >= self.min_evals:
                stop_training = False
                
                # Check if win rate is consistently below target
                if self.check_win_rate and len(self.win_rates) >= 3:
                    if all(wr < self.target_win_rate for wr in self.win_rates[-3:]):
                        if self.verbose > 0:
                            print(f"Win rate consistently below target ({self.target_win_rate})")
                        self.warning_count += 1
                
                # Check if Sharpe ratio is consistently negative
                if self.check_sharpe and len(self.sharpe_ratios) >= 3:
                    if all(sr < 0 for sr in self.sharpe_ratios[-3:]):
                        if self.verbose > 0:
                            print("Sharpe ratio consistently negative")
                        self.warning_count += 1
                
                # Check if warnings exceed threshold
                if self.warning_count >= self.warning_threshold:
                    if self.verbose > 0:
                        print(f"Early stopping triggered after {self.eval_count} evaluations")
                    return False  # Stop training
            
            # Log the metrics
            if self.verbose > 0:
                print(f"Trading metrics - Win rate: {win_rate:.2f}, Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2%}")
        
        return continue_training
        
    def _evaluate_trading_metrics(self) -> Tuple[int, int, float, float]:
        """
        Evaluate trading-specific metrics.
        
        Returns:
            wins: Number of winning trades
            losses: Number of losing trades
            sharpe: Sharpe ratio
            max_dd: Maximum drawdown
        """
        # Run evaluation episodes to collect metrics
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            return_episode_rewards=True,
            deterministic=True
        )
        
        # Get the unwrapped environment
        env = self.eval_env.envs[0].env.env  # Unwrap to get StockTradingEnv
        
        # Get trading metrics
        wins = env.take_profit_count + env.exit_signal_count
        losses = env.stop_loss_count + env.max_holding_count
        
        # Calculate returns and drawdowns
        returns = np.array(episode_rewards) / env.initial_capital
        # Approximate Sharpe ratio using episode returns
        mean_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1.0
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Approximate max drawdown from returns
        max_dd = env.state.drawdown if hasattr(env, 'state') else 0.0
        
        return wins, losses, sharpe, max_dd


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