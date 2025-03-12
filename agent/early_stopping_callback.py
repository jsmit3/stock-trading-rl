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