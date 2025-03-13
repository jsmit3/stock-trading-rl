"""
agent/metrics_callback.py

Implements the MetricsLoggerCallback for capturing training metrics.

This module provides a callback for:
- Logging detailed trading metrics 
- Recording environment diagnostics
- Tracking performance over time

Author: [Your Name]
Date: March 13, 2025
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, is_vecenv_wrapped

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
            env = self._get_unwrapped_env()
            
            # Initialize diagnostics with default values
            diagnostics = {
                'sl_count': 0,
                'tp_count': 0,
                'max_hold_count': 0,
                'exit_count': 0
            }
            
            # Extract counters if they exist in the environment
            if env is not None:
                diagnostics['sl_count'] = getattr(env, 'stop_loss_count', 0)
                diagnostics['tp_count'] = getattr(env, 'take_profit_count', 0)
                diagnostics['max_hold_count'] = getattr(env, 'max_holding_count', 0)
                diagnostics['exit_count'] = getattr(env, 'exit_signal_count', 0)
                
                # Reset diagnostics after evaluation
                if hasattr(env, 'stop_loss_count'):
                    env.stop_loss_count = 0
                if hasattr(env, 'take_profit_count'):
                    env.take_profit_count = 0
                if hasattr(env, 'max_holding_count'):
                    env.max_holding_count = 0
                if hasattr(env, 'exit_signal_count'):
                    env.exit_signal_count = 0
            
            # Calculate win rate
            trades = diagnostics['sl_count'] + diagnostics['tp_count'] + diagnostics['max_hold_count'] + diagnostics['exit_count']
            win_rate = (diagnostics['tp_count'] + diagnostics['exit_count']) / max(trades, 1)
            loss_rate = diagnostics['sl_count'] / max(trades, 1)
            
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

    def _get_unwrapped_env(self):
        """
        Safely unwrap the environment to get access to the base StockTradingEnv.
        This handles both VecEnv wrapped environments and regular environments.
        
        Returns:
            The unwrapped StockTradingEnv or None if it can't be unwrapped
        """
        env = self.eval_env
        
        # Handle VecEnv
        if isinstance(env, VecEnv):
            try:
                # For VecEnv, first get the first env
                if hasattr(env, 'envs'):
                    base_env = env.envs[0]
                elif hasattr(env, 'env'):
                    # Sometimes VecEnv has a different structure
                    base_env = env.env
                else:
                    # If we can't find a structure we recognize, try the first venv
                    base_env = env.venv.envs[0] if hasattr(env, 'venv') and hasattr(env.venv, 'envs') else None
                
                # Now unwrap the base env until we get to StockTradingEnv
                while base_env is not None and hasattr(base_env, 'env'):
                    if hasattr(base_env, 'env') and base_env.__class__.__name__ == 'StockTradingEnv':
                        return base_env
                    base_env = base_env.env
                
                # If we've reached the innermost env
                if base_env is not None and base_env.__class__.__name__ == 'StockTradingEnv':
                    return base_env
            except Exception as e:
                print(f"Warning: Could not unwrap VecEnv: {e}")
                return None
        
        # Handle non-VecEnv (direct environment)
        else:
            # Unwrap until we get to StockTradingEnv
            current_env = env
            while current_env is not None:
                if current_env.__class__.__name__ == 'StockTradingEnv':
                    return current_env
                
                if hasattr(current_env, 'env'):
                    current_env = current_env.env
                else:
                    break
        
        # If we couldn't find it
        if self.verbose > 0:
            print("Warning: Could not find StockTradingEnv in environment chain.")
        return None