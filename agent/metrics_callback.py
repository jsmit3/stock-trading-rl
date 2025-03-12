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