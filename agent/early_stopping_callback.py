"""
agent/early_stopping_callback.py

Implements early stopping based on trading-specific metrics.

This callback extends the standard EvalCallback to consider:
- Win rate
- Sharpe ratio
- Maximum drawdown
- Consistent profitability

Author: [Your Name]
Date: March 13, 2025
"""

import os
import numpy as np
from typing import Dict, Any
from stable_baselines3.common.callbacks import EvalCallback
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
        
    def _evaluate_trading_metrics(self) -> tuple:
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
        env = self._get_unwrapped_env()
        
        # Default metrics if we can't get the environment
        wins = 0
        losses = 0
        
        # Try to get metrics from environment
        if env is not None:
            wins = getattr(env, 'take_profit_count', 0) + getattr(env, 'exit_signal_count', 0)
            losses = getattr(env, 'stop_loss_count', 0) + getattr(env, 'max_holding_count', 0)
        
        # Calculate returns and drawdowns
        returns = np.array(episode_rewards) / (env.initial_capital if env is not None else 100000.0)
        
        # Approximate Sharpe ratio using episode returns
        mean_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1.0
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Approximate max drawdown from returns
        max_dd = getattr(env, 'state', {}).drawdown if hasattr(env, 'state') and hasattr(env.state, 'drawdown') else 0.0
        
        return wins, losses, sharpe, max_dd
    
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