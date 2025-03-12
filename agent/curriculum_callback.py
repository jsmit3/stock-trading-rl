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


