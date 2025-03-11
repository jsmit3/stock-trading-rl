"""
agent/model.py

Implements the reinforcement learning agent for stock trading.

This module provides a wrapper for the SAC algorithm from Stable Baselines 3
and handles model initialization, hyperparameter tuning, and GPU utilization.

Author: [Your Name]
Date: March 10, 2025
"""

import os
import numpy as np
import torch
from typing import Optional, Dict, Any, Union, Tuple, List
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


class TradingAgent:
    """
    Trading agent based on Soft Actor-Critic (SAC) algorithm.
    
    This class provides:
    - Model configuration and initialization
    - Training with hyperparameters optimized for trading
    - Inference for making trading decisions
    - Utilities for managing model checkpoints
    """
    
    def __init__(
        self,
        env,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
        learning_rate: float = 5e-5,
        buffer_size: int = 500000,
        batch_size: int = 512,
        device: Optional[str] = None,
        verbose: int = 1,
        seed: Optional[int] = None
    ):
        """
        Initialize the trading agent.
        
        Args:
            env: Trading environment
            model_path: Path to load a pre-trained model
            use_gpu: Whether to use GPU for training if available
            learning_rate: Learning rate for optimization
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            device: Device to use (auto, cpu, cuda, or cuda:0, cuda:1, etc.)
            verbose: Verbosity level
            seed: Random seed for reproducibility
        """
        self.env = env
        self.use_gpu = use_gpu
        self.model_path = model_path
        self.verbose = verbose
        self.seed = seed
        
        # Determine device (GPU or CPU)
        if device is None:
            self.device = self._get_device()
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Create or load the model
        if model_path is not None and os.path.exists(model_path + ".zip"):
            print(f"Loading model from {model_path}.zip")
            self.model = SAC.load(model_path, env=env, device=self.device)
        else:
            print("Creating new model")
            self.model = self._create_model(
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size
            )
    
    def _get_device(self) -> str:
        """
        Determine the appropriate device to use.
        
        Returns:
            device: The device to use (cpu or cuda:0, etc.)
        """
        if self.use_gpu and torch.cuda.is_available():
            # Get the device with the most free memory
            if torch.cuda.device_count() > 1:
                # If multiple GPUs, choose one with most free memory
                free_mem = []
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    free_mem.append(torch.cuda.get_device_properties(i).total_memory - 
                                    torch.cuda.memory_allocated(i))
                device_id = free_mem.index(max(free_mem))
                return f"cuda:{device_id}"
            else:
                return "cuda:0"
        else:
            return "cpu"
    
    def _create_model(
        self,
        learning_rate: float = 5e-5,
        buffer_size: int = 500000,
        batch_size: int = 512
    ) -> SAC:
        """
        Create and configure the SAC model.
        
        Args:
            learning_rate: Learning rate for optimization
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            
        Returns:
            model: Configured SAC model
        """
        model = SAC(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            gamma=0.99,
            buffer_size=buffer_size,
            batch_size=batch_size,
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
            verbose=self.verbose,
            device=self.device,
            seed=self.seed
        )
        
        return model
    
    def train(
        self,
        total_timesteps: int,
        callback_list: Optional[List[BaseCallback]] = None,
        eval_env=None,
        eval_freq: int = 10000,
        log_dir: Optional[str] = None,
        model_dir: Optional[str] = None
    ) -> None:
        """
        Train the agent.
        
        Args:
            total_timesteps: Number of steps to train for
            callback_list: List of callbacks to use during training
            eval_env: Environment to use for evaluation
            eval_freq: Frequency of evaluation during training
            log_dir: Directory to save logs
            model_dir: Directory to save model checkpoints
        """
        callbacks = callback_list or []
        
        # Add evaluation callback if eval_env is provided
        if eval_env is not None:
            from stable_baselines3.common.callbacks import EvalCallback
            eval_callback = EvalCallback(
                eval_env=eval_env,
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                verbose=1
            )
            callbacks.append(eval_callback)
        
        # Add checkpoint callback if model_dir is provided
        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=max(total_timesteps // 10, 1000),
                save_path=model_dir,
                name_prefix="sac_trading"
            )
            callbacks.append(checkpoint_callback)
        
        # Set up TensorBoard logging if log_dir is provided
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.model.tensorboard_log = log_dir
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            log_interval=100
        )
        
        # Save the final model if model_dir is provided
        if model_dir is not None:
            self.model.save(os.path.join(model_dir, "sac_trading_final"))
            print(f"Model saved to {os.path.join(model_dir, 'sac_trading_final')}")
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Predict action for given observation.
        
        Args:
            observation: The observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: The action to take
            state: The updated state
        """
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load a model.
        
        Args:
            path: Path to load the model from
        """
        self.model = SAC.load(path, env=self.env, device=self.device)
        print(f"Model loaded from {path}")
    
    @classmethod
    def from_saved(
        cls,
        env,
        model_path: str,
        use_gpu: bool = True,
        device: Optional[str] = None,
        verbose: int = 1
    ) -> 'TradingAgent':
        """
        Create a TradingAgent from a saved model.
        
        Args:
            env: Trading environment
            model_path: Path to the saved model
            use_gpu: Whether to use GPU if available
            device: Device to use (auto, cpu, cuda)
            verbose: Verbosity level
            
        Returns:
            agent: TradingAgent instance with loaded model
        """
        return cls(
            env=env,
            model_path=model_path,
            use_gpu=use_gpu,
            device=device,
            verbose=verbose
        )