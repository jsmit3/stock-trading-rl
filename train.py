import os
import sys
import traceback
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Import your environment and agent
from env.core import StockTradingEnv
from data.processor import DataProcessor
from agent.model import TradingAgent
from agent.callbacks import MetricsLoggerCallback, VisualizeTradesCallback
from utils.debug_utils import validate_environment, run_debug_episodes, plot_debug_results
from observation.generator import ObservationGenerator

# Function to load sample data
def load_sample_data():
    try:
        print("Generating synthetic data...")
        # Generate synthetic data for testing
        dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
        np.random.seed(42)
        
        # Start with a base price
        base_price = 100.0
        daily_returns = np.random.normal(0.0005, 0.015, len(dates))
        price_series = base_price * (1 + daily_returns).cumprod()
        
        # Generate OHLCV data
        df = pd.DataFrame({
            'open': price_series * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': price_series * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': price_series * (1 - np.random.uniform(0, 0.02, len(dates))),
            'close': price_series,
            'volume': np.random.lognormal(15, 0.5, len(dates))
        }, index=dates)
        
        print(f"Generated data with shape: {df.shape}")
        
        return df
    except Exception as e:
        print(f"Error generating sample data: {e}")
        traceback.print_exc()
        sys.exit(1)

def main():
    try:
        # Create output directories
        print("Creating output directories...")
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("visualizations", exist_ok=True)
        
        # Current time for log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load data
        print("Loading data...")
        price_data = load_sample_data()
        
        # Process data
        print("Processing data...")
        data_processor = DataProcessor(price_data, window_size=20)
        processed_data = data_processor.process_data()
        
        print(f"Processed data shape: {processed_data.shape}")
        
        # Create a simplified observation generator for consistent dimensions
        print("Creating observation generator...")
        observation_generator = ObservationGenerator(
            window_size=20,
            include_sentiment=False,
            use_pca=False,  # Disable PCA to ensure consistent dimensions
            include_market_context=False  # Disable market context for simplicity
        )
        
        # Modify the feature flags for a simpler, more consistent observation space
        observation_generator.feature_flags = {
            'price_data': True,
            'volume_data': True,
            'trend_indicators': False,
            'momentum_indicators': True,
            'volatility_indicators': True,
            'volume_indicators': False,
            'position_info': True,
            'account_status': True,
            'time_features': False,
            'market_context': False
        }
        
        # Create environment with the simplified observation generator
        print("Creating training environment...")
        env = StockTradingEnv(
            price_data=processed_data,
            initial_capital=100000.0,
            window_size=20,
            max_position_pct=0.25,
            curriculum_level=1,
            debug_mode=True
        )
        
        # Replace the environment's observation generator with our simplified one
        env.observation_generator = observation_generator
        
        # Get observation shape from a test reset
        obs, _ = env.reset()
        obs_shape = obs.shape
        print(f"Observation shape: {obs_shape}")
        
        # Create evaluation environment with the same observation generator
        print("Creating evaluation environment...")
        eval_env = StockTradingEnv(
            price_data=processed_data,
            initial_capital=100000.0,
            window_size=20,
            max_position_pct=0.25,
            curriculum_level=1,
            debug_mode=False
        )
        
        # Replace the evaluation environment's observation generator as well
        eval_env.observation_generator = observation_generator
        
        # Reset evaluation environment to ensure consistent observation space
        eval_obs, _ = eval_env.reset()
        eval_obs_shape = eval_obs.shape
        print(f"Evaluation observation shape: {eval_obs_shape}")
        
        if obs_shape != eval_obs_shape:
            raise ValueError(f"Observation shapes don't match: {obs_shape} vs {eval_obs_shape}")
        
        # Wrap environments in Monitor for logging
        env = Monitor(env, f"logs/train_monitor_{timestamp}")
        eval_env = Monitor(eval_env, f"logs/eval_monitor_{timestamp}")
        
        # Test a random step to ensure the environment works properly
        print("\n=== Testing Environment with Random Actions ===")
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action shape: {action.shape}")
        print(f"Next observation shape: {next_obs.shape}")
        print(f"Reward: {reward}")
        
        # Initialize agent
        print("\n=== Creating Agent ===")
        agent = TradingAgent(
            env=env,
            learning_rate=1e-4,
            buffer_size=100000,
            batch_size=256,
            verbose=1
        )
        
        # Set up callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=2000,
            save_path=f"models/checkpoints_{timestamp}/",
            name_prefix="trading_model",
            verbose=1
        )
        
        # Metrics logger callback
        metrics_callback = MetricsLoggerCallback(
            eval_env=eval_env,
            log_path=f"logs/metrics_{timestamp}.csv",
            log_freq=2000,
            verbose=1
        )
        
        # Visualization callback
        viz_callback = VisualizeTradesCallback(
            eval_env=eval_env,
            log_dir=f"visualizations/trades_{timestamp}/",
            plot_freq=5000,
            n_eval_episodes=1,
            verbose=1
        )
        
        # Combine callbacks
        callbacks = [checkpoint_callback, metrics_callback, viz_callback]
        
        # Run a small training session (5,000 steps for initial testing)
        print("\n=== Starting Training (5,000 steps) ===")
        agent.train(
            total_timesteps=5000,
            callback_list=callbacks,
            eval_env=eval_env,
            eval_freq=2000,
            log_dir=f"logs/tensorboard_{timestamp}",
            model_dir=f"models/{timestamp}"
        )
        
        # Save final model
        final_model_path = f"models/final_model_{timestamp}"
        agent.save(final_model_path)
        print(f"Model saved to {final_model_path}")
        
        # Test trained agent
        print("\n=== Testing Trained Agent ===")
        test_env = StockTradingEnv(
            price_data=processed_data,
            initial_capital=100000.0,
            window_size=20,
            max_position_pct=0.25,
            curriculum_level=1,
            debug_mode=True
        )
        
        # Use the same observation generator
        test_env.observation_generator = observation_generator
        
        # Reset environment
        obs, _ = test_env.reset()
        done = False
        cumulative_reward = 0
        step_count = 0
        portfolio_values = []
        
        # Run one episode
        print("Running one episode with trained agent...")
        while not done and step_count < 50:  # Limit to 50 steps for testing
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            cumulative_reward += reward
            step_count += 1
            
            portfolio_values.append(info['portfolio_value'])
            
            if step_count % 10 == 0:
                print(f"Step {step_count}: Portfolio = ${info['portfolio_value']:.2f}, Reward = {reward:.4f}")
            
            if info.get('trade_executed', False):
                trade_type = info.get('trade_type', 'unknown')
                price = info.get('trade_price', 0)
                shares = info.get('trade_shares', 0)
                print(f"Step {step_count}: {trade_type.capitalize()} {shares:.2f} shares at ${price:.2f}")
                
            if info.get('trade_completed', False):
                profit = info.get('trade_profit', 0)
                profit_pct = info.get('trade_profit_pct', 0)
                reason = info.get('trade_reason', 'unknown')
                print(f"Trade completed: ${profit:.2f} ({profit_pct:.2f}%), reason: {reason}")
        
        # Print final results
        print(f"\nEpisode completed after {step_count} steps")
        print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
        print(f"Return: {(info['portfolio_value']/100000 - 1) * 100:.2f}%")
        print(f"Cumulative reward: {cumulative_reward:.2f}")
        
        print("\nTraining and testing completed!")
        
    except Exception as e:
        print(f"Error in training process: {e}")
        traceback.print_exc()
        with open(f"logs/error_log_{timestamp}.txt", "w") as f:
            f.write(f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    print("Starting stock trading RL training script...")
    main()
    print("Script execution completed.")