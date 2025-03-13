import os
import sys
import traceback
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Import environment and agent components
from env.core import StockTradingEnv
from data.processor import DataProcessor
from agent.model import TradingAgent
from agent.curriculum_callback import CurriculumLearningCallback
from agent.metrics_callback import MetricsLoggerCallback
from agent.early_stopping_callback import TradingEarlyStoppingCallback
from agent.visualization_callback import VisualizeTradesCallback
from observation.generator import ObservationGenerator
from utils.debug_utils import validate_environment, run_debug_episodes, plot_debug_results

# Function to load sample data
def load_stock_data(symbol):
    """
    Load stock data for a symbol from the data cache.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        DataFrame with OHLCV data or synthetic data if not found
    """
    data_dir = Path("data_cache/stocks")
    file_path = data_dir / f"{symbol}.csv"
    
    try:
        if file_path.exists():
            # Load from CSV file
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
            print(f"Loaded {len(df)} days of {symbol} data from {file_path}")
            return df
    except Exception as e:
        print(f"Error loading {symbol} data: {e}")
    
    print(f"Data not found for {symbol}, generating synthetic data")
    return generate_synthetic_data()

def generate_synthetic_data():
    """Generate synthetic stock data for testing."""
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
    
    print(f"Generated synthetic data with shape: {df.shape}")
    return df

def train_on_symbol(symbol, training_params):
    """
    Train a model on a specific symbol.
    
    Args:
        symbol: Stock symbol
        training_params: Dictionary of training parameters
        
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*30} Training on {symbol} {'='*30}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Load data
        stock_data = load_stock_data(symbol)
        
        if stock_data is None or len(stock_data) < 100:
            return {
                'symbol': symbol,
                'status': 'failed',
                'reason': 'insufficient_data'
            }
        
        print(f"Using {len(stock_data)} days of data for {symbol}")
        
        # Process data
        window_size = training_params.get('window_size', 20)
        data_processor = DataProcessor(stock_data, window_size=window_size)
        processed_data = data_processor.process_data()
        
        # Split into train and validation sets
        train_size = int(len(processed_data) * 0.8)
        train_data = processed_data.iloc[:train_size]
        val_data = processed_data.iloc[train_size:]
        
        print(f"Data split: {len(train_data)} training days, {len(val_data)} validation days")
        
        # Create output directories
        model_dir = Path(f"models/{symbol}_{timestamp}")
        log_dir = Path(f"logs/{symbol}_{timestamp}")
        viz_dir = Path(f"visualizations/{symbol}_{timestamp}")
        
        model_dir.mkdir(exist_ok=True, parents=True)
        log_dir.mkdir(exist_ok=True, parents=True)
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a consistent observation generator first
        observation_generator = ObservationGenerator(
            window_size=window_size,
            include_sentiment=False,
            use_pca=False,
            include_market_context=False
        )
        
        # Configure consistent feature flags
        observation_generator.feature_flags = {
            'price_data': True,
            'volume_data': True,
            'trend_indicators': True,
            'momentum_indicators': True,
            'volatility_indicators': True,
            'volume_indicators': True,
            'position_info': True,
            'account_status': True,
            'time_features': False,
            'market_context': False
        }
        
        # Calculate expected dimension
        expected_dimension = observation_generator._calculate_expected_dimension()
        print(f"Expected observation dimension: {expected_dimension}")
        
        # Create environments with the consistent observation generator
        train_env = StockTradingEnv(
            price_data=train_data,
            initial_capital=training_params.get('initial_capital', 100000.0),
            window_size=window_size,
            max_position_pct=training_params.get('max_position_pct', 0.25),
            transaction_cost_pct=training_params.get('transaction_cost_pct', 0.0015),
            curriculum_level=training_params.get('curriculum_level', 1),
            debug_mode=True,
            observation_generator=observation_generator  # Pass the same generator to both envs
        )
        
        # Create validation environment with the SAME observation generator
        val_env = StockTradingEnv(
            price_data=val_data,
            initial_capital=training_params.get('initial_capital', 100000.0),
            window_size=window_size,
            max_position_pct=training_params.get('max_position_pct', 0.25),
            transaction_cost_pct=training_params.get('transaction_cost_pct', 0.0015),
            curriculum_level=training_params.get('curriculum_level', 1),
            debug_mode=False,
            observation_generator=observation_generator  # Pass the same generator to both envs
        )
        
        # Validate observation space dimensions match
        train_obs_shape = train_env.observation_space.shape
        val_obs_shape = val_env.observation_space.shape
        
        print(f"Train observation shape: {train_obs_shape}")
        print(f"Validation observation shape: {val_obs_shape}")
        
        if train_obs_shape != val_obs_shape:
            print(f"ERROR: Observation space mismatch: train={train_obs_shape}, val={val_obs_shape}")
            return {
                'symbol': symbol,
                'status': 'failed',
                'reason': 'observation_mismatch',
                'train_shape': str(train_obs_shape),
                'val_shape': str(val_obs_shape)
            }
        
        # Validate environments with debug episodes
        print("\nRunning validation on training environment...")
        validate_environment(train_env)
        
        # Run a quick test episode to ensure stability
        print("\nRunning test episode...")
        obs, _ = train_env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Take a test step to verify consistency
        action = train_env.action_space.sample()
        next_obs, reward, term, trunc, info = train_env.step(action)
        print(f"Next observation shape: {next_obs.shape}")
        
        if obs.shape != next_obs.shape:
            print(f"ERROR: Observation shape changes after step: {obs.shape} -> {next_obs.shape}")
            return {
                'symbol': symbol,
                'status': 'failed',
                'reason': 'observation_instability',
                'initial_shape': str(obs.shape),
                'next_shape': str(next_obs.shape)
            }
            
        # Reset environments for training
        train_env.reset()
        val_env.reset()
        
        # Wrap environments with Monitor
        train_env = Monitor(train_env, str(log_dir / "train_monitor"))
        val_env = Monitor(val_env, str(log_dir / "val_monitor"))
        
        # Create agent
        agent = TradingAgent(
            env=train_env,
            learning_rate=training_params.get('learning_rate', 1e-4),
            buffer_size=training_params.get('buffer_size', 100000),
            batch_size=training_params.get('batch_size', 256),
            seed=training_params.get('seed', 42)
        )
        
        # Set up callbacks as a LIST, not a CallbackList object
        callbacks = []
        
        # 1. Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=max(training_params.get('total_timesteps', 100000) // 10, 1000),
            save_path=str(model_dir),
            name_prefix=f"{symbol}_model",
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # 2. Curriculum learning callback
        curriculum_callback = CurriculumLearningCallback(
            env=train_env,
            target_reward=training_params.get('curriculum_target_reward', 0.5),
            window_size=training_params.get('curriculum_window_size', 20),
            verbose=1
        )
        callbacks.append(curriculum_callback)
        
        # 3. Metrics logger callback
        metrics_callback = MetricsLoggerCallback(
            eval_env=val_env,
            log_path=str(log_dir / "metrics.csv"),
            log_freq=5000,
            verbose=1
        )
        callbacks.append(metrics_callback)
        
        # 4. Visualization callback
        viz_callback = VisualizeTradesCallback(
            eval_env=val_env,
            log_dir=str(viz_dir),
            plot_freq=10000,
            n_eval_episodes=1,
            verbose=1
        )
        callbacks.append(viz_callback)
        
        # 5. Early stopping callback
        early_stopping = TradingEarlyStoppingCallback(
            eval_env=val_env,
            log_dir=str(log_dir),
            n_eval_episodes=3,
            eval_freq=5000,
            warning_threshold=3,
            min_evals=5,
            check_win_rate=True,
            target_win_rate=0.4,
            check_sharpe=True,
            target_sharpe=0.5,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Train agent
        print(f"\nStarting training for {symbol} with {training_params.get('total_timesteps', 100000)} timesteps...")
        start_time = datetime.now()
        
        agent.train(
            total_timesteps=training_params.get('total_timesteps', 100000),
            callback_list=callbacks,  # Pass the list of callbacks directly
            eval_env=val_env,
            eval_freq=5000,
            log_dir=str(log_dir),
            model_dir=str(model_dir)
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save final model
        final_model_path = str(model_dir / f"{symbol}_final")
        agent.save(final_model_path)
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Model saved to {final_model_path}")
        
        # Return results
        return {
            'symbol': symbol,
            'status': 'success',
            'model_path': final_model_path,
            'log_dir': str(log_dir),
            'model_dir': str(model_dir),
            'viz_dir': str(viz_dir),
            'training_time': training_time,
            'timestamp': timestamp,
            'observation_shape': train_obs_shape
        }
    
    except Exception as e:
        print(f"Error training on {symbol}: {e}")
        traceback.print_exc()
        
        return {
            'symbol': symbol,
            'status': 'failed',
            'reason': 'exception',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def evaluate_model(symbol, model_path):
    """
    Evaluate a trained model.
    
    Args:
        symbol: Stock symbol
        model_path: Path to trained model
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*30} Evaluating {symbol} {'='*30}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/{symbol}_{timestamp}")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Load data
        stock_data = load_stock_data(symbol)
        
        if stock_data is None:
            return {
                'symbol': symbol,
                'status': 'failed',
                'reason': 'data_loading_error'
            }
        
        # Process data
        data_processor = DataProcessor(stock_data, window_size=20)
        processed_data = data_processor.process_data()
        
        # Use the last 20% for evaluation
        eval_size = int(len(processed_data) * 0.2)
        eval_data = processed_data.iloc[-eval_size:]
        
        if len(eval_data) < 20:
            return {
                'symbol': symbol,
                'status': 'failed',
                'reason': 'insufficient_eval_data'
            }
        
        print(f"Evaluating on {len(eval_data)} days of data")
        
        # Create the same observation generator as used in training
        observation_generator = ObservationGenerator(
            window_size=20,
            include_sentiment=False,
            use_pca=False,
            include_market_context=False
        )
        
        # Configure consistent feature flags
        observation_generator.feature_flags = {
            'price_data': True,
            'volume_data': True,
            'trend_indicators': True,
            'momentum_indicators': True,
            'volatility_indicators': True,
            'volume_indicators': True,
            'position_info': True,
            'account_status': True,
            'time_features': False,
            'market_context': False
        }
        
        # Create evaluation environment with the same observation generator
        eval_env = StockTradingEnv(
            price_data=eval_data,
            initial_capital=100000.0,
            window_size=20,
            max_position_pct=0.25,
            transaction_cost_pct=0.0015,
            observation_generator=observation_generator
        )
        
        # Load the trained agent
        agent = TradingAgent.from_saved(
            env=eval_env,
            model_path=model_path,
            verbose=1
        )
        
        # Run evaluation episode
        obs, _ = eval_env.reset()
        done = False
        
        # Tracking variables
        portfolio_values = []
        position_sizes = []
        actions_taken = []
        rewards = []
        trades = []
        step_count = 0
        
        print("Running evaluation episode...")
        
        while not done:
            # Get action from agent
            action, _ = agent.predict(obs, deterministic=True)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Update tracking variables
            done = terminated or truncated
            obs = next_obs
            step_count += 1
            
            # Record data
            portfolio_values.append(info['portfolio_value'])
            position_sizes.append(info['current_position'])
            rewards.append(reward)
            
            # Record action
            action_data = {
                'step': step_count,
                'action': action.tolist(),
                'reward': reward,
                'portfolio_value': info['portfolio_value'],
                'position': info['current_position']
            }
            actions_taken.append(action_data)
            
            # Record trade if one occurred
            if info.get('trade_executed', False):
                trade_data = {
                    'step': step_count,
                    'type': info.get('trade_type', ''),
                    'price': info.get('trade_price', 0),
                    'shares': info.get('trade_shares', 0)
                }
                trades.append(trade_data)
                
                print(f"Trade at step {step_count}: {trade_data['type']} {trade_data['shares']:.2f} shares at ${trade_data['price']:.2f}")
            
            # Print progress every 20 steps
            if step_count % 20 == 0:
                current_return = (info['portfolio_value']/100000 - 1) * 100
                print(f"Step {step_count}: Portfolio = ${info['portfolio_value']:.2f} ({current_return:.2f}%)")
        
        # Calculate evaluation metrics
        if len(portfolio_values) > 0:
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = (final_value / initial_value - 1) * 100
            
            # Calculate returns and volatility
            returns = []
            for i in range(1, len(portfolio_values)):
                ret = (portfolio_values[i] / portfolio_values[i-1]) - 1
                returns.append(ret)
            
            avg_return = np.mean(returns) * 100 if returns else 0
            volatility = np.std(returns) * 100 * np.sqrt(252) if returns else 0  # Annualized
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
            
            # Calculate max drawdown
            peak = portfolio_values[0]
            max_drawdown = 0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Count trades
            buy_trades = len([t for t in trades if t['type'] == 'buy'])
            sell_trades = len([t for t in trades if t['type'] == 'sell'])
            
            # Prepare result metrics
            metrics = {
                'symbol': symbol,
                'status': 'success',
                'model_path': model_path,
                'initial_value': float(initial_value),
                'final_value': float(final_value),
                'total_return': float(total_return),
                'avg_daily_return': float(avg_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_drawdown * 100),
                'num_steps': step_count,
                'num_buy_trades': buy_trades,
                'num_sell_trades': sell_trades
            }
            
            # Save evaluation results
            with open(results_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Save portfolio history
            pd.DataFrame({
                "step": range(len(portfolio_values)),
                "portfolio_value": portfolio_values,
                "position_size": position_sizes,
                "reward": rewards
            }).to_csv(results_dir / "portfolio_history.csv", index=False)
            
            # Save trades
            if trades:
                pd.DataFrame(trades).to_csv(results_dir / "trades.csv", index=False)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot portfolio value
            plt.subplot(2, 1, 1)
            plt.plot(portfolio_values)
            plt.title(f"{symbol} - Portfolio Value")
            plt.xlabel("Step")
            plt.ylabel("Value ($)")
            plt.grid(True)
            
            # Plot position sizes and trade points
            plt.subplot(2, 1, 2)
            plt.plot(position_sizes, label="Position Size")
            
            # Mark buy and sell points
            buy_indices = [t['step']-1 for t in trades if t['type'] == 'buy']
            sell_indices = [t['step']-1 for t in trades if t['type'] == 'sell']
            
            if buy_indices:
                buy_sizes = [position_sizes[min(i, len(position_sizes)-1)] for i in buy_indices]
                plt.scatter(buy_indices, buy_sizes, 
                           color='green', marker='^', s=100, label='Buy')
            
            if sell_indices:
                sell_sizes = [position_sizes[min(i, len(position_sizes)-1)] for i in sell_indices]
                plt.scatter(sell_indices, sell_sizes, 
                           color='red', marker='v', s=100, label='Sell')
            
            plt.title(f"{symbol} - Position Size")
            plt.xlabel("Step")
            plt.ylabel("Shares")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(results_dir / "evaluation_plot.png", dpi=100)
            plt.close()
            
            print("\nEvaluation results:")
            print(f"Total return: {total_return:.2f}%")
            print(f"Sharpe ratio: {sharpe:.2f}")
            print(f"Max drawdown: {max_drawdown*100:.2f}%")
            print(f"Number of trades: {buy_trades + sell_trades}")
            print(f"Results saved to {results_dir}")
            
            return metrics
        
        return {
            'symbol': symbol,
            'status': 'failed',
            'reason': 'no_evaluation_data'
        }
    
    except Exception as e:
        print(f"Error evaluating {symbol}: {e}")
        traceback.print_exc()
        
        return {
            'symbol': symbol,
            'status': 'failed',
            'reason': 'exception',
            'error': str(e)
        }

def main():
    # Create necessary directories
    for dir_path in ["models", "logs", "results", "visualizations", "data_cache/stocks"]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Stock Trading Reinforcement Learning Training Script")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define training parameters
    training_params = {
        'initial_capital': 100000.0,
        'window_size': 20,
        'max_position_pct': 0.25,
        'transaction_cost_pct': 0.0015,
        'curriculum_level': 1,
        'total_timesteps': 50000,  # Reduced for faster testing
        'learning_rate': 3e-5,
        'buffer_size': 100000,
        'batch_size': 256,
        'seed': 42,
        'curriculum_target_reward': 0.5,
        'curriculum_window_size': 20
    }
    
    # Define symbol(s) to train on
    symbols = ["AAPL_daily"]  # Use your actual symbol name
    
    # Train on each symbol
    training_results = {}
    evaluation_results = {}
    
    for symbol in symbols:
        # Train the model
        train_result = train_on_symbol(symbol, training_params)
        training_results[symbol] = train_result
        
        # Evaluate if training was successful
        if train_result['status'] == 'success':
            model_path = train_result['model_path']
            eval_result = evaluate_model(symbol, model_path)
            evaluation_results[symbol] = eval_result
    
    # Save overall results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/training_summary_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "training_results": training_results,
            "evaluation_results": evaluation_results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print("\nTraining Summary:")
    for symbol, result in training_results.items():
        status = result['status']
        if status == 'success':
            print(f"  {symbol}: SUCCESS - Model saved to {result['model_path']}")
        else:
            reason = result.get('reason', 'unknown')
            print(f"  {symbol}: FAILED - Reason: {reason}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main execution: {e}")
        traceback.print_exc()
        sys.exit(1)