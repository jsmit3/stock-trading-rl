import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import traceback

# Import your existing components
from env.core import StockTradingEnv
from data.processor import DataProcessor
from agent.model import TradingAgent
from agent.callbacks import (
    CurriculumLearningCallback, 
    MetricsLoggerCallback, 
    VisualizeTradesCallback, 
    TradingEarlyStoppingCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

# Define paths
DATA_DIR = Path("data_cache/stocks")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")
RESULTS_DIR = Path("results")
VISUALIZATIONS_DIR = Path("visualizations")

def setup_directories():
    """Create necessary directories"""
    for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR, VISUALIZATIONS_DIR]:
        directory.mkdir(exist_ok=True, parents=True)
    print(f"Created directories")

def get_available_symbols():
    """Get list of available processed symbols"""
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return []
    
    # Look for CSV files in the data directory
    csv_files = list(DATA_DIR.glob("*.csv"))
    
    # Extract symbol names from filenames
    symbols = [file.stem for file in csv_files]
    
    return symbols

def load_stock_data(symbol):
    """
    Load processed stock data for a symbol.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        DataFrame with OHLCV data or None if not found
    """
    file_path = DATA_DIR / f"{symbol}.csv"
    
    if not file_path.exists():
        print(f"Data file not found: {file_path}")
        return None
    
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Check for timestamp column and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                print(f"Missing required column: {col}")
                return None
        
        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        return df
    
    except Exception as e:
        print(f"Error loading data for {symbol}: {e}")
        return None

def filter_symbols_by_data_quality(symbols, min_days=252, min_price=5.0):
    """
    Filter symbols based on data quality criteria.
    
    Args:
        symbols: List of symbols to filter
        min_days: Minimum number of trading days required
        min_price: Minimum average price required
        
    Returns:
        List of filtered symbols
    """
    print(f"Filtering {len(symbols)} symbols by data quality...")
    filtered_symbols = []
    
    for symbol in symbols:
        df = load_stock_data(symbol)
        
        if df is None:
            continue
        
        # Check for minimum number of days
        if len(df) < min_days:
            print(f"Skipping {symbol}: Insufficient data ({len(df)} days < {min_days})")
            continue
        
        # Check for minimum price
        avg_price = df['close'].mean()
        if avg_price < min_price:
            print(f"Skipping {symbol}: Low average price (${avg_price:.2f} < ${min_price:.2f})")
            continue
        
        # Check for abnormal prices
        if df['close'].max() / df['close'].min() > 100:
            print(f"Skipping {symbol}: Abnormal price range (max/min > 100)")
            continue
        
        filtered_symbols.append(symbol)
    
    print(f"Filtered to {len(filtered_symbols)} symbols")
    return filtered_symbols

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
        
        if stock_data is None:
            return {
                'symbol': symbol,
                'status': 'failed',
                'reason': 'data_loading_error'
            }
        
        print(f"Loaded {len(stock_data)} days of data for {symbol}")
        
        # Process data with DataProcessor
        data_processor = DataProcessor(
            stock_data, 
            window_size=training_params.get('window_size', 20)
        )
        processed_data = data_processor.process_data()
        
        # Split into train and validation sets
        train_size = int(len(processed_data) * 0.8)
        train_data = processed_data.iloc[:train_size]
        val_data = processed_data.iloc[train_size:]
        
        print(f"Data split: {len(train_data)} training days, {len(val_data)} validation days")
        
        # Create output directories
        model_dir = MODELS_DIR / f"{symbol}_{timestamp}"
        log_dir = LOGS_DIR / f"{symbol}_{timestamp}"
        viz_dir = VISUALIZATIONS_DIR / f"{symbol}_{timestamp}"
        
        model_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        viz_dir.mkdir(exist_ok=True)
        
        # Create environments
        train_env = StockTradingEnv(
            price_data=train_data,
            initial_capital=training_params.get('initial_capital', 100000.0),
            window_size=training_params.get('window_size', 20),
            max_position_pct=training_params.get('max_position_pct', 0.25),
            transaction_cost_pct=training_params.get('transaction_cost_pct', 0.0015),
            curriculum_level=training_params.get('curriculum_level', 1),
        )
        
        val_env = StockTradingEnv(
            price_data=val_data,
            initial_capital=training_params.get('initial_capital', 100000.0),
            window_size=training_params.get('window_size', 20),
            max_position_pct=training_params.get('max_position_pct', 0.25),
            transaction_cost_pct=training_params.get('transaction_cost_pct', 0.0015),
            curriculum_level=training_params.get('curriculum_level', 1),
        )
        
        # Wrap environments with Monitor
        train_env = Monitor(train_env, str(log_dir / "train_monitor"))
        val_env = Monitor(val_env, str(log_dir / "val_monitor"))
        
        # Create agent
        agent = TradingAgent(
            env=train_env,
            model_path=None,  # Start fresh
            learning_rate=training_params.get('learning_rate', 1e-4),
            buffer_size=training_params.get('buffer_size', 100000),
            batch_size=training_params.get('batch_size', 256),
            seed=training_params.get('seed', 42)
        )
        
        # Create callbacks
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
        
        # Train agent - THE KEY FIX: Pass callback_list directly, not CallbackList object
        print(f"Starting training for {symbol} with {training_params.get('total_timesteps', 100000)} timesteps...")
        start_time = time.time()
        
        agent.train(
            total_timesteps=training_params.get('total_timesteps', 100000),
            callback_list=callbacks,  # Pass the list directly, not a CallbackList object
            eval_env=val_env,
            eval_freq=5000,
            log_dir=str(log_dir),
            model_dir=str(model_dir)
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
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
            'timestamp': timestamp
        }
    
    except Exception as e:
        print(f"Error training on {symbol}: {e}")
        traceback.print_exc()
        
        return {
            'symbol': symbol,
            'status': 'failed',
            'reason': 'exception',
            'error': str(e)
        }

def evaluate_model(symbol, model_path):
    """
    Evaluate a trained model on validation data.
    
    Args:
        symbol: Stock symbol
        model_path: Path to trained model
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*30} Evaluating {symbol} {'='*30}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_DIR / f"{symbol}_{timestamp}"
    result_dir.mkdir(exist_ok=True)
    
    try:
        # Load data
        stock_data = load_stock_data(symbol)
        
        if stock_data is None:
            return {
                'symbol': symbol,
                'status': 'failed',
                'reason': 'data_loading_error'
            }
        
        # Process data with DataProcessor
        data_processor = DataProcessor(
            stock_data, 
            window_size=20  # Default window size
        )
        processed_data = data_processor.process_data()
        
        # Use the last 20% for evaluation
        eval_size = int(len(processed_data) * 0.2)
        eval_data = processed_data.iloc[-eval_size:]
        
        print(f"Evaluating on {len(eval_data)} days of data")
        
        # Create evaluation environment
        eval_env = StockTradingEnv(
            price_data=eval_data,
            initial_capital=100000.0,
            window_size=20,
            max_position_pct=0.25,
            transaction_cost_pct=0.0015
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
        
        print("Running evaluation episode...")
        
        step = 0
        while not done:
            # Get action from agent
            action, _ = agent.predict(obs, deterministic=True)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            
            # Update tracking variables
            done = terminated or truncated
            obs = next_obs
            step += 1
            
            # Record data
            portfolio_values.append(info['portfolio_value'])
            position_sizes.append(info['current_position'])
            rewards.append(reward)
            
            # Record action
            action_data = {
                'step': step,
                'action': action.tolist(),
                'reward': reward,
                'portfolio_value': info['portfolio_value'],
                'position': info['current_position']
            }
            actions_taken.append(action_data)
            
            # Record trade if one occurred
            if info.get('trade_executed', False):
                trade_data = {
                    'step': step,
                    'type': info.get('trade_type', ''),
                    'price': info.get('trade_price', 0),
                    'shares': info.get('trade_shares', 0),
                    'value': info.get('trade_value', 0) if info.get('trade_type') == 'sell' else info.get('trade_cost', 0),
                    'reason': info.get('trade_reason', '')
                }
                trades.append(trade_data)
                
                if step % 20 == 0:
                    print(f"Step {step}: {trade_data['type']} {trade_data['shares']:.2f} shares at ${trade_data['price']:.2f}")
        
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
            
            if len(returns) > 0:
                avg_return = np.mean(returns) * 100
                volatility = np.std(returns) * 100 * np.sqrt(252)  # Annualized
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                avg_return = 0
                volatility = 0
                sharpe = 0
            
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
                'num_steps': step,
                'num_buy_trades': buy_trades,
                'num_sell_trades': sell_trades
            }
            
            # Save evaluation results
            
            # Save metrics
            with open(result_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Save portfolio values history
            pd.DataFrame({
                "step": range(len(portfolio_values)),
                "portfolio_value": portfolio_values,
                "position_size": position_sizes,
                "reward": rewards
            }).to_csv(result_dir / "portfolio_history.csv", index=False)
            
            # Save trades
            if trades:
                pd.DataFrame(trades).to_csv(result_dir / "trades.csv", index=False)
            
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
            plt.savefig(result_dir / "evaluation_plot.png", dpi=100)
            plt.close()
            
            print("\nEvaluation results:")
            print(f"Total return: {total_return:.2f}%")
            print(f"Sharpe ratio: {sharpe:.2f}")
            print(f"Max drawdown: {max_drawdown*100:.2f}%")
            print(f"Number of trades: {buy_trades + sell_trades}")
            print(f"Results saved to {result_dir}")
            
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

def select_best_models(results, top_n=5):
    """
    Select the best models based on Sharpe ratio.
    
    Args:
        results: Dictionary mapping symbols to evaluation results
        top_n: Number of top models to select
        
    Returns:
        List of top model info dictionaries
    """
    # Filter successful evaluations
    successful = [r for r in results.values() if r.get('status') == 'success']
    
    # Sort by Sharpe ratio (descending)
    sorted_results = sorted(
        successful,
        key=lambda x: x.get('sharpe_ratio', -999),
        reverse=True
    )
    
    # Return top N
    return sorted_results[:top_n]

def save_training_summary(training_results, evaluation_results):
    """
    Save summary of training and evaluation results.
    
    Args:
        training_results: Dictionary mapping symbols to training results
        evaluation_results: Dictionary mapping symbols to evaluation results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = {
        'timestamp': timestamp,
        'training_results': training_results,
        'evaluation_results': evaluation_results
    }
    
    # Create summary file
    summary_path = RESULTS_DIR / f"training_summary_{timestamp}.json"
    
    # Save summary
    with open(summary_path, 'w') as f:
        # Convert values that aren't JSON serializable to strings
        serializable_summary = json.dumps(summary, default=str)
        f.write(serializable_summary)
    
    print(f"Training summary saved to {summary_path}")
    
    # Also save a CSV report of evaluation results
    report_data = []
    
    for symbol, result in evaluation_results.items():
        if result.get('status') == 'success':
            report_data.append({
                'symbol': symbol,
                'total_return': result.get('total_return', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'trades': result.get('num_buy_trades', 0) + result.get('num_sell_trades', 0)
            })
    
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('sharpe_ratio', ascending=False)
        
        report_path = RESULTS_DIR / f"evaluation_report_{timestamp}.csv"
        report_df.to_csv(report_path, index=False)
        
        print(f"Evaluation report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Train RL Agents with Alpaca Data")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to train on")
    parser.add_argument("--top", type=int, default=10, help="Number of top symbols by data quality to use")
    parser.add_argument("--timesteps", type=int, default=50000, help="Training timesteps")
    parser.add_argument("--evaluate", action="store_true", help="Only evaluate existing models")
    parser.add_argument("--model", type=str, help="Path to model for evaluation (with --evaluate)")
    
    args = parser.parse_args()
    
    print("RL Trading Agent Training with Alpaca Data")
    
    # Setup directories
    setup_directories()
    
    # Get available symbols
    all_symbols = get_available_symbols()
    print(f"Found {len(all_symbols)} processed symbols")
    
    if len(all_symbols) == 0:
        print("No processed data found. Please run the data processor first.")
        return
    
    # Determine which symbols to use
    if args.symbols:
        symbols_to_use = args.symbols
        print(f"Using specified symbols: {', '.join(symbols_to_use)}")
    else:
        # Filter by data quality
        filtered_symbols = filter_symbols_by_data_quality(all_symbols)
        
        # Take the top N by data quality
        symbols_to_use = filtered_symbols[:args.top]
        print(f"Using top {len(symbols_to_use)} symbols by data quality")
    
    # Define training parameters
    training_params = {
        'initial_capital': 100000.0,
        'window_size': 20,
        'max_position_pct': 0.25,
        'transaction_cost_pct': 0.0015,
        'curriculum_level': 1,
        'total_timesteps': args.timesteps,
        'learning_rate': 3e-5,
        'buffer_size': 100000,
        'batch_size': 256,
        'seed': 42
    }
    
    # Train or evaluate based on arguments
    if args.evaluate:
        if args.model and args.symbols and len(args.symbols) == 1:
            # Evaluate a specific model
            symbol = args.symbols[0]
            print(f"Evaluating model for {symbol}: {args.model}")
            result = evaluate_model(symbol, args.model)
            print(f"Evaluation {'successful' if result.get('status') == 'success' else 'failed'}")
        else:
            print("When using --evaluate, please specify a single symbol with --symbols and a model path with --model")
            return
    else:
        # Train models for selected symbols
        training_results = {}
        evaluation_results = {}
        
        for symbol in symbols_to_use:
            # Train model
            train_result = train_on_symbol(symbol, training_params)
            training_results[symbol] = train_result
            
            # Evaluate if training was successful
            if train_result.get('status') == 'success':
                model_path = train_result.get('model_path')
                eval_result = evaluate_model(symbol, model_path)
                evaluation_results[symbol] = eval_result
        
        # Find best models
        best_models = select_best_models(evaluation_results)
        
        print("\nTraining and evaluation completed!")
        print(f"Trained and evaluated {len(symbols_to_use)} symbols")
        
        # Print top models
        if best_models:
            print("\nTop models by Sharpe ratio:")
            for i, model in enumerate(best_models, 1):
                symbol = model.get('symbol', 'unknown')
                sharpe = model.get('sharpe_ratio', 0)
                returns = model.get('total_return', 0)
                drawdown = model.get('max_drawdown', 0)
                
                print(f"{i}. {symbol}: Sharpe={sharpe:.2f}, Return={returns:.2f}%, Drawdown={drawdown:.2f}%")
        
        # Save summary
        save_training_summary(training_results, evaluation_results)

if __name__ == "__main__":
    main()