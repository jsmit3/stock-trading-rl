"""
multi_stock_trainer.py

Trains a single reinforcement learning model across multiple stocks.

This script:
1. Loads and processes data for multiple stock symbols
2. Creates a multi-stock trading environment
3. Trains a single agent on all stocks
4. Evaluates the model on individual stocks
5. Saves training results and visualizations

Author: [Your Name]
Date: March 13, 2025
"""

import os
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

# Import stable-baselines3 components
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

# Import your modules
from env.multi_stock_env import MultiStockTradingEnv
from env.core import StockTradingEnv
from data.processor import DataProcessor
from agent.model import TradingAgent
from agent.curriculum_callback import CurriculumLearningCallback
from agent.metrics_callback import MetricsLoggerCallback
from agent.early_stopping_callback import TradingEarlyStoppingCallback
from agent.visualization_callback import VisualizeTradesCallback
from observation.generator import ObservationGenerator
from flexible_data_processor import get_daily_data


class MultiStockTrainer:
    """
    Trainer for reinforcement learning models across multiple stocks.
    
    This class handles:
    - Data acquisition from SQLite database or CSV files
    - Data processing and preparation
    - Training a single model across multiple stocks
    - Evaluating the model on individual stocks
    - Tracking results and visualizations
    """
    
    def __init__(
        self, 
        db_path: str = "data_cache/alpaca_data.db",
        timeframe: str = "1Day",
        output_dir: str = "output",
        training_params: Optional[Dict[str, Any]] = None,
        observation_generator: Optional[ObservationGenerator] = None,
        observation_dim: int = 325,
        symbol_feature_dim: int = 20
    ):
        """
        Initialize the multi-stock trainer.
        
        Args:
            db_path: Path to SQLite database with stock data
            timeframe: Timeframe to use (e.g., "1Day")
            output_dir: Output directory for models and results
            training_params: Parameters for training
            observation_generator: Custom observation generator
            observation_dim: Fixed dimension for observations
            symbol_feature_dim: Dimension for symbol embedding features
        """
        self.db_path = Path(db_path)
        self.timeframe = timeframe
        self.output_dir = Path(output_dir)
        self.observation_dim = observation_dim
        self.symbol_feature_dim = symbol_feature_dim
        
        # Create output directories
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        self.viz_dir = self.output_dir / "visualizations"
        self.results_dir = self.output_dir / "results"
        
        for directory in [self.models_dir, self.logs_dir, self.viz_dir, self.results_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Default training parameters
        self.training_params = training_params or {
            'initial_capital': 100000.0,
            'window_size': 20,
            'max_position_pct': 0.25,
            'transaction_cost_pct': 0.0015,
            'curriculum_level': 1,
            'learning_rate': 3e-5,
            'buffer_size': 200000,
            'batch_size': 256,
            'total_timesteps': 200000,
            'seed': 42,
            'observation_dim': observation_dim,
            'symbol_feature_dim': symbol_feature_dim
        }
        
        # Use provided observation generator or create default
        self.observation_generator = observation_generator
        if self.observation_generator is None:
            window_size = self.training_params.get('window_size', 20)
            obs_dim = self.observation_dim - self.symbol_feature_dim
            self.observation_generator = self._create_observation_generator(window_size, obs_dim)
        
        # Initialize containers for data and results
        self.stock_data = {}
        self.processed_data = {}
        self.training_results = {}
        self.evaluation_results = {}
    
    def _create_observation_generator(self, window_size=20, fixed_dim=305):
        """
        Create a consistently configured observation generator with fixed dimension.
        
        Args:
            window_size: Lookback window size for observations
            fixed_dim: Fixed dimension for observations (excluding symbol features)
            
        Returns:
            ObservationGenerator instance
        """
        observation_generator = ObservationGenerator(
            window_size=window_size,
            include_sentiment=False,
            use_pca=False,
            include_market_context=False,
            fixed_dim=fixed_dim
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
        
        print(f"Observation generator created with dimension: {observation_generator.observation_dim}")
        return observation_generator
    
    def load_symbols(self, symbols_file: Optional[str] = None) -> List[str]:
        """
        Load stock symbols from file or use defaults.
        
        Args:
            symbols_file: Path to file with symbols (one per line)
            
        Returns:
            List of stock symbols
        """
        if symbols_file and os.path.exists(symbols_file):
            with open(symbols_file, 'r') as f:
                symbols = [line.strip() for line in f.readlines()]
                return [s for s in symbols if s]  # Filter out empty lines
        
        # Default to top 10 stocks if file not provided
        return [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 
            'BRK-B', 'JPM', 'JNJ', 'V', 'PG'
        ]
    
    def load_stock_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load stock data for multiple symbols.
        
        Args:
            symbols: List of stock symbols to load
            
        Returns:
            Dictionary mapping symbols to DataFrames with stock data
        """
        loaded_data = {}
        
        for symbol in symbols:
            print(f"Loading data for {symbol}...")
            
            # Try loading from CSV file first
            csv_path = Path(f"data_cache/stocks/{symbol}.csv")
            
            if csv_path.exists():
                try:
                    # Load from CSV
                    df = pd.read_csv(csv_path)
                    
                    # Convert timestamp to datetime and set as index
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    
                    print(f"Loaded {len(df)} days of {symbol} data from CSV")
                    loaded_data[symbol] = df
                    continue
                except Exception as e:
                    print(f"Error loading {symbol} from CSV: {e}")
            
            # If CSV loading failed, try from database
            try:
                db_data = get_daily_data(symbol, timeframe=self.timeframe)
                
                if db_data is not None and len(db_data) >= 252:  # Minimum 1 year of data
                    print(f"Loaded {len(db_data)} days of {symbol} data from database")
                    loaded_data[symbol] = db_data
                else:
                    print(f"Insufficient data for {symbol}, skipping")
            except Exception as e:
                print(f"Error loading {symbol} from database: {e}")
        
        print(f"Successfully loaded data for {len(loaded_data)}/{len(symbols)} symbols")
        return loaded_data
    
    def process_stock_data(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process raw stock data for all symbols.
        
        Args:
            stock_data: Dictionary mapping symbols to raw stock DataFrames
            
        Returns:
            Dictionary mapping symbols to processed DataFrames
        """
        processed_data = {}
        window_size = self.training_params.get('window_size', 20)
        
        for symbol, data in stock_data.items():
            print(f"Processing data for {symbol}...")
            
            try:
                # Process data
                processor = DataProcessor(data, window_size=window_size)
                processed = processor.process_data()
                
                # Ensure we have sufficient data after processing
                if len(processed) < window_size + 50:
                    print(f"Insufficient processed data for {symbol}, skipping")
                    continue
                
                processed_data[symbol] = processed
                print(f"Processed {len(processed)} days of {symbol} data")
            except Exception as e:
                print(f"Error processing {symbol} data: {e}")
        
        return processed_data
    
    def create_multi_stock_environment(
        self, 
        processed_data: Dict[str, pd.DataFrame],
        for_training: bool = True
    ) -> MultiStockTradingEnv:
        """
        Create a multi-stock trading environment.
        
        Args:
            processed_data: Dictionary mapping symbols to processed DataFrames
            for_training: Whether the environment is for training or evaluation
            
        Returns:
            MultiStockTradingEnv instance
        """
        print(f"Creating multi-stock environment with {len(processed_data)} symbols")
        
        # Create the multi-stock environment
        env = MultiStockTradingEnv(
            stock_data=processed_data,
            initial_capital=self.training_params.get('initial_capital', 100000.0),
            max_holding_period=self.training_params.get('max_holding_period', 20),
            transaction_cost_pct=self.training_params.get('transaction_cost_pct', 0.0015),
            window_size=self.training_params.get('window_size', 20),
            reward_scaling=self.training_params.get('reward_scaling', 2.0),
            risk_aversion=self.training_params.get('risk_aversion', 0.5),
            drawdown_penalty=self.training_params.get('drawdown_penalty', 1.0),
            opportunity_cost=self.training_params.get('opportunity_cost', 0.05),
            drawdown_threshold=self.training_params.get('drawdown_threshold', 0.05),
            max_drawdown_pct=self.training_params.get('max_drawdown_pct', 0.25),
            include_sentiment=False,
            max_position_pct=self.training_params.get('max_position_pct', 0.25),
            min_position_pct=self.training_params.get('min_position_pct', 0.05),
            curriculum_level=self.training_params.get('curriculum_level', 1),
            debug_mode=False,  # Set to False to reduce output verbosity
            min_episode_length=20,
            observation_generator=self.observation_generator,
            observation_dim=self.observation_dim,
            symbol_feature_dim=self.symbol_feature_dim
        )
        
        # Test the environment
        obs, info = env.reset()
        print(f"Environment created successfully with observation shape: {obs.shape}")
        
        return env
    
    def create_single_stock_environment(
        self, 
        symbol: str, 
        data: pd.DataFrame,
        for_training: bool = False
    ) -> StockTradingEnv:
        """
        Create a single-stock trading environment for evaluation.
        
        Args:
            symbol: Stock symbol
            data: Processed stock data
            for_training: Whether the environment is for training
            
        Returns:
            StockTradingEnv instance
        """
        print(f"Creating evaluation environment for {symbol}")
        
        # Create the single-stock environment
        env = StockTradingEnv(
            price_data=data,
            initial_capital=self.training_params.get('initial_capital', 100000.0),
            max_holding_period=self.training_params.get('max_holding_period', 20),
            transaction_cost_pct=self.training_params.get('transaction_cost_pct', 0.0015),
            window_size=self.training_params.get('window_size', 20),
            reward_scaling=self.training_params.get('reward_scaling', 2.0),
            risk_aversion=self.training_params.get('risk_aversion', 0.5),
            drawdown_penalty=self.training_params.get('drawdown_penalty', 1.0),
            opportunity_cost=self.training_params.get('opportunity_cost', 0.05),
            drawdown_threshold=self.training_params.get('drawdown_threshold', 0.05),
            max_drawdown_pct=self.training_params.get('max_drawdown_pct', 0.25),
            include_sentiment=False,
            max_position_pct=self.training_params.get('max_position_pct', 0.25),
            min_position_pct=self.training_params.get('min_position_pct', 0.05),
            curriculum_level=self.training_params.get('curriculum_level', 1),
            debug_mode=for_training,
            min_episode_length=20,
            observation_generator=self.observation_generator,
            observation_dim=self.observation_dim,  # Add missing observation dimension
            symbol_feature_dim=self.symbol_feature_dim  # Add symbol feature dimension
        )
        
        return env
    
    def train_model(self, env: MultiStockTradingEnv) -> Tuple[TradingAgent, Dict[str, Any]]:
        """
        Train a model on the multi-stock environment.
        
        Args:
            env: Multi-stock trading environment
            
        Returns:
            Tuple of (trained agent, training results dictionary)
        """
        print("\n" + "="*20 + " Training multi-stock model " + "="*20)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories for this run
        model_dir = self.models_dir / f"multi_stock_{timestamp}"
        log_dir = self.logs_dir / f"multi_stock_{timestamp}"
        viz_dir = self.viz_dir / f"multi_stock_{timestamp}"
        
        for directory in [model_dir, log_dir, viz_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Wrap environment with Monitor
        monitor_env = Monitor(env, str(log_dir / "train_monitor"))
        
        # Create agent
        agent = TradingAgent(
            env=monitor_env,
            learning_rate=self.training_params.get('learning_rate', 3e-5),
            buffer_size=self.training_params.get('buffer_size', 200000),
            batch_size=self.training_params.get('batch_size', 256),
            seed=self.training_params.get('seed', 42)
        )
        
        # Set up callbacks
        callbacks = []
        
        # 1. Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=max(self.training_params.get('total_timesteps', 200000) // 10, 1000),
            save_path=str(model_dir),
            name_prefix="multi_stock_model",
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # 2. Curriculum learning callback
        curriculum_callback = CurriculumLearningCallback(
            env=monitor_env,
            target_reward=0.5,
            window_size=20,
            verbose=1
        )
        callbacks.append(curriculum_callback)
        
        # 3. Visualization callback
        viz_callback = VisualizeTradesCallback(
            eval_env=env,
            log_dir=str(viz_dir),
            plot_freq=10000,
            n_eval_episodes=1,
            verbose=1
        )
        callbacks.append(viz_callback)
        
        # Train agent
        start_time = datetime.now()
        print(f"\nStarting model training with {self.training_params.get('total_timesteps', 200000)} timesteps...")
        
        try:
            agent.train(
                total_timesteps=self.training_params.get('total_timesteps', 200000),
                callback_list=callbacks,  # Pass callbacks as a list
                log_dir=str(log_dir),
                model_dir=str(model_dir)
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Save final model
            final_model_path = str(model_dir / "multi_stock_final")
            agent.save(final_model_path)
            
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Model saved to {final_model_path}")
            
            # Return results
            return agent, {
                'status': 'success',
                'model_path': final_model_path,
                'log_dir': str(log_dir),
                'model_dir': str(model_dir),
                'viz_dir': str(viz_dir),
                'training_time': training_time,
                'timestamp': timestamp
            }
            
        except Exception as e:
            print(f"Error in model training: {e}")
            traceback.print_exc()
            
            return None, {
                'status': 'failed',
                'reason': 'exception',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def evaluate_on_symbol(
        self, 
        agent: TradingAgent, 
        symbol: str, 
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model on a specific symbol.
        
        Args:
            agent: Trained TradingAgent
            symbol: Stock symbol to evaluate on
            data: Processed stock data for the symbol
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*20} Evaluating on {symbol} {'='*20}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = self.results_dir / f"{symbol}_{timestamp}"
        eval_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Split data into training and test sets
            test_size = int(len(data) * 0.2)
            test_data = data.iloc[-test_size:]
            
            if len(test_data) < 20:
                return {"status": "failed", "reason": "insufficient_eval_data"}
            
            print(f"Evaluating on {len(test_data)} days of {symbol} data")
            
            # Create evaluation environment
            eval_env = self.create_single_stock_environment(
                symbol=symbol,
                data=test_data,
                for_training=False
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
            
            print(f"Running evaluation episode for {symbol}...")
            
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
                    
                    if step_count % 50 == 0:  # Only show trades occasionally
                        print(f"Sample trade: {trade_data['type']} {trade_data['shares']:.2f} shares at ${trade_data['price']:.2f}")
                
                # Print progress less frequently
                if step_count % 100 == 0:
                    current_return = (info['portfolio_value']/self.training_params['initial_capital'] - 1) * 100
                    print(f"Evaluation step {step_count}: Portfolio = ${info['portfolio_value']:.2f} ({current_return:.2f}%)")
            
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
                with open(eval_dir / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                
                # Save portfolio history
                pd.DataFrame({
                    "step": range(len(portfolio_values)),
                    "portfolio_value": portfolio_values,
                    "position_size": position_sizes,
                    "reward": rewards
                }).to_csv(eval_dir / "portfolio_history.csv", index=False)
                
                # Save trades
                if trades:
                    pd.DataFrame(trades).to_csv(eval_dir / "trades.csv", index=False)
                
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
                plt.savefig(eval_dir / "evaluation_plot.png", dpi=100)
                plt.close()
                
                print(f"\nEvaluation results for {symbol}:")
                print(f"Total return: {total_return:.2f}%")
                print(f"Sharpe ratio: {sharpe:.2f}")
                print(f"Max drawdown: {max_drawdown*100:.2f}%")
                print(f"Number of trades: {buy_trades + sell_trades}")
                print(f"Results saved to {eval_dir}")
                
                return metrics
        
        except Exception as e:
            print(f"Error evaluating on {symbol}: {e}")
            traceback.print_exc()
            
            return {
                'symbol': symbol,
                'status': 'failed',
                'reason': 'exception',
                'error': str(e)
            }
        
        return {
            'symbol': symbol,
            'status': 'failed',
            'reason': 'no_evaluation_data'
        }
    
    def run(self, symbols: Optional[List[str]] = None, symbols_file: Optional[str] = None) -> Dict:
        """
        Run the full training and evaluation process.
        
        Args:
            symbols: List of symbols to use
            symbols_file: Path to file with symbols
            
        Returns:
            Dictionary with training and evaluation results
        """
        # Get symbols to use
        symbols_to_use = symbols or self.load_symbols(symbols_file)
        print(f"Working with {len(symbols_to_use)} symbols: {', '.join(symbols_to_use[:min(5, len(symbols_to_use))])}")
        
        # Load and process stock data
        self.stock_data = self.load_stock_data(symbols_to_use)
        self.processed_data = self.process_stock_data(self.stock_data)
        
        # Only proceed if we have data for at least 2 symbols
        if len(self.processed_data) < 2:
            print("Insufficient data: Need at least 2 symbols with good data")
            return {
                "training_results": {"status": "failed", "reason": "insufficient_data"},
                "evaluation_results": {}
            }
        
        # Create multi-stock environment
        env = self.create_multi_stock_environment(self.processed_data)
        
        # Train model
        agent, train_result = self.train_model(env)
        self.training_results = train_result
        
        # Evaluate model on each symbol
        if agent is not None and train_result.get('status') == 'success':
            for symbol, data in self.processed_data.items():
                eval_result = self.evaluate_on_symbol(agent, symbol, data)
                self.evaluation_results[symbol] = eval_result
        
        # Save overall results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"multi_stock_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "training_params": self.training_params,
                "training_results": self.training_results,
                "evaluation_results": self.evaluation_results
            }, f, indent=2, default=str)
        
        print(f"Results saved to {results_file}")
        
        # Print summary of best symbols
        successful_evals = [
            (symbol, result) 
            for symbol, result in self.evaluation_results.items() 
            if result.get('status') == 'success'
        ]
        
        if successful_evals:
            # Sort by Sharpe ratio
            sorted_results = sorted(
                successful_evals,
                key=lambda x: x[1].get('sharpe_ratio', -999),
                reverse=True
            )
            
            print("\nPerformance across symbols (ranked by Sharpe ratio):")
            for i, (symbol, result) in enumerate(sorted_results, 1):
                sharpe = result.get('sharpe_ratio', 0)
                returns = result.get('total_return', 0)
                drawdown = result.get('max_drawdown', 0)
                trades = result.get('num_buy_trades', 0) + result.get('num_sell_trades', 0)
                
                print(f"{i}. {symbol}: Sharpe={sharpe:.2f}, Return={returns:.2f}%, " 
                    f"Drawdown={drawdown:.2f}%, Trades={trades}")
        
        return {
            "training_results": self.training_results,
            "evaluation_results": self.evaluation_results
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a multi-stock trading agent")
    parser.add_argument("--db-path", type=str, default="data_cache/alpaca_data.db", help="Path to database")
    parser.add_argument("--timeframe", type=str, default="1Day", help="Timeframe to use")
    parser.add_argument("--output-dir", type=str, default="multi_stock_output", help="Output directory")
    parser.add_argument("--symbols-file", type=str, help="Path to file with symbols")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to use")
    parser.add_argument("--timesteps", type=int, default=200000, help="Number of timesteps for training")
    parser.add_argument("--obs-dim", type=int, default=325, help="Observation dimension")
    parser.add_argument("--symbol-dim", type=int, default=20, help="Symbol embedding dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MultiStockTrainer(
        db_path=args.db_path,
        timeframe=args.timeframe,
        output_dir=args.output_dir,
        training_params={
            'total_timesteps': args.timesteps,
            'seed': args.seed,
            'observation_dim': args.obs_dim,
            'symbol_feature_dim': args.symbol_dim
        },
        observation_dim=args.obs_dim,
        symbol_feature_dim=args.symbol_dim
    )
    
    # Run training and evaluation
    symbols = args.symbols if args.symbols else None
    results = trainer.run(symbols=symbols, symbols_file=args.symbols_file)
    
    # Print final status
    if results["training_results"].get("status") == "success":
        print("\nTraining completed successfully!")
        
        success_count = sum(1 for r in results["evaluation_results"].values() 
                          if r.get("status") == "success")
        
        print(f"Successfully evaluated on {success_count}/{len(results['evaluation_results'])} symbols")
    else:
        print("\nTraining failed:", results["training_results"].get("reason", "unknown reason"))