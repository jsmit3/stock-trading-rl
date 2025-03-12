import os
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import json
import traceback
from typing import Dict, List, Optional, Any, Tuple

# Import stable_baselines3 components
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

# Import your modules
from env.core import StockTradingEnv
from data.processor import DataProcessor
from agent.model import TradingAgent
from agent.curriculum_callback import CurriculumLearningCallback
from agent.metrics_callback import MetricsLoggerCallback
from agent.early_stopping_callback import TradingEarlyStoppingCallback
from agent.visualization_callback import VisualizeTradesCallback
from flexible_data_processor import get_daily_data


class SP100Trainer:
    """
    Trainer class for reinforcement learning models on S&P 100 stocks.
    
    This class handles:
    - Data acquisition from SQLite database
    - Data processing
    - Training and evaluation of RL agents
    - Results tracking and visualization
    """
    
    def __init__(
        self, 
        db_path: str = "data_cache/alpaca_data.db",
        timeframe: str = "1Day",
        output_dir: str = "output",
        training_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the SP100 Trainer.
        
        Args:
            db_path: Path to SQLite database
            timeframe: Timeframe to use (e.g., "1Day")
            output_dir: Base directory for outputs
            training_params: Parameters for training
        """
        self.db_path = Path(db_path)
        self.timeframe = timeframe
        self.output_dir = Path(output_dir)
        
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
            'seed': 42
        }
        
        # Results tracking
        self.training_results = {}
        self.evaluation_results = {}
    
    def load_symbols(self, symbols_file: Optional[str] = None) -> List[str]:
        """
        Load S&P 100 symbols from file or use defaults.
        
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
    
    def get_daily_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get daily data for a symbol from the database.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with stock data or None if not available
        """
        try:
            return get_daily_data(symbol, timeframe=self.timeframe)
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return None
    
    def train_on_symbol(self, symbol: str) -> Tuple[Optional[TradingAgent], Dict[str, Any]]:
        """
        Train an agent on a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (agent, results dictionary)
        """
        print(f"\n{'='*20} Training on {symbol} {'='*20}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Fetch data from SQLite database
            stock_data = self.get_daily_data(symbol)
            
            if stock_data is None or len(stock_data) < 252:  # Minimum 1 year of data
                print(f"Insufficient data for {symbol}. Skipping.")
                return None, {"status": "skipped", "reason": "insufficient_data"}
            
            print(f"Loaded {len(stock_data)} days of data for {symbol}")
            
            # Process data
            data_processor = DataProcessor(
                stock_data, 
                window_size=self.training_params['window_size']
            )
            processed_data = data_processor.process_data()
            
            # Split into train and validation sets (80/20)
            train_size = int(len(processed_data) * 0.8)
            train_data = processed_data.iloc[:train_size]
            val_data = processed_data.iloc[train_size:]
            
            print(f"Training data: {len(train_data)} days, Validation data: {len(val_data)} days")
            
            # Create output directories for this run
            model_dir = self.models_dir / f"{symbol}_{timestamp}"
            log_dir = self.logs_dir / f"{symbol}_{timestamp}"
            viz_dir = self.viz_dir / f"{symbol}_{timestamp}"
            
            for directory in [model_dir, log_dir, viz_dir]:
                directory.mkdir(exist_ok=True, parents=True)
            
            # Create environments
            train_env = StockTradingEnv(
                price_data=train_data,
                initial_capital=self.training_params['initial_capital'],
                window_size=self.training_params['window_size'],
                max_position_pct=self.training_params['max_position_pct'],
                transaction_cost_pct=self.training_params['transaction_cost_pct'],
                curriculum_level=self.training_params['curriculum_level']
            )
            
            val_env = StockTradingEnv(
                price_data=val_data,
                initial_capital=self.training_params['initial_capital'],
                window_size=self.training_params['window_size'],
                max_position_pct=self.training_params['max_position_pct'],
                transaction_cost_pct=self.training_params['transaction_cost_pct'],
                curriculum_level=self.training_params['curriculum_level']
            )
            
            # Wrap environments with Monitor
            train_env = Monitor(train_env, str(log_dir / "train_monitor"))
            val_env = Monitor(val_env, str(log_dir / "val_monitor"))
            
            # Create agent
            agent = TradingAgent(
                env=train_env,
                learning_rate=self.training_params['learning_rate'],
                buffer_size=self.training_params['buffer_size'],
                batch_size=self.training_params['batch_size'],
                seed=self.training_params['seed']
            )
            
            # Set up callbacks
            callbacks = [
                # Checkpoint callback
                CheckpointCallback(
                    save_freq=max(self.training_params['total_timesteps'] // 10, 1000),
                    save_path=str(model_dir),
                    name_prefix=f"{symbol}_model",
                    verbose=1
                ),
                
                # Curriculum learning callback
                CurriculumLearningCallback(
                    env=train_env,
                    target_reward=0.5,
                    window_size=20,
                    verbose=1
                ),
                
                # Metrics logger callback
                MetricsLoggerCallback(
                    eval_env=val_env,
                    log_path=str(log_dir / "metrics.csv"),
                    log_freq=5000,
                    verbose=1
                ),
                
                # Visualization callback
                VisualizeTradesCallback(
                    eval_env=val_env,
                    log_dir=str(viz_dir),
                    plot_freq=10000,
                    n_eval_episodes=1,
                    verbose=1
                ),
                
                # Early stopping callback
                TradingEarlyStoppingCallback(
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
            ]
            
            # Train agent
            print(f"Starting training for {symbol}...")
            start_time = datetime.now()
            
            agent.train(
                total_timesteps=self.training_params['total_timesteps'],
                callback_list=callbacks,
                eval_env=val_env,
                eval_freq=5000,
                log_dir=str(log_dir),
                model_dir=str(model_dir)
            )
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds() / 60  # in minutes
            
            # Save final model
            final_model_path = str(model_dir / f"{symbol}_final")
            agent.save(final_model_path)
            
            print(f"Training completed in {training_time:.2f} minutes")
            print(f"Model saved to {final_model_path}")
            
            # Return results
            return agent, {
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
            
            return None, {
                'symbol': symbol,
                'status': 'failed',
                'reason': 'exception',
                'error': str(e)
            }
    
    def evaluate_model(self, symbol: str, model_path: str) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            symbol: Stock symbol
            model_path: Path to trained model
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*20} Evaluating {symbol} {'='*20}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = self.results_dir / f"{symbol}_{timestamp}"
        eval_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Load test data (last 20% of available data)
            stock_data = self.get_daily_data(symbol)
            
            if stock_data is None:
                return {"status": "failed", "reason": "no_data"}
            
            # Process data
            data_processor = DataProcessor(stock_data, window_size=self.training_params['window_size'])
            processed_data = data_processor.process_data()
            
            # Use the last 20% for evaluation
            test_size = int(len(processed_data) * 0.2)
            test_data = processed_data.iloc[-test_size:]
            
            print(f"Evaluating on {len(test_data)} days of data")
            
            # Create evaluation environment
            eval_env = StockTradingEnv(
                price_data=test_data,
                initial_capital=self.training_params['initial_capital'],
                window_size=self.training_params['window_size'],
                max_position_pct=self.training_params['max_position_pct'],
                transaction_cost_pct=self.training_params['transaction_cost_pct']
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
                
                # Record trade if one occurred
                if info.get('trade_executed', False):
                    trade_data = {
                        'step': step,
                        'type': info.get('trade_type', ''),
                        'price': info.get('trade_price', 0),
                        'shares': info.get('trade_shares', 0)
                    }
                    trades.append(trade_data)
            
            # Calculate evaluation metrics
            final_value = portfolio_values[-1]
            initial_value = portfolio_values[0]
            total_return = (final_value / initial_value - 1) * 100
            
            # Calculate returns and volatility
            returns = []
            for i in range(1, len(portfolio_values)):
                ret = (portfolio_values[i] / portfolio_values[i-1]) - 1
                returns.append(ret)
            
            avg_return = np.mean(returns) * 100 if returns else 0
            volatility = np.std(returns) * 100 * np.sqrt(252) if returns else 0
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
            
            # Calculate max drawdown
            peak = portfolio_values[0]
            max_drawdown = 0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
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
                'num_trades': len(trades)
            }
            
            # Save evaluation results
            with open(eval_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Save portfolio history
            pd.DataFrame({
                "portfolio_value": portfolio_values,
                "position_size": position_sizes,
                "reward": rewards
            }).to_csv(eval_dir / "portfolio_history.csv", index=False)
            
            # Save trades
            if trades:
                pd.DataFrame(trades).to_csv(eval_dir / "trades.csv", index=False)
            
            # Create visualization
            self._create_evaluation_plot(
                portfolio_values, position_sizes, trades, 
                symbol, eval_dir / "evaluation_plot.png"
            )
            
            print("\nEvaluation results:")
            print(f"Total return: {total_return:.2f}%")
            print(f"Sharpe ratio: {sharpe:.2f}")
            print(f"Max drawdown: {max_drawdown*100:.2f}%")
            print(f"Number of trades: {len(trades)}")
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating {symbol}: {e}")
            traceback.print_exc()
            
            return {
                'symbol': symbol,
                'status': 'failed',
                'reason': 'exception',
                'error': str(e)
            }
    
    def _create_evaluation_plot(
        self, 
        portfolio_values: List[float], 
        position_sizes: List[float],
        trades: List[Dict[str, Any]],
        symbol: str, 
        output_path: Path
    ) -> None:
        """
        Create evaluation plot with portfolio value and positions.
        
        Args:
            portfolio_values: List of portfolio values
            position_sizes: List of position sizes
            trades: List of trades executed
            symbol: Stock symbol
            output_path: Path to save the plot
        """
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
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def run(self, symbols: Optional[List[str]] = None, symbols_file: Optional[str] = None) -> Dict:
        """
        Run the training and evaluation process for multiple symbols.
        
        Args:
            symbols: List of symbols to train on
            symbols_file: Path to file with symbols (one per line)
            
        Returns:
            Dictionary with training and evaluation results
        """
        # Get symbols
        symbols_to_use = symbols or self.load_symbols(symbols_file)
        print(f"Training on {len(symbols_to_use)} symbols: {', '.join(symbols_to_use[:5])}...")
        
        # Train and evaluate each symbol
        for symbol in symbols_to_use:
            # Train model
            agent, train_result = self.train_on_symbol(symbol)
            self.training_results[symbol] = train_result
            
            # If training successful, evaluate
            if agent is not None and train_result.get('status') == 'success':
                model_path = train_result.get('model_path')
                eval_result = self.evaluate_model(symbol, model_path)
                self.evaluation_results[symbol] = eval_result
        
        # Save overall results
        results_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"sp100_training_summary_{results_timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "training_params": self.training_params,
                "training_results": self.training_results,
                "evaluation_results": self.evaluation_results
            }, f, indent=2)
        
        print(f"Results saved to {results_file}")
        
        # Print summary of best models
        self.print_summary()
        
        return {
            "training_results": self.training_results,
            "evaluation_results": self.evaluation_results
        }
    
    def print_summary(self) -> None:
        """Print a summary of the results."""
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
            
            print("\nTop 5 models by Sharpe ratio:")
            for i, (symbol, result) in enumerate(sorted_results[:5], 1):
                sharpe = result.get('sharpe_ratio', 0)
                returns = result.get('total_return', 0)
                drawdown = result.get('max_drawdown', 0)
                trades = result.get('num_trades', 0)
                
                print(f"{i}. {symbol}: Sharpe={sharpe:.2f}, Return={returns:.2f}%, " 
                      f"Drawdown={drawdown:.2f}%, Trades={trades}")
        else:
            print("No successful evaluations to report.")


# Example usage
if __name__ == "__main__":
    # Configure trainer
    trainer = SP100Trainer(
        db_path="data_cache/alpaca_data.db",
        timeframe="1Day",
        output_dir="sp100_training"
    )
    
    # Run with default or custom symbols
    # trainer.run(symbols=['AAPL', 'MSFT', 'GOOGL'])  # Using specific symbols
    trainer.run(symbols_file="sp100_symbols.txt")  # Using symbols from file