import os
import sys
import json
import time
import pickle
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

# Import environment and agent components
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

# Data cache paths
DATA_DIR = Path("data_cache")
SP500_LIST_FILE = DATA_DIR / "sp500_list.json"
SP500_DATA_DIR = DATA_DIR / "stocks"
METADATA_FILE = DATA_DIR / "metadata.json"


def setup_folders():
    """Create necessary folders for data caching and outputs."""
    folders = [
        DATA_DIR,
        SP500_DATA_DIR,
        Path("logs"),
        Path("models"),
        Path("visualizations")
    ]
    
    for folder in folders:
        folder.mkdir(exist_ok=True, parents=True)
    
    print(f"Created folder structure at {os.getcwd()}")


def get_sp500_list(force_update: bool = False) -> List[str]:
    """
    Get list of S&P 500 companies using Wikipedia or cached list.
    
    Args:
        force_update: Whether to force an update of the list
        
    Returns:
        tickers: List of S&P 500 ticker symbols
    """
    if SP500_LIST_FILE.exists() and not force_update:
        # Use cached list if available and not forcing update
        with open(SP500_LIST_FILE, 'r') as f:
            tickers = json.load(f)
            print(f"Loaded {len(tickers)} S&P 500 tickers from cache")
            return tickers
    
    try:
        # Scrape the current list from Wikipedia
        print("Downloading current S&P 500 list from Wikipedia...")
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = table['Symbol'].str.replace('.', '-').tolist()
        
        # Save to cache
        with open(SP500_LIST_FILE, 'w') as f:
            json.dump(tickers, f)
        
        print(f"Downloaded and cached {len(tickers)} S&P 500 tickers")
        return tickers
    
    except Exception as e:
        print(f"Error getting S&P 500 list: {e}")
        
        # If cache exists, fall back to it
        if SP500_LIST_FILE.exists():
            with open(SP500_LIST_FILE, 'r') as f:
                tickers = json.load(f)
                print(f"Falling back to cached list with {len(tickers)} tickers")
                return tickers
        
        # Last resort: return a few major tickers
        fallback_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'JPM', 'JNJ', 'V', 'PG']
        print(f"Using fallback list of {len(fallback_tickers)} major tickers")
        return fallback_tickers


def load_metadata() -> Dict:
    """Load metadata about cached stocks."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {"last_update": "", "stocks": {}}


def save_metadata(metadata: Dict) -> None:
    """Save metadata about cached stocks."""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def download_stock_data(
    ticker: str, 
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD), defaults to today
        interval: Data interval (1d, 1wk, 1mo)
        
    Returns:
        data: OHLCV dataframe for the stock
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    try:
        # Download data using yfinance
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval=interval)
        
        # Rename columns to lowercase for consistency with our project
        data.columns = [col.lower() for col in data.columns]
        
        # Check if we got valid data
        if data.empty or len(data) < 50:
            print(f"⚠️ Not enough data for {ticker}, skipping")
            return pd.DataFrame()
        
        print(f"✓ Downloaded {len(data)} rows for {ticker}")
        return data
    
    except Exception as e:
        print(f"❌ Error downloading data for {ticker}: {e}")
        return pd.DataFrame()


def update_stock_cache(
    tickers: List[str],
    force_update: bool = False,
    min_update_days: int = 7
) -> Dict[str, str]:
    """
    Update the local cache of stock data.
    
    Args:
        tickers: List of ticker symbols to update
        force_update: Whether to force an update regardless of cache status
        min_update_days: Minimum days since last update to trigger a new download
        
    Returns:
        status: Dictionary mapping tickers to status messages
    """
    # Load metadata
    metadata = load_metadata()
    metadata.setdefault("stocks", {})
    
    # Set current date as the update date
    today = datetime.now().strftime("%Y-%m-%d")
    metadata["last_update"] = today
    
    # Track status of each ticker
    status = {}
    
    # Create ticker list with relative trading volumes for prioritization
    ticker_priority = []
    
    for ticker in tickers:
        # Create cache path
        cache_file = SP500_DATA_DIR / f"{ticker}.pkl"
        
        # Check if data exists in cache and is recent enough
        need_update = force_update
        
        if not cache_file.exists():
            need_update = True
            last_updated = "never"
        else:
            last_updated = metadata.get("stocks", {}).get(ticker, {}).get("last_updated", "never")
            
            # Check if data is older than min_update_days
            if last_updated != "never":
                last_date = datetime.strptime(last_updated, "%Y-%m-%d")
                days_since_update = (datetime.now() - last_date).days
                if days_since_update >= min_update_days:
                    need_update = True
        
        # Add to priority list (we'll sort by trading volume later)
        avg_volume = metadata.get("stocks", {}).get(ticker, {}).get("avg_volume", 0)
        ticker_priority.append((ticker, avg_volume, need_update, last_updated))
    
    # Sort by need_update (True first) and then by volume (higher volume first)
    ticker_priority.sort(key=lambda x: (-int(x[2]), -x[1]))
    
    # Process tickers with rate limiting to avoid API issues
    for i, (ticker, _, need_update, last_updated) in enumerate(ticker_priority):
        cache_file = SP500_DATA_DIR / f"{ticker}.pkl"
        
        if need_update:
            print(f"[{i+1}/{len(ticker_priority)}] Updating {ticker} (last updated: {last_updated})")
            
            # Download data (from 2018 or beginning of COVID pandemic)
            data = download_stock_data(ticker, start_date="2018-01-01")
            
            if not data.empty:
                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                
                # Update metadata
                metadata.setdefault("stocks", {}).setdefault(ticker, {})
                metadata["stocks"][ticker]["last_updated"] = today
                metadata["stocks"][ticker]["rows"] = len(data)
                metadata["stocks"][ticker]["start_date"] = data.index[0].strftime("%Y-%m-%d")
                metadata["stocks"][ticker]["end_date"] = data.index[-1].strftime("%Y-%m-%d")
                metadata["stocks"][ticker]["avg_volume"] = float(data["volume"].mean()) if "volume" in data.columns else 0
                
                status[ticker] = "updated"
                
                # Save metadata after each successful download to track progress
                save_metadata(metadata)
            else:
                status[ticker] = "failed"
                
            # Add delay to avoid hitting API limits
            if i < len(ticker_priority) - 1 and need_update:
                time.sleep(1.5)  # 1.5 seconds between requests
        else:
            status[ticker] = "cached"
    
    # Final save of metadata
    save_metadata(metadata)
    
    # Print summary
    updated = sum(1 for s in status.values() if s == "updated")
    cached = sum(1 for s in status.values() if s == "cached")
    failed = sum(1 for s in status.values() if s == "failed")
    
    print(f"\nStock cache update summary:")
    print(f"  Updated: {updated}")
    print(f"  Already cached: {cached}")
    print(f"  Failed: {failed}")
    
    return status


def load_stock_data(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load stock data from cache.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        data: OHLCV dataframe or None if not found
    """
    cache_file = SP500_DATA_DIR / f"{ticker}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Error loading cached data for {ticker}: {e}")
    
    return None


def filter_suitable_stocks(
    metadata: Dict, 
    min_data_points: int = 500,
    min_avg_volume: float = 500000,
    min_price: float = 5.0
) -> List[str]:
    """
    Filter stocks suitable for training based on data quality.
    
    Args:
        metadata: Metadata dictionary with stock information
        min_data_points: Minimum number of data points required
        min_avg_volume: Minimum average daily volume
        min_price: Minimum average price
        
    Returns:
        suitable_tickers: List of suitable ticker symbols
    """
    suitable_tickers = []
    
    for ticker, info in metadata.get("stocks", {}).items():
        if info.get("rows", 0) >= min_data_points and info.get("avg_volume", 0) >= min_avg_volume:
            # Load data to check price
            data = load_stock_data(ticker)
            if data is not None and not data.empty:
                avg_price = data["close"].mean()
                if avg_price >= min_price:
                    suitable_tickers.append(ticker)
    
    return suitable_tickers


def train_on_stock(
    ticker: str,
    training_params: Dict[str, Any],
    model_path: Optional[str] = None
) -> Tuple[TradingAgent, Dict[str, Any]]:
    """
    Train an agent on a single stock.
    
    Args:
        ticker: Stock ticker symbol
        training_params: Parameters for training
        model_path: Path to existing model to continue training (optional)
        
    Returns:
        agent: Trained TradingAgent
        metrics: Training metrics
    """
    print(f"\n{'='*20} Training on {ticker} {'='*20}\n")
    
    # Load data
    data = load_stock_data(ticker)
    if data is None or data.empty:
        print(f"No data available for {ticker}")
        return None, {"status": "failed", "reason": "no_data"}
    
    # Process data
    window_size = training_params.get("window_size", 20)
    data_processor = DataProcessor(data, window_size=window_size)
    processed_data = data_processor.process_data()
    
    # Split into train and validation sets
    train_size = int(len(processed_data) * 0.8)
    train_data = processed_data.iloc[:train_size]
    val_data = processed_data.iloc[train_size:]
    
    if len(train_data) < 200 or len(val_data) < 50:
        print(f"Insufficient data for {ticker} after split")
        return None, {"status": "failed", "reason": "insufficient_data"}
    
    print(f"Data split: {len(train_data)} training samples, {len(val_data)} validation samples")
    
    # Create timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/{ticker}_{timestamp}")
    model_dir = Path(f"models/{ticker}_{timestamp}")
    viz_dir = Path(f"visualizations/{ticker}_{timestamp}")
    
    # Create directories
    log_dir.mkdir(exist_ok=True, parents=True)
    model_dir.mkdir(exist_ok=True, parents=True)
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Create environments
    train_env = StockTradingEnv(
        price_data=train_data,
        initial_capital=training_params.get("initial_capital", 100000.0),
        window_size=window_size,
        max_position_pct=training_params.get("max_position_pct", 0.25),
        transaction_cost_pct=training_params.get("transaction_cost_pct", 0.0015),
        curriculum_level=training_params.get("curriculum_level", 1),
    )
    
    val_env = StockTradingEnv(
        price_data=val_data,
        initial_capital=training_params.get("initial_capital", 100000.0),
        window_size=window_size,
        max_position_pct=training_params.get("max_position_pct", 0.25),
        transaction_cost_pct=training_params.get("transaction_cost_pct", 0.0015),
        curriculum_level=training_params.get("curriculum_level", 1),
    )
    
    # Wrap environments with monitor
    train_env = Monitor(train_env, str(log_dir / "train_monitor"))
    val_env = Monitor(val_env, str(log_dir / "val_monitor"))
    
    # Create agent
    agent = TradingAgent(
        env=train_env,
        model_path=model_path,
        learning_rate=training_params.get("learning_rate", 1e-4),
        buffer_size=training_params.get("buffer_size", 100000),
        batch_size=training_params.get("batch_size", 256),
        seed=training_params.get("seed", 42)
    )
    
    # Set up callbacks
    # 1. Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(training_params.get("total_timesteps", 100000) // 10, 1000),
        save_path=str(model_dir),
        name_prefix=f"{ticker}_model",
        verbose=1
    )
    
    # 2. Curriculum learning callback
    curriculum_callback = CurriculumLearningCallback(
        env=train_env,
        target_reward=training_params.get("curriculum_target_reward", 0.5),
        window_size=training_params.get("curriculum_window_size", 20),
        verbose=1
    )
    
    # 3. Metrics logger callback
    metrics_callback = MetricsLoggerCallback(
        eval_env=val_env,
        log_path=str(log_dir / "metrics.csv"),
        log_freq=5000,
        verbose=1
    )
    
    # 4. Visualization callback
    viz_callback = VisualizeTradesCallback(
        eval_env=val_env,
        log_dir=str(viz_dir),
        plot_freq=10000,
        n_eval_episodes=1,
        verbose=1
    )
    
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
    
    # Combine callbacks
    callbacks = CallbackList([
        checkpoint_callback,
        curriculum_callback,
        metrics_callback,
        viz_callback,
        early_stopping
    ])
    
    # Train agent
    try:
        print(f"Starting training for {ticker}...")
        agent.train(
            total_timesteps=training_params.get("total_timesteps", 100000),
            callback_list=callbacks,
            eval_env=val_env,
            eval_freq=5000,
            log_dir=str(log_dir),
            model_dir=str(model_dir)
        )
        
        # Save final model
        final_model_path = str(model_dir / f"{ticker}_final")
        agent.save(final_model_path)
        print(f"Model saved to {final_model_path}")
        
        return agent, {
            "status": "success",
            "ticker": ticker,
            "model_path": final_model_path,
            "log_dir": str(log_dir),
            "train_samples": len(train_data),
            "val_samples": len(val_data)
        }
    
    except Exception as e:
        print(f"Error training on {ticker}: {e}")
        traceback.print_exc()
        
        return None, {
            "status": "failed",
            "reason": "training_error",
            "error": str(e)
        }


def evaluate_agent(
    agent: TradingAgent,
    ticker: str,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained agent on the full dataset.
    
    Args:
        agent: Trained TradingAgent
        ticker: Stock ticker symbol
        output_dir: Directory for evaluation outputs
        
    Returns:
        metrics: Evaluation metrics
    """
    print(f"\n{'='*20} Evaluating agent on {ticker} {'='*20}\n")
    
    # Load data
    data = load_stock_data(ticker)
    if data is None or data.empty:
        print(f"No data available for {ticker}")
        return {"status": "failed", "reason": "no_data"}
    
    # Process data
    window_size = agent.env.window_size if hasattr(agent.env, 'window_size') else 20
    data_processor = DataProcessor(data, window_size=window_size)
    processed_data = data_processor.process_data()
    
    # Create evaluation environment
    eval_env = StockTradingEnv(
        price_data=processed_data,
        initial_capital=100000.0,
        window_size=window_size,
        max_position_pct=0.25,
        transaction_cost_pct=0.0015,
        max_episodes=1
    )
    
    # Reset environment
    obs, _ = eval_env.reset()
    done = False
    cumulative_reward = 0
    step_count = 0
    portfolio_values = []
    actions_taken = []
    position_sizes = []
    returns = []
    drawdowns = []
    
    # Run one episode
    print("Running evaluation episode...")
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        cumulative_reward += reward
        step_count += 1
        
        # Record information
        portfolio_values.append(info['portfolio_value'])
        position_sizes.append(info['current_position'])
        drawdowns.append(info['drawdown'])
        
        # Calculate returns
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] / portfolio_values[-2]) - 1
            returns.append(daily_return)
        
        # Record action
        actions_taken.append({
            'step': step_count,
            'action': action.tolist(),
            'reward': reward,
            'portfolio_value': info['portfolio_value'],
            'position': info['current_position'],
            'drawdown': info['drawdown'],
            'trade_executed': info.get('trade_executed', False),
            'trade_type': info.get('trade_type', None),
            'trade_price': info.get('trade_price', None)
        })
        
        # Print progress
        if step_count % 50 == 0:
            print(f"Step {step_count}: Portfolio = ${info['portfolio_value']:.2f}, Return: {(info['portfolio_value']/100000-1)*100:.2f}%")
    
    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value / 100000 - 1) * 100
    max_drawdown = max(drawdowns) * 100
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
    
    # Count trades
    trades = [a for a in actions_taken if a.get('trade_executed', False)]
    buy_trades = [t for t in trades if t.get('trade_type') == 'buy']
    sell_trades = [t for t in trades if t.get('trade_type') == 'sell']
    
    metrics = {
        "status": "success",
        "ticker": ticker,
        "final_portfolio_value": final_value,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "total_steps": step_count,
        "number_of_trades": len(buy_trades),
        "avg_position_size": np.mean([p for p in position_sizes if p > 0]) if any(p > 0 for p in position_sizes) else 0,
        "position_utilization": sum(1 for p in position_sizes if p > 0) / len(position_sizes) if position_sizes else 0
    }
    
    # Save evaluation results if output_dir is provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save metrics
        with open(output_path / f"{ticker}_eval_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save portfolio value history
        pd.DataFrame({
            'portfolio_value': portfolio_values,
            'position_size': position_sizes,
            'drawdown': drawdowns
        }).to_csv(output_path / f"{ticker}_eval_history.csv")
        
        # Create plots
        try:
            # Plot portfolio value
            plt.figure(figsize=(12, 8))
            plt.subplot(3, 1, 1)
            plt.plot(portfolio_values)
            plt.title(f"Portfolio Value - {ticker}")
            plt.ylabel("Value ($)")
            
            # Plot drawdown
            plt.subplot(3, 1, 2)
            plt.plot(np.array(drawdowns) * 100)
            plt.fill_between(range(len(drawdowns)), np.array(drawdowns) * 100, 0, alpha=0.3)
            plt.title("Drawdown (%)")
            plt.ylabel("Drawdown (%)")
            
            # Plot position sizes
            plt.subplot(3, 1, 3)
            plt.plot(position_sizes)
            plt.title("Position Size")
            plt.ylabel("Shares")
            plt.xlabel("Step")
            
            plt.tight_layout()
            plt.savefig(output_path / f"{ticker}_eval_summary.png", dpi=100)
            plt.close()
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    # Print summary
    print(f"\nEvaluation summary for {ticker}:")
    print(f"  Final portfolio value: ${final_value:.2f}")
    print(f"  Total return: {total_return:.2f}%")
    print(f"  Maximum drawdown: {max_drawdown:.2f}%")
    print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"  Number of trades: {len(buy_trades)}")
    
    return metrics


def main():
    """Main execution function."""
    try:
        print("\n=== S&P 500 Reinforcement Learning Trading Bot ===\n")
        
        # Create necessary folders
        setup_folders()
        
        # Get S&P 500 ticker list
        tickers = get_sp500_list(force_update=False)
        print(f"Working with {len(tickers)} S&P 500 tickers")
        
        # Update stock data cache (only update tickers older than 7 days)
        update_stock_cache(tickers, force_update=False, min_update_days=7)
        
        # Load metadata
        metadata = load_metadata()
        
        # Filter suitable stocks for training
        suitable_tickers = filter_suitable_stocks(
            metadata, 
            min_data_points=500,
            min_avg_volume=1000000,  # 1M average volume
            min_price=10.0           # $10 minimum price
        )
        
        print(f"\nFound {len(suitable_tickers)} suitable stocks for training")
        if len(suitable_tickers) > 10:
            print(f"Sample of suitable tickers: {', '.join(suitable_tickers[:10])}")
        
        # Select a subset for training
        max_stocks = 5  # Start with a small number for testing
        training_tickers = suitable_tickers[:max_stocks]
        
        print(f"\nSelected {len(training_tickers)} stocks for training: {', '.join(training_tickers)}")
        
        # Define training parameters
        training_params = {
            "initial_capital": 100000.0,
            "window_size": 20,
            "max_position_pct": 0.25,
            "transaction_cost_pct": 0.0015,
            "curriculum_level": 1,
            "learning_rate": 3e-5,
            "buffer_size": 200000,
            "batch_size": 256,
            "total_timesteps": 100000,  # Increase for better results
            "seed": 42,
            "curriculum_target_reward": 0.5,
            "curriculum_window_size": 20
        }
        
        # Store results
        training_results = {}
        evaluation_results = {}
        
        # Train on each stock
        for ticker in training_tickers:
            try:
                # Train agent
                agent, train_result = train_on_stock(ticker, training_params)
                training_results[ticker] = train_result
                
                if train_result["status"] == "success":
                    # Evaluate agent
                    eval_output_dir = f"evaluations/{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    eval_result = evaluate_agent(agent, ticker, eval_output_dir)
                    evaluation_results[ticker] = eval_result
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                traceback.print_exc()
                training_results[ticker] = {"status": "failed", "reason": "exception", "error": str(e)}
        
        # Save overall results
        results_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/training_summary_{results_timestamp}.json", 'w') as f:
            json.dump({
                "training_params": training_params,
                "training_results": training_results,
                "evaluation_results": evaluation_results
            }, f, indent=2)
        
        # Print summary
        successful_training = sum(1 for r in training_results.values() if r.get("status") == "success")
        successful_evaluation = sum(1 for r in evaluation_results.values() if r.get("status") == "success")
        
        print(f"\n{'='*40}")
        print(f"Training completed: {successful_training}/{len(training_tickers)} successful")
        print(f"Evaluation completed: {successful_evaluation}/{successful_training} successful")
        print(f"Results saved to logs/training_summary_{results_timestamp}.json")
        print(f"{'='*40}\n")
        
        # Print top performing models based on Sharpe ratio
        if evaluation_results:
            print("Top performing models by Sharpe ratio:")
            sorted_results = sorted(
                [(ticker, res) for ticker, res in evaluation_results.items() if res.get("status") == "success"],
                key=lambda x: x[1].get("sharpe_ratio", -999),
                reverse=True
            )
            
            for i, (ticker, res) in enumerate(sorted_results[:5], 1):
                print(f"{i}. {ticker}: Sharpe={res.get('sharpe_ratio', 0):.2f}, Return={res.get('total_return', 0):.2f}%, Drawdown={res.get('max_drawdown', 0):.2f}%")
        
    except Exception as e:
        print(f"Fatal error in main execution: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())