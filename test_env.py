import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your environment
from env.core import StockTradingEnv
from data.processor import DataProcessor
from utils.debug_utils import (
    run_debug_episodes,
    plot_debug_results,
    analyze_rewards,
    validate_environment
)

# Function to load sample data
def load_sample_data():
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
    
    return df

def main():
    # Load data
    print("Loading data...")
    price_data = load_sample_data()
    
    # Process data
    print("Processing data...")
    data_processor = DataProcessor(price_data, window_size=20)
    processed_data = data_processor.process_data()
    
    # Create environment
    print("Creating environment...")
    env = StockTradingEnv(
        price_data=processed_data,
        initial_capital=100000.0,
        debug_mode=True
    )
    
    # Validate environment
    print("\n=== Environment Validation ===")
    validation_results = validate_environment(env)
    
    # Run debug episodes
    print("\n=== Running Debug Episodes ===")
    debug_stats = run_debug_episodes(env, n_episodes=3, verbose=True)
    
    # Plot results
    print("\n=== Plotting Results ===")
    plot_debug_results(debug_stats)
    
    # Analyze rewards
    print("\n=== Analyzing Rewards ===")
    analyze_rewards(debug_stats)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()