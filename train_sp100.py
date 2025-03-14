#!/usr/bin/env python3
"""
train_sp100.py

Script for training a multi-stock model on up to 100 stocks from the database.

This script:
1. Extracts all available stock symbols from the database
2. Filters symbols to ensure sufficient data quality
3. Trains a single reinforcement learning model on all filtered stocks
4. Evaluates the model performance on individual stocks
5. Provides detailed performance metrics

Usage:
    python train_sp100.py [arguments]

Author: [Your Name]
Date: March 14, 2025
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
from datetime import datetime
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Import components
from multi_stock_trainer import MultiStockTrainer
from flexible_data_processor import get_available_symbols, get_daily_data


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a reinforcement learning agent on up to 100 stocks"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="data_cache/alpaca_data.db",
        help="Path to SQLite database with stock data"
    )
    
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1Day",
        help="Timeframe to use for training (e.g., '1Day')"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"sp100_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=100,
        help="Maximum number of symbols to include"
    )
    
    parser.add_argument(
        "--min-days",
        type=int,
        default=252,
        help="Minimum number of trading days required (default: 1 year)"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="Number of timesteps for training (increased for larger symbol set)"
    )
    
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=325,
        help="Observation dimension"
    )
    
    parser.add_argument(
        "--symbol-dim",
        type=int,
        default=20,
        help="Symbol embedding dimension"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def get_quality_symbols(db_path: str, timeframe: str, min_days: int, max_symbols: int) -> List[str]:
    """
    Get a list of symbols with good data quality.
    
    Args:
        db_path: Path to the database
        timeframe: Timeframe to use (e.g., '1Day')
        min_days: Minimum number of days required
        max_symbols: Maximum number of symbols to include
        
    Returns:
        List of filtered symbols
    """
    # Get all available symbols
    all_symbols = get_available_symbols()
    print(f"Found {len(all_symbols)} total symbols in database")
    
    # Filter symbols by data quality
    quality_symbols = []
    
    print("Checking symbol data quality...")
    for i, symbol in enumerate(all_symbols):
        # Show progress
        if i % 10 == 0:
            print(f"Checked {i}/{len(all_symbols)} symbols...", end='\r')
        
        # Get data for symbol
        data = get_daily_data(symbol, timeframe=timeframe)
        
        # Check if we have enough data
        if data is not None and len(data) >= min_days:
            quality_symbols.append(symbol)
        
        # Stop if we have enough symbols
        if len(quality_symbols) >= max_symbols:
            break
    
    print(f"\nFound {len(quality_symbols)} symbols with at least {min_days} days of data")
    return quality_symbols[:max_symbols]


def main():
    """Main function to train and evaluate models on multiple stocks."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Print configuration
        print("\n=== SP100 Multi-Stock RL Training ===")
        print(f"Database: {args.db_path}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Output directory: {args.output_dir}")
        print(f"Training timesteps: {args.timesteps}")
        print(f"Observation dimension: {args.obs_dim}")
        print(f"Symbol embedding dimension: {args.symbol_dim}")
        print(f"Random seed: {args.seed}")
        print(f"Maximum symbols: {args.max_symbols}")
        print(f"Minimum days: {args.min_days}")
        
        # Get quality symbols
        symbols_to_train = get_quality_symbols(
            args.db_path, 
            args.timeframe, 
            args.min_days, 
            args.max_symbols
        )
        
        # Print symbols
        print(f"\nTraining on {len(symbols_to_train)} symbols:")
        for i in range(0, len(symbols_to_train), 10):
            print(f"  {', '.join(symbols_to_train[i:i+10])}")
        
        # Define training parameters
        training_params = {
            'initial_capital': 100000.0,
            'window_size': 20,
            'max_position_pct': 0.25,
            'transaction_cost_pct': 0.0015,
            'curriculum_level': 1,
            'learning_rate': 1e-5,  # Smaller learning rate for more stability with more symbols
            'buffer_size': 500000,  # Larger buffer for more symbols
            'batch_size': 512,      # Larger batch size
            'total_timesteps': args.timesteps,
            'seed': args.seed,
            'observation_dim': args.obs_dim,
            'symbol_feature_dim': args.symbol_dim
        }
        
        # Initialize the trainer
        trainer = MultiStockTrainer(
            db_path=args.db_path,
            timeframe=args.timeframe,
            output_dir=args.output_dir,
            training_params=training_params,
            observation_dim=args.obs_dim,
            symbol_feature_dim=args.symbol_dim
        )
        
        # Start training and evaluation
        print("\nStarting multi-stock training process...")
        results = trainer.run(symbols=symbols_to_train)
        
        # Print final summary
        print("\n=== Training Complete ===")
        
        # Check training results
        if results['training_results'].get('status') == 'success':
            # If single model training succeeded
            print("\nMulti-stock model training successful!")
            
            # Count successful evaluations
            successful_evals = sum(1 for r in results['evaluation_results'].values() 
                                 if r.get('status') == 'success')
            print(f"Successfully evaluated on {successful_evals}/{len(results['evaluation_results'])} symbols")
            
            # Print details of evaluations
            if successful_evals > 0:
                print("\nTop 20 symbols by Sharpe ratio:")
                
                # Sort by Sharpe ratio
                sorted_evals = sorted(
                    [(s, r) for s, r in results['evaluation_results'].items() if r.get('status') == 'success'],
                    key=lambda x: x[1].get('sharpe_ratio', -999),
                    reverse=True
                )
                
                for i, (symbol, result) in enumerate(sorted_evals[:20], 1):
                    print(f"{i:2d}. {symbol:5s}: Return={result.get('total_return', 0):6.2f}%, " +
                         f"Sharpe={result.get('sharpe_ratio', 0):5.2f}, " +
                         f"Drawdown={result.get('max_drawdown', 0):5.2f}%")
                
                # Calculate average metrics
                avg_return = np.mean([r.get('total_return', 0) for s, r in sorted_evals])
                avg_sharpe = np.mean([r.get('sharpe_ratio', 0) for s, r in sorted_evals])
                avg_drawdown = np.mean([r.get('max_drawdown', 0) for s, r in sorted_evals])
                
                print(f"\nAverage metrics across {len(sorted_evals)} symbols:")
                print(f"  Return: {avg_return:.2f}%")
                print(f"  Sharpe: {avg_sharpe:.2f}")
                print(f"  Drawdown: {avg_drawdown:.2f}%")
                
                # Save summary to CSV
                results_df = pd.DataFrame([
                    {
                        'symbol': symbol,
                        'total_return': result.get('total_return', 0),
                        'sharpe_ratio': result.get('sharpe_ratio', 0),
                        'max_drawdown': result.get('max_drawdown', 0),
                        'num_trades': result.get('num_buy_trades', 0) + result.get('num_sell_trades', 0)
                    }
                    for symbol, result in results['evaluation_results'].items()
                    if result.get('status') == 'success'
                ])
                
                results_csv = Path(args.output_dir) / "evaluation_results.csv"
                results_df.to_csv(results_csv, index=False)
                print(f"\nDetailed results saved to {results_csv}")
        else:
            # If training failed
            print("Multi-stock model training failed.")
            print(f"Reason: {results['training_results'].get('reason', 'unknown')}")
        
        print(f"\nAll results and models saved to {args.output_dir}")
        
    except Exception as e:
        print(f"\nError in main execution: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())