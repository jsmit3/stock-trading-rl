#!/usr/bin/env python3
"""
train_multi_stock.py

Script for training a single reinforcement learning model across multiple stocks.

This script:
1. Sets up the MultiStockTrainer with configuration parameters
2. Loads data for multiple stock symbols
3. Creates a multi-stock trading environment
4. Trains a single model across all stocks
5. Evaluates the model on individual stocks
6. Prints a summary of results

Usage:
    python train_multi_stock.py [arguments]

Author: [Your Name]
Date: March 13, 2025
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# Import components
from multi_stock_trainer import MultiStockTrainer
from observation.generator import ObservationGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a multi-stock RL agent"
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
        default="multi_stock_output",
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--symbols-file",
        type=str,
        help="Optional file with symbols to train on (one per line)"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Specific symbols to train on (space-separated)"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Number of timesteps for training"
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


def get_symbols_from_file(symbols_file):
    """Read symbols from a text file."""
    if not Path(symbols_file).exists():
        print(f"Symbols file not found: {symbols_file}")
        return []
    
    with open(symbols_file, 'r') as f:
        symbols = [line.strip() for line in f.readlines()]
    
    return [s for s in symbols if s and not s.startswith('#')]


def main():
    """Main function to train and evaluate models."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir
        if not output_dir.endswith(timestamp):
            output_dir = f"{output_dir}_{timestamp}"
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Print configuration
        print("\n=== Multi-Stock Trading RL Training ===")
        print(f"Database: {args.db_path}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Output directory: {output_dir}")
        print(f"Training timesteps: {args.timesteps}")
        print(f"Observation dimension: {args.obs_dim}")
        print(f"Symbol embedding dimension: {args.symbol_dim}")
        print(f"Random seed: {args.seed}")
        
        # Define training parameters
        training_params = {
            'initial_capital': 100000.0,
            'window_size': 20,
            'max_position_pct': 0.25,
            'transaction_cost_pct': 0.0015,
            'curriculum_level': 1,
            'learning_rate': 3e-5,
            'buffer_size': 200000,
            'batch_size': 256,
            'total_timesteps': args.timesteps,
            'seed': args.seed,
            'observation_dim': args.obs_dim,
            'symbol_feature_dim': args.symbol_dim
        }
        
        # Initialize the trainer
        trainer = MultiStockTrainer(
            db_path=args.db_path,
            timeframe=args.timeframe,
            output_dir=output_dir,
            training_params=training_params,
            observation_dim=args.obs_dim,
            symbol_feature_dim=args.symbol_dim
        )
        
        # Determine symbols to train on
        symbols_to_train = []
        
        # First priority: Command line symbols
        if args.symbols:
            symbols_to_train = args.symbols
            print(f"Using symbols from command line: {', '.join(symbols_to_train)}")
        
        # Second priority: Symbols file
        elif args.symbols_file:
            symbols_to_train = get_symbols_from_file(args.symbols_file)
            print(f"Loaded {len(symbols_to_train)} symbols from {args.symbols_file}")
        
        # Default: let the trainer use its default symbols
        else:
            print("No symbols specified. Using trainer defaults.")
        
        # Start training and evaluation
        print("\nStarting multi-stock training process...")
        results = trainer.run(symbols=symbols_to_train, symbols_file=args.symbols_file)
        
        # Print final summary
        print("\n=== Training Complete ===")
        
        # Check training results
        if results['training_results'].get('status') == 'success':
            # If single model training succeeded
            print("Multi-stock model training successful!")
            
            # Count successful evaluations
            successful_evals = sum(1 for r in results['evaluation_results'].values() 
                                 if r.get('status') == 'success')
            print(f"Successfully evaluated on {successful_evals}/{len(results['evaluation_results'])} symbols")
            
            # Print details of evaluations
            if successful_evals > 0:
                print("\nEvaluation results by symbol:")
                
                # Sort by Sharpe ratio
                sorted_evals = sorted(
                    [(s, r) for s, r in results['evaluation_results'].items() if r.get('status') == 'success'],
                    key=lambda x: x[1].get('sharpe_ratio', -999),
                    reverse=True
                )
                
                for symbol, result in sorted_evals:
                    print(f"  {symbol}: Return={result.get('total_return', 0):.2f}%, " +
                         f"Sharpe={result.get('sharpe_ratio', 0):.2f}, " +
                         f"Drawdown={result.get('max_drawdown', 0):.2f}%")
        else:
            # If training failed
            print("Multi-stock model training failed.")
            print(f"Reason: {results['training_results'].get('reason', 'unknown')}")
        
        print(f"\nAll results saved to {output_dir}/results/")
        
    except Exception as e:
        print(f"\nError in main execution: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())