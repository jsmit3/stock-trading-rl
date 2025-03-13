#!/usr/bin/env python3
"""
Main script for training and evaluating reinforcement learning agents on stock data.

This script:
1. Sets up the SP100Trainer with configuration parameters
2. Trains models on selected symbols
3. Evaluates the trained models
4. Prints a summary of results

Usage:
    python main.py

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
from sp100_trainer import SP100Trainer
from observation.generator import ObservationGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agents on stock data"
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
        default=f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=325,
        help="Fixed observation dimension"
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


def create_observation_generator(window_size=20, fixed_dim=325):
    """Create a consistently configured observation generator with fixed dimension."""
    observation_generator = ObservationGenerator(
        window_size=window_size,
        include_sentiment=False,
        use_pca=False,
        include_market_context=False,
        fixed_dim=fixed_dim  # Pass fixed dimension directly to constructor
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
    
    print(f"Observation generator created with fixed dimension: {observation_generator.observation_dim}")
    
    return observation_generator


def main():
    """Main function to train and evaluate models."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Print configuration
        print("\n=== Stock Trading RL Training ===")
        print(f"Database: {args.db_path}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Output directory: {args.output_dir}")
        print(f"Training timesteps: {args.timesteps}")
        print(f"Random seed: {args.seed}")
        print(f"Observation dimension: {args.obs_dim}")
        
        # Create a consistent observation generator
        observation_generator = create_observation_generator(
            window_size=20,
            fixed_dim=args.obs_dim
        )
        
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
            'observation_dim': args.obs_dim  # Add observation dimension parameter
        }
        
        # Initialize the trainer with the observation generator
        trainer = SP100Trainer(
            db_path=args.db_path,
            timeframe=args.timeframe,
            output_dir=args.output_dir,
            training_params=training_params,
            observation_generator=observation_generator  # Pass the observation generator
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
        
        # Default symbols
        else:
            symbols_to_train = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
            print(f"Using default symbols: {', '.join(symbols_to_train)}")
        
        if not symbols_to_train:
            print("No symbols to train on. Exiting.")
            return
        
        print(f"\nTraining on {len(symbols_to_train)} symbols: {', '.join(symbols_to_train[:5])}" + 
              (f"..." if len(symbols_to_train) > 5 else ""))
        
        # Start training and evaluation
        print("\nStarting training process...")
        results = trainer.run(symbols=symbols_to_train)
        
        # Print final summary
        print("\n=== Training Complete ===")
        print(f"Trained on {len(results['training_results'])} symbols")
        
        successful_evals = len([r for r in results['evaluation_results'].values() 
                              if r.get('status') == 'success'])
        print(f"Successfully evaluated {successful_evals} models")
        
        # Print details of successful models
        if successful_evals > 0:
            print("\nSuccessful models:")
            for symbol, result in results['evaluation_results'].items():
                if result.get('status') == 'success':
                    print(f"  {symbol}: Return={result.get('total_return', 0):.2f}%, " +
                          f"Sharpe={result.get('sharpe_ratio', 0):.2f}")
        
        print(f"\nAll results saved to {args.output_dir}/results/")
        
    except Exception as e:
        print(f"\nError in main execution: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())