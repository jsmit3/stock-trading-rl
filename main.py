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
Date: [Current Date]
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

# Import the trainer class
from sp100_trainer import SP100Trainer


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
    
    return parser.parse_args()


def main():
    """Main function to train and evaluate models."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Print configuration
    print("\n=== Stock Trading RL Training ===")
    print(f"Database: {args.db_path}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training timesteps: {args.timesteps}")
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
        'seed': args.seed
    }
    
    # Initialize the trainer
    trainer = SP100Trainer(
        db_path=args.db_path,
        timeframe=args.timeframe,
        output_dir=args.output_dir,
        training_params=training_params
    )
    
    # Define specific symbols to train on
    symbols_to_train = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    print(f"\nTraining on symbols: {', '.join(symbols_to_train)}")
    
    # Start training and evaluation
    print("\nStarting training process...")
    results = trainer.run(symbols=symbols_to_train)
    
    # Print final summary
    print("\n=== Training Complete ===")
    print(f"Trained on {len(results['training_results'])} symbols")
    print(f"Successfully evaluated {len([r for r in results['evaluation_results'].values() if r.get('status') == 'success'])} models")
    print(f"All results saved to {args.output_dir}/results/")


if __name__ == "__main__":
    main()