"""
utils/debug_utils.py

This module provides utilities for debugging the trading environment.
Helps diagnose issues with the environment behavior, agent actions,
and reward signals.

Author: [Your Name]
Date: March 10, 2025
"""

import os
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt


def generate_balanced_random_action() -> np.ndarray:
    """
    Generate random actions that are more likely to result in valid trades.
    
    Returns:
        action: Balanced random action array
    """
    # Position size (biased toward taking positions)
    position_size = random.uniform(0.3, 1.0) if random.random() > 0.5 else 0.0
    
    # Stop loss and take profit (sensible ranges)
    stop_loss = random.uniform(0.01, 0.1) 
    take_profit = random.uniform(0.01, 0.2)
    
    # Exit signal (mostly stay in positions)
    exit_signal = 0.0 if random.random() > 0.3 else 1.0
    
    return np.array([position_size, stop_loss, take_profit, exit_signal], dtype=np.float32)


def run_debug_episodes(env, n_episodes: int = 3, verbose: bool = True) -> Dict:
    """
    Run episodes with random actions to test environment behavior.
    
    Args:
        env: Trading environment
        n_episodes: Number of episodes to run
        verbose: Whether to print debug information
        
    Returns:
        dict: Statistics from debug run
    """
    stats = {
        'episode_lengths': [],
        'episode_returns': [],
        'positions_opened': [],
        'trades_completed': [],
        'portfolio_values': [],
        'all_actions': []
    }
    
    if verbose:
        print("\n=== Running Debug Episodes ===")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        if verbose:
            print(f"\nEpisode {episode+1} start info:")
            print(f"  Date: {info['date']}")
            print(f"  Initial portfolio: ${info['portfolio_value']:.2f}")
        
        terminated = False
        truncated = False
        step_count = 0
        actions_taken = []
        
        positions_opened = 0
        trades_completed = 0
        
        while not (terminated or truncated):
            # Generate better random actions
            action = generate_balanced_random_action()
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Record action
            action_info = {
                'step': step_count,
                'position_size': action[0],
                'stop_loss': action[1],
                'take_profit': action[2],
                'exit_signal': action[3],
                'reward': reward,
                'portfolio_value': info['portfolio_value'],
                'current_position': info['current_position'],
                'drawdown': info['drawdown']
            }
            actions_taken.append(action_info)
            
            # Track when a position is opened
            if info.get('trade_executed', False) and info.get('trade_type') == 'buy':
                positions_opened += 1
                if verbose:
                    print(f"Opened position at step {step_count}, price: ${info['trade_price']:.2f}, shares: {info['trade_shares']:.2f}")
                
            # Track when a trade is completed
            if info.get('trade_completed', False):
                trades_completed += 1
                profit_loss = info.get('trade_profit', 0)
                profit_pct = info.get('trade_profit_pct', 0)
                reason = info.get('trade_reason', 'unknown')
                if verbose:
                    print(f"Closed position at step {step_count}, profit/loss: ${profit_loss:.2f} ({profit_pct:.2f}%), reason: {reason}")
            
            # Print step info
            if verbose and (step_count % 10 == 0 or terminated or truncated):
                print(f"  Step {step_count}: Reward = {reward:.4f}, Portfolio = ${info['portfolio_value']:.2f}, Drawdown = {info['drawdown']:.2%}")
            
            # Check if we're holding a position
            if verbose and info['current_position'] > 0:
                print(f"    Holding position: {info['current_position']:.2f} shares, P&L: {info['position_pnl']:.2%}")
                
            # Print diagnostics if available
            if verbose and 'diagnostics' in info and (terminated or truncated or step_count % 50 == 0):
                diagnostics = info['diagnostics']
                print(f"    Diagnostics: SL={diagnostics['stop_loss_count']}, TP={diagnostics['take_profit_count']}, MH={diagnostics['max_holding_count']}, EX={diagnostics['exit_signal_count']}")
        
        # Record episode statistics
        stats['episode_lengths'].append(step_count)
        stats['episode_returns'].append(info['portfolio_value'] / info['initial_capital'] - 1)
        stats['positions_opened'].append(positions_opened)
        stats['trades_completed'].append(trades_completed)
        stats['portfolio_values'].append(info['portfolio_value'])
        stats['all_actions'].extend(actions_taken)
        
        # Print episode summary
        if verbose:
            print(f"\nEpisode {episode+1} ended after {step_count} steps")
            print(f"  Final portfolio: ${info['portfolio_value']:.2f} ({(info['portfolio_value']/info['initial_capital']-1)*100:.2f}%)")
            print(f"  Positions opened: {positions_opened}, Trades completed: {trades_completed}")
            
            # Print termination reason if available
            if 'termination_reason' in info:
                print(f"  Termination reason: {info['termination_reason']}")
            elif truncated:
                print("  Episode was truncated (reached end of data)")
            else:
                print("  Episode ended normally")
    
    # Print overall statistics
    if verbose:
        print("\nOverall Debug Statistics:")
        print(f"  Average episode length: {np.mean(stats['episode_lengths']):.1f} steps")
        print(f"  Average return: {np.mean(stats['episode_returns'])*100:.2f}%")
        print(f"  Average positions opened: {np.mean(stats['positions_opened']):.1f}")
        print(f"  Average trades completed: {np.mean(stats['trades_completed']):.1f}")
    
    return stats


def plot_debug_results(stats: Dict) -> None:
    """
    Plot statistics from debug runs.
    
    Args:
        stats: Statistics from debug runs
    """
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot episode lengths
    axs[0, 0].bar(range(1, len(stats['episode_lengths'])+1), stats['episode_lengths'])
    axs[0, 0].set_title('Episode Lengths')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Steps')
    
    # Plot episode returns
    axs[0, 1].bar(range(1, len(stats['episode_returns'])+1), [r*100 for r in stats['episode_returns']])
    axs[0, 1].set_title('Episode Returns (%)')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Return (%)')
    
    # Plot positions and trades
    x = range(1, len(stats['positions_opened'])+1)
    axs[1, 0].bar(x, stats['positions_opened'], label='Positions Opened')
    axs[1, 0].bar(x, stats['trades_completed'], alpha=0.7, label='Trades Completed')
    axs[1, 0].set_title('Trading Activity')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].legend()
    
    # Plot action statistics if available
    if stats['all_actions']:
        actions_df = pd.DataFrame(stats['all_actions'])
        actions_df[['position_size', 'stop_loss', 'take_profit', 'exit_signal']].hist(
            bins=20, ax=axs[1, 1], layout=(2, 2), figsize=(6, 5)
        )
        axs[1, 1].set_title('Action Distributions')
    
    plt.tight_layout()
    plt.show()


def analyze_rewards(stats: Dict) -> None:
    """
    Analyze reward signals from debug runs.
    
    Args:
        stats: Statistics from debug runs
    """
    if not stats['all_actions']:
        print("No actions recorded for reward analysis")
        return
    
    # Create DataFrame for analysis
    actions_df = pd.DataFrame(stats['all_actions'])
    
    # Analyze reward distribution
    print("\nReward Analysis:")
    print(f"  Mean reward: {actions_df['reward'].mean():.6f}")
    print(f"  Median reward: {actions_df['reward'].median():.6f}")
    print(f"  Min reward: {actions_df['reward'].min():.6f}")
    print(f"  Max reward: {actions_df['reward'].max():.6f}")
    print(f"  Reward standard deviation: {actions_df['reward'].std():.6f}")
    print(f"  Percentage of positive rewards: {(actions_df['reward'] > 0).mean()*100:.1f}%")
    
    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.subplot(2, 1, 1)
    plt.hist(actions_df['reward'], bins=50)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    
    # Time series
    plt.subplot(2, 1, 2)
    plt.plot(actions_df['reward'])
    plt.title('Reward Time Series')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.show()


def validate_environment(env) -> Dict:
    """
    Perform validation checks on the environment.
    
    Args:
        env: Trading environment
        
    Returns:
        dict: Validation results
    """
    results = {
        'observation_space_valid': True,
        'action_space_valid': True,
        'reset_valid': True,
        'step_valid': True,
        'issues': []
    }
    
    # Check observation and action spaces
    try:
        assert hasattr(env, 'observation_space'), "Environment missing observation_space"
        assert hasattr(env, 'action_space'), "Environment missing action_space"
    except AssertionError as e:
        results['issues'].append(str(e))
        if "observation_space" in str(e):
            results['observation_space_valid'] = False
        if "action_space" in str(e):
            results['action_space_valid'] = False
    
    # Test reset
    try:
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray), "Reset should return observation as numpy array"
        assert isinstance(info, dict), "Reset should return info as dictionary"
    except Exception as e:
        results['reset_valid'] = False
        results['issues'].append(f"Reset error: {str(e)}")
    
    # Test step
    if results['reset_valid']:
        try:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(next_obs, np.ndarray), "Step should return observation as numpy array"
            assert isinstance(reward, (int, float)), "Step should return reward as number"
            assert isinstance(terminated, bool), "Step should return terminated as boolean"
            assert isinstance(truncated, bool), "Step should return truncated as boolean"
            assert isinstance(info, dict), "Step should return info as dictionary"
        except Exception as e:
            results['step_valid'] = False
            results['issues'].append(f"Step error: {str(e)}")
    
    # Print validation results
    all_valid = all(value is True for key, value in results.items() if key != 'issues')
    
    print("\nEnvironment Validation Results:")
    print(f"  All checks passed: {all_valid}")
    
    if results['issues']:
        print("\nIssues found:")
        for issue in results['issues']:
            print(f"  - {issue}")
    else:
        print("  No issues found.")
    
    return results