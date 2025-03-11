"""
Stock Trading Reinforcement Learning Environment

This package provides a modular, Gymnasium-compatible environment for
reinforcement learning in stock trading.

The environment simulates daily stock trading with realistic constraints:
- Single stock trading
- Long-only positions
- Risk-adjusted position sizing
- Stop-loss and take-profit mechanisms
- Handling of market gaps and transaction costs
- Maximum holding periods

Author: [Your Name]
Date: March 10, 2025
"""

from env.core import StockTradingEnv
from env.state import EnvironmentState
from env.renderer import EnvironmentRenderer

from action.interpreter import ActionInterpreter
from observation.generator import ObservationGenerator
from reward.calculator import RewardCalculator

from trading.position_manager import PositionManager
from trading.risk_manager import RiskManager

from market.simulator import MarketSimulator
from data.processor import DataProcessor

from utils.debug_utils import (
    run_debug_episodes, 
    generate_balanced_random_action,
    plot_debug_results,
    analyze_rewards,
    validate_environment
)

__version__ = '0.1.0'