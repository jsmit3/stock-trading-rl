# Stock Trading Reinforcement Learning Framework

A modular, Gymnasium-compatible environment for reinforcement learning in stock trading. This framework simulates daily stock trading with realistic constraints including risk management, position sizing, and market dynamics.

## 📚 Overview

This framework provides:
- A flexible simulation environment for stock trading
- Components for realistic market simulation
- Risk management and position sizing
- Data processing with technical indicators
- Observation generation for RL agents
- Integration with Stable Baselines 3 for training

## 🏗️ Architecture

The framework is built with modularity in mind, splitting responsibilities into specialized components. Below is a breakdown of the main modules and their dependencies.

### Core Environment Module (`env/`)

| Component | File | Purpose | Dependencies | 
|-----------|------|---------|--------------|
| `StockTradingEnv` | `env/core.py` | Main Gymnasium environment that simulates stock trading with realistic constraints | Depends on components from all other modules |
| `EnvironmentState` | `env/state.py` | Tracks the state of the trading environment (portfolio, positions, history) | None |
| `EnvironmentRenderer` | `env/renderer.py` | Visualizes the trading environment state in console and graphical formats | `matplotlib` |

### Action Module (`action/`)

| Component | File | Purpose | Dependencies |
|-----------|------|---------|--------------|
| `ActionInterpreter` | `action/interpreter.py` | Converts raw agent actions into trading decisions (position sizing, stop-loss, etc.) | `numpy`, `pandas` |

Key methods in `ActionInterpreter`:
- `interpret_action`: Translates raw agent actions to concrete trading parameters
- `_interpret_position_size`: Calculates position size based on agent's action
- `_interpret_stop_loss`: Determines stop-loss level
- `_interpret_take_profit`: Calculates take-profit targets with exponential scaling
- `_adjust_position_size_for_risk`: Risk-adjusts position sizes

### Observation Module (`observation/`)

| Component | File | Purpose | Dependencies |
|-----------|------|---------|--------------|
| `ObservationGenerator` | `observation/generator.py` | Creates comprehensive state observations for the RL agent | `numpy`, `pandas`, `talib`, `sklearn` |

Key methods in `ObservationGenerator`:
- `generate_observation`: Main method that assembles the complete observation vector
- `_get_price_features`: Extracts price-related features
- `_get_volume_features`: Extracts volume-related features
- `_get_trend_indicators`, `_get_momentum_indicators`, `_get_volatility_indicators`: Generate technical indicators
- `_get_position_features`: Encodes current position information
- `_get_account_features`: Encodes portfolio and account status
- `_get_time_features`: Encodes time-based features (day of week, month)

### Reward Module (`reward/`)

| Component | File | Purpose | Dependencies |
|-----------|------|---------|--------------|
| `RewardCalculator` | `reward/calculator.py` | Calculates rewards for the agent based on multiple factors | `numpy` |

Key methods in `RewardCalculator`:
- `calculate_reward`: Implements the reward function combining returns, risk, and other factors
- `smooth_reward`: Applies exponential smoothing to rewards
- `get_trade_statistics`: Calculates statistics about recent trades

### Market Module (`market/`)

| Component | File | Purpose | Dependencies |
|-----------|------|---------|--------------|
| `MarketSimulator` | `market/simulator.py` | Simulates realistic market mechanics for trade execution | `numpy` |

Key methods in `MarketSimulator`:
- `execute_buy_order`: Simulates buy order with slippage and costs
- `execute_sell_order`: Simulates sell order with slippage and costs
- `simulate_gap`: Simulates price gaps between trading sessions
- `handle_circuit_breaker`: Handles market circuit breakers and trading halts
- `update_market_state`: Updates internal market state based on conditions

### Trading Module (`trading/`)

| Component | File | Purpose | Dependencies |
|-----------|------|---------|--------------|
| `PositionManager` | `trading/position_manager.py` | Manages opening and closing of trading positions | `MarketSimulator`, `RiskManager` |
| `RiskManager` | `trading/risk_manager.py` | Handles risk management strategies for position sizing | `numpy`, `pandas` |

Key methods in `PositionManager`:
- `open_position`: Opens new trading positions
- `close_position`: Closes existing trading positions

Key methods in `RiskManager`:
- `adjust_position_size`: Adjusts position size based on risk parameters
- `calculate_position_size`: Calculates position size using fixed-risk principles
- `update_trade_result`: Updates internal state based on trade results
- `get_stop_loss_for_gap_risk`: Calculates adjusted stop loss for gap risk

### Data Module (`data/`)

| Component | File | Purpose | Dependencies |
|-----------|------|---------|--------------|
| `DataProcessor` | `data/processor.py` | Processes and prepares market data with technical indicators | `numpy`, `pandas`, `talib`, `sklearn` |
| `DataNormalizer` | `data/normalizer.py` | Handles data normalization strategies for neural network inputs | `numpy`, `pandas`, `sklearn` |

Key methods in `DataProcessor`:
- `process_data`: Main method for processing raw price data
- `_add_missing_columns`: Adds missing OHLCV columns with approximated values
- `_calculate_technical_indicators`: Calculates various technical indicators
- `normalize_data`: Normalizes data for neural network inputs

Key methods in `DataNormalizer`:
- `fit`: Fits normalizer to data
- `transform`: Transforms data using fitted normalizer
- `inverse_transform`: Inverts normalization

### Agent Module (`agent/`)

| Component | File | Purpose | Dependencies |
|-----------|------|---------|--------------|
| `TradingAgent` | `agent/model.py` | Agent implementation using the SAC algorithm | `torch`, `stable_baselines3` |
| `CurriculumLearningCallback` | `agent/curriculum_callback.py` | Implements curriculum learning by advancing difficulty levels | `stable_baselines3` |
| `MetricsLoggerCallback` | `agent/metrics_callback.py` | Logs detailed trading performance metrics | `stable_baselines3`, `os` |
| `TradingEarlyStoppingCallback` | `agent/early_stopping_callback.py` | Implements early stopping based on trading metrics | `stable_baselines3` |
| `VisualizeTradesCallback` | `agent/visualization_callback.py` | Creates visualizations of trading activity | `stable_baselines3`, `os`, `matplotlib` |

Key methods in `TradingAgent`:
- `train`: Trains the agent with callbacks
- `predict`: Predicts actions for given observations
- `from_saved`: Creates agent from saved model

### Utility Functions (`utils/`)

| Function | File | Purpose | Dependencies |
|----------|------|---------|--------------|
| `run_debug_episodes` | `utils/debug_utils.py` | Runs debug episodes with random actions | `numpy`, `random` |
| `plot_debug_results` | `utils/debug_utils.py` | Plots statistics from debug runs | `matplotlib`, `pandas` |
| `analyze_rewards` | `utils/debug_utils.py` | Analyzes reward signals from debug runs | `pandas`, `matplotlib` |
| `validate_environment` | `utils/debug_utils.py` | Performs validation checks on the environment | `numpy` |
| `generate_balanced_random_action` | `utils/debug_utils.py` | Generates better random actions for testing | `numpy`, `random` |

## 🔄 Data Flow

The framework follows this general flow:

1. **Data Processing**: `DataProcessor` prepares market data and calculates indicators
2. **Environment Step**: 
   - Agent produces an action
   - `ActionInterpreter` translates raw actions to trading parameters
   - `PositionManager` and `MarketSimulator` execute the trades
   - `EnvironmentState` tracks the portfolio and positions
   - `RewardCalculator` generates rewards
   - `ObservationGenerator` creates the next observation

## 🚀 Training Pipeline

The training process involves:

1. Load and process market data with `DataProcessor`
2. Create the `StockTradingEnv` environment
3. Initialize the `TradingAgent`
4. Set up callbacks for monitoring and visualization
5. Train the agent on historical data
6. Evaluate performance on out-of-sample data

## 📊 Key Performance Metrics

The framework tracks several key metrics:
- Total return and Sharpe ratio
- Win rate and profit/loss ratio
- Maximum drawdown
- Number of trades and position utilization

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.9+
- TA-Lib (technical analysis library)
- PyTorch
- Stable Baselines 3
- Gymnasium

### Installation

```bash
# Create a new conda environment
conda create -n stock-trading-rl python=3.9
conda activate stock-trading-rl

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Framework

### Testing the Environment

```python
from utils.debug_utils import run_debug_episodes, plot_debug_results
from env.core import StockTradingEnv
from data.processor import DataProcessor

# Load and process data
data = load_sample_data()  # Your data loading function
processor = DataProcessor(data)
processed_data = processor.process_data()

# Create environment
env = StockTradingEnv(price_data=processed_data)

# Run debug episodes
stats = run_debug_episodes(env, n_episodes=3)
plot_debug_results(stats)
```

### Training an Agent

```python
from agent.model import TradingAgent
from agent.curriculum_callback import CurriculumLearningCallback
from agent.metrics_callback import MetricsLoggerCallback

# Create environment (as above)
env = StockTradingEnv(price_data=processed_data)

# Create agent
agent = TradingAgent(env=env)

# Create callbacks
curriculum_callback = CurriculumLearningCallback(env)
metrics_callback = MetricsLoggerCallback(eval_env=eval_env)

# Train agent
agent.train(total_timesteps=100000, callback_list=[curriculum_callback, metrics_callback])
```

## 📝 License

[Specify your license here]
