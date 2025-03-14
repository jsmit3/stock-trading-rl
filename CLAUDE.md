# Stock Trading RL Codebase Guidelines

## Build/Run Commands
- Create environment: `conda env create -f environment.yml`
- Activate environment: `conda activate stock-trading-rl`
- Main training: `python main.py --symbols AAPL --timeframe 1Day --timesteps 100000`
- Test environment: `python test_env.py`
- Run single test: `python test.py`
- Data processing: `python process-alpaca-data.py`

## Code Style Guidelines
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Imports**: group standard library, third-party, and local imports
- **Typing**: use type annotations from `typing` module (List, Dict, Optional, etc.)
- **Docstrings**: Google-style with Args and Returns sections
- **Error handling**: use try/except with specific exceptions, include descriptive messages
- **Logging**: use structured logging with appropriate levels
- **Module structure**: maintain separation of concerns between env, agent, data, and trading logic

## Project Structure
- `env/`: reinforcement learning environment
- `agent/`: RL agent implementation
- `data/`: data processing and normalization
- `trading/`: trading logic and position management
- `observation/`: observation space generation
- `reward/`: reward calculation

Remember to run `test_env.py` before committing changes to verify environment functionality.