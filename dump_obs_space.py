
import json
import numpy as np
import gymnasium as gym

def dump_observation_space(env: gym.Env, filename: str) -> None:
    """
    Dump the observation space of a Gymnasium environment to a JSON file.

    Args:
        env: The Gymnasium environment.
        filename: Path to the output JSON file.
    """
    obs_space = env.observation_space

    # Extract details from the observation space
    # Here we assume it's a Box space; adjust if using a different space type
    obs_details = {
        "shape": obs_space.shape,
        "low": obs_space.low.tolist() if isinstance(obs_space.low, np.ndarray) else obs_space.low,
        "high": obs_space.high.tolist() if isinstance(obs_space.high, np.ndarray) else obs_space.high,
        "dtype": str(obs_space.dtype)
    }
    
    # Write the details to a JSON file
    with open(filename, "w") as f:
        json.dump(obs_details, f, indent=4)
    print(f"Observation space dumped to {filename}")


if __name__ == "__main__":
    # Example: create your environment instance
    # Replace 'StockTradingEnv' with your environment class and supply any required arguments
    from env.core import StockTradingEnv
    import pandas as pd

    # For demonstration, load some sample price data; adjust as needed
    # Here, we create a dummy DataFrame with OHLCV columns
    sample_data = pd.DataFrame({
        "open": np.random.rand(100) * 100,
        "high": np.random.rand(100) * 100,
        "low": np.random.rand(100) * 100,
        "close": np.random.rand(100) * 100,
        "volume": np.random.randint(1000, 5000, size=100)
    }, index=pd.date_range("2025-01-01", periods=100))

    env = StockTradingEnv(price_data=sample_data)
    dump_observation_space(env, "observation_space.json")
