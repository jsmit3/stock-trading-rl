"""
Agent package for the reinforcement learning trading system.

This package provides:
- Custom callbacks for curriculum learning, early stopping, metrics logging, and visualization.
- A TradingAgent implementation wrapping the SAC algorithm.

Modules:
    curriculum_callback.py: Contains CurriculumLearningCallback.
    early_stopping_callback.py: Contains TradingEarlyStoppingCallback.
    metrics_callback.py: Contains MetricsLoggerCallback.
    visualization_callback.py: Contains VisualizeTradesCallback.
    model.py: Contains TradingAgent.
"""

from agent.curriculum_callback import CurriculumLearningCallback
from agent.early_stopping_callback import TradingEarlyStoppingCallback
from agent.metrics_callback import MetricsLoggerCallback
from agent.visualization_callback import VisualizeTradesCallback
from agent.model import TradingAgent

__all__ = [
    "CurriculumLearningCallback",
    "TradingEarlyStoppingCallback",
    "MetricsLoggerCallback",
    "VisualizeTradesCallback",
    "TradingAgent",
]
