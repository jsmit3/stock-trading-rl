"""
market/simulator.py

Simulates realistic market mechanics for order execution.

This module handles:
- Order execution with slippage
- Transaction costs and spreads
- Market gaps
- Liquidity constraints
- Circuit breakers

Author: [Your Name]
Date: March 10, 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Union, Any


class MarketSimulator:
    """
    Simulates realistic market mechanics for trading operations.
    
    This class handles:
    - Order execution with slippage
    - Transaction costs
    - Market gaps
    - Liquidity constraints
    - Circuit breakers
    """
    
    def __init__(
        self,
        transaction_cost_pct: float = 0.0015,
        slippage_std_dev: float = 0.001,
        liquidity_constraints: bool = True,
        gap_simulation: bool = True,
        max_volume_pct: float = 0.01,
        spreads: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the market simulator.
        
        Args:
            transaction_cost_pct: Base transaction cost (spread + commissions)
            slippage_std_dev: Standard deviation for slippage simulation
            liquidity_constraints: Whether to simulate liquidity constraints
            gap_simulation: Whether to simulate market gaps
            max_volume_pct: Maximum percentage of daily volume to trade
            spreads: Custom spreads for specific assets
        """
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_std_dev = slippage_std_dev
        self.liquidity_constraints = liquidity_constraints
        self.gap_simulation = gap_simulation
        self.max_volume_pct = max_volume_pct
        self.spreads = spreads or {}
        
        # Internal state
        self.last_executed_price = None
        self.market_state = "normal"  # normal, volatile, illiquid
    
    def execute_buy_order(
        self,
        shares: float,
        current_price: float,
        capital: float,
        transaction_cost_pct: Optional[float] = None,
        volatility: float = 0.01,
        volume: Optional[float] = None,
        ticker: Optional[str] = None
    ) -> Tuple[float, float, float]:
        """
        Execute a buy order with realistic market dynamics.
        
        Args:
            shares: Number of shares to buy
            current_price: Current market price
            capital: Available capital
            transaction_cost_pct: Transaction cost percentage (override default)
            volatility: Current market volatility (for slippage calculation)
            volume: Current trading volume (for liquidity constraints)
            ticker: Ticker symbol (for custom spreads)
            
        Returns:
            executed_shares: Number of shares actually executed
            execution_price: Average execution price
            total_cost: Total cost including transaction costs
        """
        # Ensure positive shares and price
        if shares <= 0 or current_price <= 0:
            return 0.0, current_price, 0.0
            
        # Use provided transaction cost or default
        transaction_cost_pct = transaction_cost_pct or self.transaction_cost_pct
        
        # Apply custom spread if available for this ticker
        if ticker and ticker in self.spreads:
            transaction_cost_pct = self.spreads[ticker]
            
        # Apply slippage based on volatility
        slippage_factor = self._calculate_slippage(volatility, "buy")
        execution_price = current_price * (1 + slippage_factor)
        
        # Calculate maximum shares based on available capital
        max_capital = capital * 0.999  # Small buffer to avoid precision issues
        max_shares = max_capital / (execution_price * (1 + transaction_cost_pct))
        
        # Apply liquidity constraints if enabled
        if self.liquidity_constraints and volume is not None:
            # Assume we can't buy more than max_volume_pct of daily volume
            max_liquidity_shares = volume * self.max_volume_pct
            max_shares = min(max_shares, max_liquidity_shares)
        
        # Execute order with the minimum of requested and maximum shares
        executed_shares = min(shares, max_shares)
        
        # Ensure executed_shares is positive
        executed_shares = max(0, executed_shares)
        
        # Calculate total cost including transaction costs
        base_cost = executed_shares * execution_price
        transaction_cost = base_cost * transaction_cost_pct
        total_cost = base_cost + transaction_cost
        
        # Update internal state
        self.last_executed_price = execution_price
        
        return executed_shares, execution_price, total_cost
    
    def execute_sell_order(
        self,
        shares: float,
        current_price: float,
        transaction_cost_pct: Optional[float] = None,
        volatility: float = 0.01,
        volume: Optional[float] = None,
        ticker: Optional[str] = None
    ) -> Tuple[float, float, float]:
        """
        Execute a sell order with realistic market dynamics.
        
        Args:
            shares: Number of shares to sell
            current_price: Current market price
            transaction_cost_pct: Transaction cost percentage (override default)
            volatility: Current market volatility (for slippage calculation)
            volume: Current trading volume (for liquidity constraints)
            ticker: Ticker symbol (for custom spreads)
            
        Returns:
            executed_shares: Number of shares actually executed
            execution_price: Average execution price
            sale_proceeds: Net proceeds after transaction costs
        """
        # Ensure positive shares and price
        if shares <= 0 or current_price <= 0:
            return 0.0, current_price, 0.0
            
        # Use provided transaction cost or default
        transaction_cost_pct = transaction_cost_pct or self.transaction_cost_pct
        
        # Apply custom spread if available for this ticker
        if ticker and ticker in self.spreads:
            transaction_cost_pct = self.spreads[ticker]
            
        # Apply slippage based on volatility
        slippage_factor = self._calculate_slippage(volatility, "sell")
        execution_price = current_price * (1 - slippage_factor)
        
        # Apply liquidity constraints if enabled
        executed_shares = shares
        if self.liquidity_constraints and volume is not None:
            # Assume we can't sell more than max_volume_pct of daily volume
            max_liquidity_shares = volume * self.max_volume_pct
            executed_shares = min(executed_shares, max_liquidity_shares)
        
        # Calculate proceeds after transaction costs
        base_proceeds = executed_shares * execution_price
        transaction_cost = base_proceeds * transaction_cost_pct
        net_proceeds = base_proceeds - transaction_cost
        
        # Update internal state
        self.last_executed_price = execution_price
        
        return executed_shares, execution_price, net_proceeds
    
    def simulate_gap(
        self,
        prev_close: float,
        volatility: float,
        overnight: bool = True,
        weekend: bool = False,
        earnings: bool = False,
        market_event: bool = False
    ) -> float:
        """
        Simulate a price gap between trading sessions.
        
        Args:
            prev_close: Previous closing price
            volatility: Current market volatility
            overnight: Whether this is an overnight gap
            weekend: Whether this is a weekend gap
            earnings: Whether earnings were released
            market_event: Whether a major market event occurred
            
        Returns:
            gap_price: Opening price after the gap
        """
        if not self.gap_simulation:
            return prev_close
            
        # Base gap parameters
        base_std = volatility * 1.5  # Base standard deviation for gaps
        
        # Adjust for different scenarios
        if weekend:
            base_std *= 1.5  # Weekend gaps are larger
        if earnings:
            base_std *= 3.0  # Earnings announcements cause larger gaps
        if market_event:
            base_std *= 5.0  # Major market events cause the largest gaps
            
        # Generate random gap
        gap_factor = np.random.normal(0, base_std)
        gap_price = prev_close * (1 + gap_factor)
        
        return gap_price
    
    def handle_circuit_breaker(
        self,
        prev_price: float,
        market_change: float,
        market_closed: bool = False
    ) -> Dict:
        """
        Handle market circuit breakers and trading halts.
        
        Args:
            prev_price: Previous price
            market_change: Percentage change in broader market
            market_closed: Whether the market is closed due to a circuit breaker
            
        Returns:
            result: Dictionary with circuit breaker status and adjusted price
        """
        # Initialize result
        result = {
            "circuit_breaker_triggered": False,
            "trading_halted": False,
            "adjusted_price": prev_price,
            "halt_duration": 0
        }
        
        # Check for market-wide circuit breakers
        if abs(market_change) > 0.07:
            # Level 1 circuit breaker (7% decline)
            result["circuit_breaker_triggered"] = True
            result["trading_halted"] = True
            result["halt_duration"] = 15  # 15 minute halt
            
        elif abs(market_change) > 0.13:
            # Level 2 circuit breaker (13% decline)
            result["circuit_breaker_triggered"] = True
            result["trading_halted"] = True
            result["halt_duration"] = 30  # 30 minute halt
            
        elif abs(market_change) > 0.20:
            # Level 3 circuit breaker (20% decline)
            result["circuit_breaker_triggered"] = True
            result["trading_halted"] = True
            result["halt_duration"] = 0  # Market closed for the day
            result["market_closed"] = True
            
        # If market is already closed due to a circuit breaker
        if market_closed:
            result["trading_halted"] = True
            result["market_closed"] = True
            
        return result
    
    def update_market_state(
        self,
        volatility: float,
        volume: float,
        avg_volume: float
    ) -> None:
        """
        Update the internal market state based on current conditions.
        
        Args:
            volatility: Current market volatility
            volume: Current trading volume
            avg_volume: Average trading volume
        """
        # Determine market volatility state
        if volatility > 0.03:  # 3% daily volatility is high
            volatility_state = "high"
        elif volatility > 0.015:
            volatility_state = "medium"
        else:
            volatility_state = "low"
            
        # Determine market liquidity state
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        if volume_ratio < 0.5:
            liquidity_state = "low"
        elif volume_ratio > 1.5:
            liquidity_state = "high"
        else:
            liquidity_state = "normal"
            
        # Update overall market state
        if volatility_state == "high" and liquidity_state == "low":
            self.market_state = "illiquid_volatile"
        elif volatility_state == "high":
            self.market_state = "volatile"
        elif liquidity_state == "low":
            self.market_state = "illiquid"
        else:
            self.market_state = "normal"
    
    def _calculate_slippage(
        self,
        volatility: float,
        order_type: str
    ) -> float:
        """
        Calculate price slippage based on market conditions.
        
        Args:
            volatility: Current market volatility
            order_type: Type of order ("buy" or "sell")
            
        Returns:
            slippage_factor: Price slippage as a percentage
        """
        # Base slippage is proportional to volatility
        base_slippage = volatility * 0.5
        
        # Add random component based on configured standard deviation
        random_component = np.random.normal(0, self.slippage_std_dev)
        
        # Combine components
        slippage_factor = base_slippage + random_component
        
        # Ensure slippage is positive
        slippage_factor = max(0, slippage_factor)
        
        # Apply market state adjustments
        if self.market_state == "volatile":
            slippage_factor *= 2.0
        elif self.market_state == "illiquid":
            slippage_factor *= 3.0
        elif self.market_state == "illiquid_volatile":
            slippage_factor *= 5.0
            
        return slippage_factor