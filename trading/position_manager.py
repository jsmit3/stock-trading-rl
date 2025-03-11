"""
trading/position_manager.py

Manages the opening and closing of trading positions.

This module handles:
- Position sizing
- Order execution
- Trade tracking
- P&L calculation

Author: [Your Name]
Date: March 10, 2025
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
from market.simulator import MarketSimulator
from trading.risk_manager import RiskManager


class PositionManager:
    """
    Manages the opening, tracking, and closing of trading positions.
    
    This class interacts with the market simulator to execute trades
    and keeps track of position details.
    """
    
    def __init__(
        self,
        market_simulator: MarketSimulator,
        risk_manager: RiskManager,
        debug_mode: bool = False
    ):
        """
        Initialize the position manager.
        
        Args:
            market_simulator: Market simulator for executing trades
            risk_manager: Risk manager for position sizing
            debug_mode: Whether to print debug information
        """
        self.market_simulator = market_simulator
        self.risk_manager = risk_manager
        self.debug_mode = debug_mode
    
    def open_position(
        self,
        state,
        position_size_pct: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        current_price: float,
        current_step: int,
        dates: List
    ) -> Dict[str, Any]:
        """
        Open a new trading position.
        
        Args:
            state: Current environment state
            position_size_pct: Position size as percentage of available capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            current_price: Current market price
            current_step: Current step in the environment
            dates: List of dates
            
        Returns:
            trade_info: Information about the trade
        """
        # Ensure positive position size and valid price
        if position_size_pct <= 0 or current_price <= 0:
            return None
            
        # Calculate position size in shares
        capital_to_use = state.cash_balance * position_size_pct
        
        # Ensure we have enough capital to open a position
        if capital_to_use <= 0:
            if self.debug_mode:
                print(f"Not enough capital to open position: {capital_to_use}")
            return None
            
        shares_to_buy = capital_to_use / current_price
        
        # Ensure shares to buy is positive
        if shares_to_buy <= 0:
            if self.debug_mode:
                print(f"Invalid shares to buy: {shares_to_buy}")
            return None
            
        # Execute the trade through market simulator
        try:
            executed_shares, executed_price, actual_cost = self.market_simulator.execute_buy_order(
                shares=shares_to_buy,
                current_price=current_price,
                capital=state.cash_balance
            )
        except Exception as e:
            if self.debug_mode:
                print(f"Error executing buy order: {e}")
            return None
        
        # Ensure we actually executed shares
        if executed_shares <= 0:
            if self.debug_mode:
                print(f"No shares executed in buy order")
            return None
            
        # Update state
        state.cash_balance -= actual_cost
        state.current_position = executed_shares
        state.position_entry_price = executed_price
        state.position_entry_step = current_step
        state.stop_loss_pct = stop_loss_pct
        state.take_profit_pct = take_profit_pct
        
        # Record the position
        state.positions_history.append(executed_shares)
        
        # Prepare trade information
        trade_info = {
            'trade_executed': True,
            'trade_type': 'buy',
            'trade_shares': executed_shares,
            'trade_price': executed_price,
            'trade_cost': actual_cost,
            'trade_date': dates[current_step] if current_step < len(dates) else dates[-1],
            'trade_completed': False,
            'stop_loss': state.position_entry_price * (1 - stop_loss_pct),
            'take_profit': state.position_entry_price * (1 + take_profit_pct)
        }
        
        # Add to trade history
        state.trade_history.append(trade_info)
        
        if self.debug_mode:
            print(f"Opened position at step {current_step}, date {dates[current_step if current_step < len(dates) else -1]}")
            print(f"Shares: {executed_shares:.2f}, Entry price: ${executed_price:.2f}")
            print(f"Stop loss: ${state.position_entry_price * (1 - stop_loss_pct):.2f} ({stop_loss_pct:.2%})")
            print(f"Take profit: ${state.position_entry_price * (1 + take_profit_pct):.2f} ({take_profit_pct:.2%})")
        
        return trade_info
    
    def close_position(
        self,
        state,
        current_price: float,
        reason: str,
        current_step: int,
        dates: List
    ) -> Dict[str, Any]:
        """
        Close the current trading position.
        
        Args:
            state: Current environment state
            current_price: Current market price
            reason: Reason for closing the position
            current_step: Current step in the environment
            dates: List of dates
            
        Returns:
            trade_info: Information about the trade
        """
        if state.current_position == 0:
            return None
            
        # Execute the trade through market simulator
        try:
            executed_shares, executed_price, sale_value = self.market_simulator.execute_sell_order(
                shares=state.current_position,
                current_price=current_price
            )
        except Exception as e:
            if self.debug_mode:
                print(f"Error executing sell order: {e}")
            # Force position closure even if market simulator fails
            executed_shares = state.current_position
            executed_price = current_price
            sale_value = executed_shares * executed_price * (1 - self.market_simulator.transaction_cost_pct)
        
        # Calculate P&L
        cost_basis = state.position_entry_price * executed_shares
        profit_loss = sale_value - cost_basis
        state.position_pnl = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
        state.total_pnl += profit_loss
        
        # Update state
        state.cash_balance += sale_value
        previous_position = state.current_position
        entry_price = state.position_entry_price
        entry_date = dates[state.position_entry_step] if state.position_entry_step < len(dates) else dates[-1]
        
        state.current_position = 0
        state.position_entry_price = 0
        state.stop_loss_pct = 0
        state.take_profit_pct = 0
        
        # Record the position closure
        state.positions_history.append(0)
        
        # Prepare trade information
        trade_info = {
            'trade_executed': True,
            'trade_type': 'sell',
            'trade_shares': executed_shares,
            'trade_price': executed_price,
            'trade_value': sale_value,
            'trade_date': dates[current_step] if current_step < len(dates) else dates[-1],
            'trade_profit': profit_loss,
            'trade_profit_pct': state.position_pnl,
            'trade_completed': True,
            'trade_reason': reason,
            'trade_duration': current_step - state.position_entry_step,
            'entry_date': entry_date,
            'entry_price': entry_price
        }
        
        # Add to trade history
        state.trade_history.append(trade_info)
        
        # Update risk manager with trade result
        self.risk_manager.update_trade_result(state.position_pnl / 100)  # Convert percentage to decimal
        
        if self.debug_mode:
            print(f"Closed position at step {current_step}, date {dates[current_step if current_step < len(dates) else -1]}")
            print(f"Reason: {reason}, Profit/Loss: ${profit_loss:.2f} ({state.position_pnl:.2%})")
            print(f"Duration: {current_step - state.position_entry_step} days")
        
        return trade_info