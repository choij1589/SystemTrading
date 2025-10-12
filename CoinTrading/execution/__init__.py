"""
Execution Module

Live trading interface with risk management and order execution.

Usage:
    from CoinTrading.execution import Trader, OrderManager, RiskManager

Example - Paper Trading:
    >>> from CoinTrading.execution import Trader, OrderManager, RiskManager, RiskLimits
    >>> from CoinTrading.strategy import MarketTimingStrategy
    >>> from CoinTrading.data import DataLoader
    >>>
    >>> # Setup components
    >>> strategy = MarketTimingStrategy(indicator='mom7')
    >>> data_loader = DataLoader()
    >>> order_manager = OrderManager(paper_trading=True)
    >>> risk_manager = RiskManager(limits=RiskLimits())
    >>>
    >>> # Create trader
    >>> trader = Trader(
    ...     strategy=strategy,
    ...     data_loader=data_loader,
    ...     order_manager=order_manager,
    ...     risk_manager=risk_manager,
    ...     initial_capital=10000.0,
    ...     dry_run=True
    ... )
    >>>
    >>> # Run backtest
    >>> equity_curve = trader.run_backtest(data, start_date='2021-01-01')

Example - Live Trading (with caution):
    >>> # Setup with live Binance client
    >>> from binance import Client
    >>> client = Client(api_key, api_secret)
    >>>
    >>> order_manager = OrderManager(
    ...     binance_client=client,
    ...     paper_trading=False
    ... )
    >>>
    >>> trader = Trader(
    ...     strategy=strategy,
    ...     data_loader=data_loader,
    ...     order_manager=order_manager,
    ...     risk_manager=risk_manager,
    ...     initial_capital=10000.0,
    ...     dry_run=False  # CAUTION: Real money!
    ... )
    >>>
    >>> trader.run_live()  # Start live trading loop
"""

from .order_manager import (
    OrderManager,
    Order,
    OrderSide,
    OrderType,
    OrderStatus
)
from .risk_manager import (
    RiskManager,
    RiskLimits
)
from .trader import Trader

__all__ = [
    # Order Management
    'OrderManager',
    'Order',
    'OrderSide',
    'OrderType',
    'OrderStatus',

    # Risk Management
    'RiskManager',
    'RiskLimits',

    # Trading
    'Trader',
]
