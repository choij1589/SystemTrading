"""
Strategy Module

Provides trading strategy implementations for cryptocurrency trading.

Usage:
    from CoinTrading.strategy import MomentumSimpleStrategy, MarketTimingStrategy

Example:
    # Simple momentum strategy (Step 1)
    strategy = MomentumSimpleStrategy(
        indicator='mom20',
        long_top_n=5,
        short_bottom_n=5
    )

    # Market timing strategy (Step 3)
    strategy = MarketTimingStrategy(
        indicator='mom7',
        long_top_n=4,
        short_bottom_n=8,
        apply_noise_filter=True
    )

    # Use with backtesting engine
    from CoinTrading.backtesting import BacktestEngine
    engine = BacktestEngine(data)
    equity_curve = engine.run(strategy.get_weights)
"""

from .base import Strategy, LongShortStrategy, LongOnlyStrategy
from .portfolio import (
    select_by_volume,
    filter_by_noise,
    rank_by_indicator,
    allocate_equal_weight,
    allocate_long_short,
    calculate_leverage
)
from .momentum_simple import MomentumSimpleStrategy, MomentumLongOnlyStrategy
from .market_timing import MarketTimingStrategy
from .regime_based import RegimeBasedStrategy
from .rebalancing import RebalancingStrategy

__all__ = [
    # Base classes
    'Strategy',
    'LongShortStrategy',
    'LongOnlyStrategy',

    # Portfolio utilities
    'select_by_volume',
    'filter_by_noise',
    'rank_by_indicator',
    'allocate_equal_weight',
    'allocate_long_short',
    'calculate_leverage',

    # Strategy implementations
    'MomentumSimpleStrategy',
    'MomentumLongOnlyStrategy',
    'MarketTimingStrategy',
    'RegimeBasedStrategy',
    'RebalancingStrategy',
]
