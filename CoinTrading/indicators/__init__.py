"""
Technical Indicators Module

All indicators are fully vectorized (no for-loops).

Usage:
    from CoinTrading.indicators import Momentum, RSI, EMA, add_momentum

Example:
    # Using classes
    mom = Momentum(period=20)
    df = mom.calculate(df)

    # Using convenience functions
    df = add_momentum(df, periods=[7, 20, 60])
    df = add_rsi(df, period=14)
    df = add_ema(df, periods=[7, 20, 50])
"""

from .base import Indicator, PriceIndicator, OHLCIndicator, VolumeIndicator
from .momentum import Momentum, add_momentum
from .oscillators import RSI, Noise, add_rsi, add_noise
from .trend import EMA, BollingerBands, PercentB, add_ema, add_bollinger_bands, add_percent_b

__all__ = [
    # Base classes
    'Indicator',
    'PriceIndicator',
    'OHLCIndicator',
    'VolumeIndicator',

    # Momentum
    'Momentum',
    'add_momentum',

    # Oscillators
    'RSI',
    'Noise',
    'add_rsi',
    'add_noise',

    # Trend
    'EMA',
    'BollingerBands',
    'PercentB',
    'add_ema',
    'add_bollinger_bands',
    'add_percent_b',
]
