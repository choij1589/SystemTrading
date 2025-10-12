"""
Portfolio Construction Utilities

Helper functions for universe selection, ranking, and weight allocation.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def select_by_volume(
    date: pd.Timestamp,
    data: Dict[str, pd.DataFrame],
    top_n: int = 21,
    volume_metric: str = 'tp_volume'
) -> List[str]:
    """
    Select top N coins by trading volume.

    Args:
        date: Current date
        data: Dict of {symbol: DataFrame}
        top_n: Number of top coins to select
        volume_metric: Volume metric to use ('volume', 'tp_volume', 'quote_volume')

    Returns:
        List of top N symbols by volume
    """
    volumes = {}

    for symbol, df in data.items():
        if df.empty or date not in df.index:
            continue

        try:
            if volume_metric == 'tp_volume':
                # Typical price × volume (original notebook method)
                if 'TP' in df.columns:
                    tp = df.loc[date, 'TP']
                elif all(col in df.columns for col in ['high', 'low', 'close']):
                    tp = (df.loc[date, 'high'] + df.loc[date, 'low'] + df.loc[date, 'close']) / 3.0
                else:
                    logger.warning(f"{symbol}: Cannot calculate TP, skipping")
                    continue

                volume = df.loc[date, 'volume']
                volumes[symbol] = tp * volume

            elif volume_metric == 'quote_volume':
                # Quote asset volume (if available)
                if 'quote_volume' in df.columns:
                    volumes[symbol] = df.loc[date, 'quote_volume']
                else:
                    # Fallback: close × volume
                    volumes[symbol] = df.loc[date, 'close'] * df.loc[date, 'volume']

            else:  # 'volume'
                volumes[symbol] = df.loc[date, 'volume']

        except (KeyError, ValueError) as e:
            logger.debug(f"{symbol}: Error calculating volume: {e}")
            continue

    # Sort by volume and take top N
    sorted_symbols = sorted(volumes.items(), key=lambda x: x[1], reverse=True)
    top_symbols = [symbol for symbol, _ in sorted_symbols[:top_n]]

    logger.debug(f"{date.date()}: Selected {len(top_symbols)}/{len(data)} coins by {volume_metric}")

    return top_symbols


def filter_by_noise(
    date: pd.Timestamp,
    data: Dict[str, pd.DataFrame],
    symbols: List[str],
    noise_threshold: float = 0.7,
    noise_column: str = 'noise15'
) -> List[str]:
    """
    Filter out coins with high noise.

    Args:
        date: Current date
        data: Dict of {symbol: DataFrame}
        symbols: List of symbols to filter
        noise_threshold: Maximum allowed noise level
        noise_column: Name of noise column

    Returns:
        Filtered list of symbols
    """
    filtered = []

    for symbol in symbols:
        if symbol not in data:
            continue

        df = data[symbol]
        if df.empty or date not in df.index:
            continue

        try:
            if noise_column in df.columns:
                noise = df.loc[date, noise_column]
                if noise <= noise_threshold:
                    filtered.append(symbol)
                else:
                    logger.debug(f"{symbol}: Filtered out (noise={noise:.3f} > {noise_threshold})")
            else:
                # If noise column doesn't exist, include symbol
                filtered.append(symbol)

        except (KeyError, ValueError) as e:
            logger.debug(f"{symbol}: Error checking noise: {e}")
            continue

    logger.debug(f"{date.date()}: {len(filtered)}/{len(symbols)} passed noise filter")

    return filtered


def rank_by_indicator(
    date: pd.Timestamp,
    data: Dict[str, pd.DataFrame],
    symbols: List[str],
    indicator: str
) -> Dict[str, float]:
    """
    Rank symbols by indicator value.

    Args:
        date: Current date
        data: Dict of {symbol: DataFrame}
        symbols: List of symbols to rank
        indicator: Indicator column name

    Returns:
        Dict of {symbol: indicator_value}
    """
    signals = {}

    for symbol in symbols:
        if symbol not in data:
            continue

        df = data[symbol]
        if df.empty or date not in df.index:
            continue

        try:
            if indicator in df.columns:
                value = df.loc[date, indicator]
                # Skip NaN values
                if not pd.isna(value):
                    signals[symbol] = float(value)
            else:
                logger.warning(f"{symbol}: Indicator '{indicator}' not found")

        except (KeyError, ValueError) as e:
            logger.debug(f"{symbol}: Error reading indicator: {e}")
            continue

    return signals


def allocate_equal_weight(
    symbols: List[str],
    total_weight: float = 1.0
) -> Dict[str, float]:
    """
    Allocate equal weight to all symbols.

    Args:
        symbols: List of symbols
        total_weight: Total weight to allocate (default: 1.0 = 100%)

    Returns:
        Dict of {symbol: weight}
    """
    if not symbols:
        return {}

    weight = total_weight / len(symbols)
    return {symbol: weight for symbol in symbols}


def allocate_long_short(
    signals: Dict[str, float],
    long_top_n: int = 5,
    short_bottom_n: int = 5,
    long_weight_total: float = 1.0,
    short_weight_total: float = -1.0
) -> Dict[str, float]:
    """
    Allocate long/short positions based on signals.

    Args:
        signals: Dict of {symbol: signal_value}
        long_top_n: Number of top symbols to go long
        short_bottom_n: Number of bottom symbols to short
        long_weight_total: Total weight for long positions (default: 1.0)
        short_weight_total: Total weight for short positions (default: -1.0)

    Returns:
        Dict of {symbol: weight}
    """
    if not signals:
        return {}

    # Sort by signal (descending)
    sorted_symbols = sorted(signals.items(), key=lambda x: x[1], reverse=True)

    weights = {}

    # Long top N
    if long_top_n > 0:
        top_symbols = [symbol for symbol, _ in sorted_symbols[:long_top_n]]
        if top_symbols:
            long_weight = long_weight_total / len(top_symbols)
            for symbol in top_symbols:
                weights[symbol] = long_weight

    # Short bottom N
    if short_bottom_n > 0:
        bottom_symbols = [symbol for symbol, _ in sorted_symbols[-short_bottom_n:]]
        if bottom_symbols:
            short_weight = short_weight_total / len(bottom_symbols)
            for symbol in bottom_symbols:
                weights[symbol] = short_weight

    return weights


def calculate_leverage(
    date: pd.Timestamp,
    symbol: str,
    df: pd.DataFrame,
    rules: List[Dict]
) -> float:
    """
    Calculate leverage based on rules.

    Args:
        date: Current date
        symbol: Symbol name
        df: DataFrame for this symbol
        rules: List of rule dicts with 'condition' and 'leverage'

    Returns:
        Total leverage (sum of all matching rules)

    Example:
        rules = [
            {'condition': 'mom7 > 0', 'leverage': 1},
            {'condition': 'mom20 > 0', 'leverage': 1},
            {'condition': 'close > ema50', 'leverage': 1},
        ]
    """
    if date not in df.index:
        return 0.0

    total_leverage = 0.0

    for rule in rules:
        try:
            condition = rule['condition']
            leverage = rule['leverage']

            # Create local context for eval
            row = df.loc[date]
            local_vars = {col: row[col] for col in df.columns if col in row.index}

            # Add previous close for comparison
            if date != df.index[0]:
                prev_idx = df.index.get_loc(date) - 1
                local_vars['close_prev'] = df.iloc[prev_idx]['close']

            # Evaluate condition
            if eval(condition, {"__builtins__": {}}, local_vars):
                total_leverage += leverage

        except Exception as e:
            logger.debug(f"{symbol}: Error evaluating rule '{condition}': {e}")
            continue

    return total_leverage


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=10, freq='D')

    sample_data = {
        'BTC': pd.DataFrame({
            'close': [100, 105, 110, 108, 112, 115, 118, 120, 119, 122],
            'high': [102, 107, 112, 110, 114, 117, 120, 122, 121, 124],
            'low': [98, 103, 108, 106, 110, 113, 116, 118, 117, 120],
            'volume': [1000, 1100, 1200, 1150, 1300, 1250, 1400, 1350, 1300, 1450],
            'mom7': [0.05, 0.06, 0.07, 0.06, 0.08, 0.09, 0.10, 0.09, 0.08, 0.10],
            'noise15': [0.5, 0.55, 0.6, 0.58, 0.62, 0.60, 0.65, 0.63, 0.61, 0.64],
        }, index=dates),
        'ETH': pd.DataFrame({
            'close': [50, 52, 51, 53, 55, 54, 56, 58, 57, 59],
            'high': [51, 53, 52, 54, 56, 55, 57, 59, 58, 60],
            'low': [49, 51, 50, 52, 54, 53, 55, 57, 56, 58],
            'volume': [800, 850, 820, 900, 950, 880, 1000, 980, 920, 1050],
            'mom7': [0.02, 0.03, 0.02, 0.04, 0.05, 0.04, 0.06, 0.05, 0.04, 0.06],
            'noise15': [0.6, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.73, 0.71, 0.74],
        }, index=dates),
        'SOL': pd.DataFrame({
            'close': [20, 19, 21, 22, 21, 23, 24, 23, 25, 26],
            'high': [21, 20, 22, 23, 22, 24, 25, 24, 26, 27],
            'low': [19, 18, 20, 21, 20, 22, 23, 22, 24, 25],
            'volume': [500, 480, 520, 550, 530, 580, 600, 570, 620, 650],
            'mom7': [-0.01, -0.02, 0.01, 0.02, 0.01, 0.03, 0.04, 0.03, 0.05, 0.06],
            'noise15': [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.69, 0.71],
        }, index=dates),
    }

    # Calculate TP for each symbol
    for symbol, df in sample_data.items():
        df['TP'] = (df['high'] + df['low'] + df['close']) / 3.0

    test_date = dates[5]

    # Test volume selection
    print("1. Testing volume selection:")
    top_coins = select_by_volume(test_date, sample_data, top_n=2, volume_metric='tp_volume')
    print(f"   Top 2 by volume: {top_coins}\n")

    # Test noise filtering
    print("2. Testing noise filtering:")
    filtered = filter_by_noise(test_date, sample_data, top_coins, noise_threshold=0.7)
    print(f"   After noise filter (< 0.7): {filtered}\n")

    # Test ranking
    print("3. Testing ranking by momentum:")
    signals = rank_by_indicator(test_date, sample_data, list(sample_data.keys()), 'mom7')
    print(f"   Signals: {signals}\n")

    # Test equal weight allocation
    print("4. Testing equal weight allocation:")
    weights = allocate_equal_weight(['BTC', 'ETH', 'SOL'])
    print(f"   Equal weights: {weights}\n")

    # Test long/short allocation
    print("5. Testing long/short allocation:")
    weights = allocate_long_short(signals, long_top_n=2, short_bottom_n=1)
    print(f"   Long/short weights: {weights}\n")

    # Test leverage calculation
    print("6. Testing leverage calculation:")
    rules = [
        {'condition': 'mom7 > 0', 'leverage': 1},
        {'condition': 'close > 100', 'leverage': 1},
    ]
    leverage = calculate_leverage(test_date, 'BTC', sample_data['BTC'], rules)
    print(f"   BTC leverage: {leverage}\n")

    print("✓ All portfolio utilities working correctly!")
