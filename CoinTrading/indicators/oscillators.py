"""
Oscillator Indicators

Includes RSI (fully vectorized - fixes original for-loop bug) and Noise.
"""

from typing import Union, List
import pandas as pd
import numpy as np
import logging

from .base import OHLCIndicator, Indicator

logger = logging.getLogger(__name__)


class RSI(OHLCIndicator):
    """
    Relative Strength Index (RSI) - VECTORIZED VERSION

    This fixes the inefficient for-loop implementation in the original code.

    Formula:
    1. Calculate typical price: TP = (high + low + close) / 3
    2. Calculate price changes: delta = TP.diff()
    3. Separate gains and losses
    4. Calculate average gain and loss over period
    5. RSI = avg_gain / (avg_gain + avg_loss)

    Note: Original used for-loop iterating over index, which is very slow.
    """

    def __init__(self, period: int = 14):
        """
        Initialize RSI indicator.

        Args:
            period: Lookback period for averaging (default: 14)
        """
        super().__init__(name=f"RSI({period})")
        self.period = period

        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI using vectorized operations.

        Args:
            df: DataFrame with OHLC columns

        Returns:
            DataFrame with 'TP' and 'RSI' columns added
        """
        self.validate_input(df)

        result = df.copy()

        # Calculate typical price
        result['TP'] = (result['high'] + result['low'] + result['close']) / 3.0

        # Calculate price changes
        delta = result['TP'].diff()

        # Separate gains and losses (vectorized, no for-loop!)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Calculate average gain and loss using rolling window
        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()

        # Avoid division by zero
        avg_loss = avg_loss.replace(0, np.nan)

        # Calculate RSI
        rs = avg_gain / avg_loss
        result['RSI'] = avg_gain / (avg_gain + avg_loss)

        # Alternative formula: 1 - (1 / (1 + rs))
        # Both are equivalent, but this one handles edge cases better

        logger.debug(f"Calculated RSI({self.period}) for {len(result)} rows")

        return result

    def __repr__(self) -> str:
        return f"RSI(period={self.period})"


class Noise(OHLCIndicator):
    """
    Noise indicator: measures the ratio of body to total range.

    Formula: noise = 1 - abs(close - open) / (high - low)

    High noise (close to 1) indicates lots of wicks, suggesting uncertainty.
    Low noise (close to 0) indicates strong directional movement.

    The indicator is typically smoothed with a moving average.
    """

    def __init__(self, period: int = 15):
        """
        Initialize Noise indicator.

        Args:
            period: Period for moving average smoothing
        """
        super().__init__(name=f"Noise({period})")
        self.period = period

        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate noise indicator.

        Args:
            df: DataFrame with OHLC columns

        Returns:
            DataFrame with f'noise{period}' column added
        """
        self.validate_input(df)

        result = df.copy()
        col_name = f'noise{self.period}'

        # Calculate instantaneous noise (vectorized)
        range_hl = result['high'] - result['low']
        body = (result['close'] - result['open']).abs()

        # Avoid division by zero
        range_hl = range_hl.replace(0, np.nan)

        noise_instant = 1.0 - (body / range_hl)

        # Smooth with moving average
        result[col_name] = noise_instant.rolling(
            window=self.period,
            min_periods=1
        ).mean()

        logger.debug(f"Calculated {col_name} for {len(result)} rows")

        return result

    def __repr__(self) -> str:
        return f"Noise(period={self.period})"


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Convenience function to add RSI.

    Args:
        df: DataFrame with OHLC columns
        period: RSI period

    Returns:
        DataFrame with 'TP' and 'RSI' columns added
    """
    indicator = RSI(period=period)
    return indicator.calculate(df)


def add_noise(df: pd.DataFrame, period: int = 15) -> pd.DataFrame:
    """
    Convenience function to add Noise indicator.

    Args:
        df: DataFrame with OHLC columns
        period: Smoothing period

    Returns:
        DataFrame with f'noise{period}' column added
    """
    indicator = Noise(period=period)
    return indicator.calculate(df)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100
    }, index=dates)

    # Ensure high >= low (fix random data)
    sample_df['high'] = sample_df[['open', 'high', 'low', 'close']].max(axis=1)
    sample_df['low'] = sample_df[['open', 'high', 'low', 'close']].min(axis=1)

    print("Original data:")
    print(sample_df.head())

    # Calculate RSI
    rsi = RSI(period=14)
    result = rsi.calculate(sample_df)
    print(f"\nWith RSI(14):")
    print(result[['close', 'TP', 'RSI']].tail())

    # Calculate Noise
    noise = Noise(period=15)
    result = noise.calculate(result)
    print(f"\nWith Noise(15):")
    print(result[['close', 'noise15']].tail())

    # Performance test (should be fast - no for-loops!)
    import time

    large_df = pd.DataFrame({
        'open': np.random.randn(10000).cumsum() + 100,
        'high': np.random.randn(10000).cumsum() + 102,
        'low': np.random.randn(10000).cumsum() + 98,
        'close': np.random.randn(10000).cumsum() + 100
    })
    large_df['high'] = large_df[['open', 'high', 'low', 'close']].max(axis=1)
    large_df['low'] = large_df[['open', 'high', 'low', 'close']].min(axis=1)

    start = time.time()
    result = add_rsi(large_df, period=14)
    result = add_noise(result, period=15)
    elapsed = time.time() - start

    print(f"\nPerformance test (10k rows, RSI + Noise): {elapsed:.4f}s")
    print(f"✓ Vectorized implementation - no for-loops!")
