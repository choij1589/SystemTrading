"""
Momentum Indicators

Vectorized momentum calculations.
"""

from typing import Union, List
import pandas as pd
import numpy as np
import logging

from .base import PriceIndicator

logger = logging.getLogger(__name__)


class Momentum(PriceIndicator):
    """
    Momentum indicator: rate of change over a period.

    Formula: (close - close_shifted) / close_shifted

    This measures the percentage change in price over the specified period.
    """

    def __init__(self, period: int = 20):
        """
        Initialize Momentum indicator.

        Args:
            period: Lookback period in days
        """
        super().__init__(name=f"Momentum({period})")
        self.period = period

        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum.

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with f'mom{period}' column added
        """
        self.validate_input(df)

        result = df.copy()
        col_name = f'mom{self.period}'

        # Vectorized calculation (no for-loop!)
        close_shifted = result['close'].shift(self.period)
        result[col_name] = (result['close'] - close_shifted) / close_shifted

        logger.debug(f"Calculated {col_name} for {len(result)} rows")

        return result

    def __repr__(self) -> str:
        return f"Momentum(period={self.period})"


def add_momentum(
    df: pd.DataFrame,
    periods: Union[int, List[int]] = 20
) -> pd.DataFrame:
    """
    Convenience function to add momentum for one or multiple periods.

    Args:
        df: DataFrame with 'close' column
        periods: Single period or list of periods

    Returns:
        DataFrame with momentum column(s) added

    Example:
        >>> df = add_momentum(df, periods=[7, 20, 60])
        >>> # Adds columns: mom7, mom20, mom60
    """
    if isinstance(periods, int):
        periods = [periods]

    result = df.copy()

    for period in periods:
        indicator = Momentum(period=period)
        result = indicator.calculate(result)

    return result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_df = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100
    }, index=dates)

    print("Original data:")
    print(sample_df.head())

    # Calculate single momentum
    mom20 = Momentum(period=20)
    result = mom20.calculate(sample_df)
    print(f"\nWith Momentum(20):")
    print(result[['close', 'mom20']].tail())

    # Calculate multiple periods using convenience function
    result_multi = add_momentum(sample_df, periods=[7, 20, 60])
    print(f"\nWith multiple periods:")
    print(result_multi[['close', 'mom7', 'mom20', 'mom60']].tail())

    # Verify no for-loops (should be very fast)
    import time
    large_df = pd.DataFrame({
        'close': np.random.randn(10000).cumsum() + 100
    })

    start = time.time()
    result = add_momentum(large_df, periods=[7, 14, 20, 21, 60])
    elapsed = time.time() - start

    print(f"\nPerformance test (10k rows, 5 periods): {elapsed:.4f}s")
    print(f"✓ Vectorized implementation is fast!")
