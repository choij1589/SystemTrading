"""
Trend Indicators

Includes EMA, Bollinger Bands, and Percent B.
"""

from typing import Union, List, Tuple
import pandas as pd
import numpy as np
import logging

from .base import PriceIndicator

logger = logging.getLogger(__name__)


class EMA(PriceIndicator):
    """
    Exponential Moving Average.

    Uses pandas .ewm() for efficient calculation.
    """

    def __init__(self, period: int = 20):
        """
        Initialize EMA indicator.

        Args:
            period: EMA period (span)
        """
        super().__init__(name=f"EMA({period})")
        self.period = period

        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA.

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with f'ema{period}' column added
        """
        self.validate_input(df)

        result = df.copy()
        col_name = f'ema{self.period}'

        # Vectorized EMA calculation
        result[col_name] = result['close'].ewm(span=self.period, adjust=False).mean()

        logger.debug(f"Calculated {col_name} for {len(result)} rows")

        return result

    def __repr__(self) -> str:
        return f"EMA(period={self.period})"


class BollingerBands(PriceIndicator):
    """
    Bollinger Bands: Moving average with upper/lower bands based on standard deviation.

    Bands:
    - Center: Simple moving average
    - Upper: Center + (num_std * std)
    - Lower: Center - (num_std * std)
    """

    def __init__(self, period: int = 20, num_std: float = 2.0):
        """
        Initialize Bollinger Bands.

        Args:
            period: Period for moving average and std
            num_std: Number of standard deviations for bands (default: 2)
        """
        super().__init__(name=f"BollingerBands({period}, {num_std})")
        self.period = period
        self.num_std = num_std

        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")
        if num_std <= 0:
            raise ValueError(f"num_std must be positive, got {num_std}")

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with 'bb_center', 'bb_upper', 'bb_lower' columns added
        """
        self.validate_input(df)

        result = df.copy()

        # Calculate center line (SMA)
        result['bb_center'] = result['close'].rolling(window=self.period).mean()

        # Calculate standard deviation
        std = result['close'].rolling(window=self.period).std()

        # Calculate bands
        result['bb_upper'] = result['bb_center'] + (self.num_std * std)
        result['bb_lower'] = result['bb_center'] - (self.num_std * std)

        logger.debug(f"Calculated Bollinger Bands({self.period}) for {len(result)} rows")

        return result

    def __repr__(self) -> str:
        return f"BollingerBands(period={self.period}, num_std={self.num_std})"


class PercentB(PriceIndicator):
    """
    Percent B (%B): Position of price relative to Bollinger Bands.

    Formula: %B = (close - lower_band) / (upper_band - lower_band)

    Values:
    - %B > 1: Price above upper band
    - %B = 0.5: Price at center
    - %B < 0: Price below lower band
    """

    def __init__(self, period: int = 20, num_std: float = 2.0):
        """
        Initialize Percent B.

        Args:
            period: Period for Bollinger Bands
            num_std: Number of standard deviations
        """
        super().__init__(name=f"PercentB({period})")
        self.period = period
        self.num_std = num_std
        self.bb = BollingerBands(period=period, num_std=num_std)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Percent B.

        Args:
            df: DataFrame with 'close' column

        Returns:
            DataFrame with 'percentB' column added
        """
        self.validate_input(df)

        # First calculate Bollinger Bands
        result = self.bb.calculate(df)

        # Calculate Percent B
        band_width = result['bb_upper'] - result['bb_lower']

        # Avoid division by zero
        band_width = band_width.replace(0, np.nan)

        result['percentB'] = (result['close'] - result['bb_lower']) / band_width

        logger.debug(f"Calculated PercentB({self.period}) for {len(result)} rows")

        return result

    def __repr__(self) -> str:
        return f"PercentB(period={self.period})"


def add_ema(
    df: pd.DataFrame,
    periods: Union[int, List[int]] = 20
) -> pd.DataFrame:
    """
    Convenience function to add EMA for one or multiple periods.

    Args:
        df: DataFrame with 'close' column
        periods: Single period or list of periods

    Returns:
        DataFrame with EMA column(s) added
    """
    if isinstance(periods, int):
        periods = [periods]

    result = df.copy()

    for period in periods:
        indicator = EMA(period=period)
        result = indicator.calculate(result)

    return result


def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Convenience function to add Bollinger Bands.

    Args:
        df: DataFrame with 'close' column
        period: BB period
        num_std: Number of standard deviations

    Returns:
        DataFrame with BB columns added
    """
    indicator = BollingerBands(period=period, num_std=num_std)
    return indicator.calculate(df)


def add_percent_b(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Convenience function to add Percent B.

    Args:
        df: DataFrame with 'close' column
        period: Period for Bollinger Bands
        num_std: Number of standard deviations

    Returns:
        DataFrame with percentB column added
    """
    indicator = PercentB(period=period, num_std=num_std)
    return indicator.calculate(df)


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

    # Calculate single EMA
    ema20 = EMA(period=20)
    result = ema20.calculate(sample_df)
    print(f"\nWith EMA(20):")
    print(result[['close', 'ema20']].tail())

    # Calculate multiple EMAs
    result = add_ema(sample_df, periods=[7, 20, 50])
    print(f"\nWith multiple EMAs:")
    print(result[['close', 'ema7', 'ema20', 'ema50']].tail())

    # Calculate Bollinger Bands
    result = add_bollinger_bands(sample_df, period=20)
    print(f"\nWith Bollinger Bands:")
    print(result[['close', 'bb_center', 'bb_upper', 'bb_lower']].tail())

    # Calculate Percent B
    result = add_percent_b(sample_df, period=20)
    print(f"\nWith Percent B:")
    print(result[['close', 'percentB']].tail())

    # Performance test
    import time

    large_df = pd.DataFrame({
        'close': np.random.randn(10000).cumsum() + 100
    })

    start = time.time()
    result = add_ema(large_df, periods=[7, 20, 30, 40, 50])
    result = add_percent_b(result, period=20)
    elapsed = time.time() - start

    print(f"\nPerformance test (10k rows, 5 EMAs + PercentB): {elapsed:.4f}s")
    print(f"✓ All calculations vectorized!")
