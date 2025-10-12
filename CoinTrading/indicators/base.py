"""
Base Indicator Class

Abstract base class for all technical indicators.
"""

from abc import ABC, abstractmethod
from typing import Union, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Indicator(ABC):
    """
    Abstract base class for technical indicators.

    All indicators should inherit from this class and implement
    the calculate() method.
    """

    def __init__(self, name: str = None):
        """
        Initialize indicator.

        Args:
            name: Indicator name (for logging/debugging)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with indicator column(s) added

        Note:
            - Should NOT modify the input DataFrame
            - Should return a copy with new columns added
            - Column names should be descriptive (e.g., 'mom20', 'rsi14')
        """
        pass

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> None:
        """
        Validate that DataFrame has required columns.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Raises:
            ValueError: If required columns are missing
        """
        if df.empty:
            raise ValueError(f"{self.name}: DataFrame is empty")

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"{self.name}: Missing required columns: {missing}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PriceIndicator(Indicator):
    """
    Base class for indicators that only need price data (close).
    """

    def validate_input(self, df: pd.DataFrame) -> None:
        """Validate that 'close' column exists."""
        self.validate_dataframe(df, ['close'])


class OHLCIndicator(Indicator):
    """
    Base class for indicators that need full OHLC data.
    """

    def validate_input(self, df: pd.DataFrame) -> None:
        """Validate that OHLC columns exist."""
        self.validate_dataframe(df, ['open', 'high', 'low', 'close'])


class VolumeIndicator(Indicator):
    """
    Base class for indicators that need volume data.
    """

    def validate_input(self, df: pd.DataFrame) -> None:
        """Validate that volume column exists."""
        self.validate_dataframe(df, ['close', 'volume'])


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    # Test validation
    price_ind = PriceIndicator()
    try:
        price_ind.validate_input(sample_df)
        print("✓ PriceIndicator validation passed")
    except ValueError as e:
        print(f"✗ PriceIndicator validation failed: {e}")

    ohlc_ind = OHLCIndicator()
    try:
        ohlc_ind.validate_input(sample_df)
        print("✓ OHLCIndicator validation passed")
    except ValueError as e:
        print(f"✗ OHLCIndicator validation failed: {e}")

    # Test with missing column
    incomplete_df = sample_df[['close']].copy()
    try:
        ohlc_ind.validate_input(incomplete_df)
        print("✗ Should have failed validation")
    except ValueError as e:
        print(f"✓ Correctly caught missing columns: {e}")
