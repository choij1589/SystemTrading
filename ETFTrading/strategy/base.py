"""
Base Strategy Classes for ETF Trading

Provides abstract base classes for implementing ETF investment strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class BaseETFStrategy(ABC):
    """
    Abstract base class for ETF investment strategies.

    All strategies must implement get_weights() method which returns
    target portfolio weights for rebalancing.
    """

    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize strategy.

        Args:
            name: Strategy name for identification
        """
        self.name = name

    @abstractmethod
    def get_weights(
        self,
        data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate target weights for each ETF at given date.

        Args:
            data: Dictionary mapping ticker to OHLCV DataFrame
            date: Current date for weight calculation
            **kwargs: Additional strategy-specific parameters

        Returns:
            Dictionary mapping ticker to target weight (0.0 to 1.0)
            Weights should sum to <= 1.0 (remainder is cash)

        Example:
            {
                "069500": 0.30,  # KODEX 200: 30%
                "360750": 0.40,  # TIGER S&P500: 40%
                "152380": 0.20,  # KODEX 국고채: 20%
                "132030": 0.10   # KODEX 골드: 10%
            }
        """
        pass

    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and normalize weights.

        Args:
            weights: Target weights dictionary

        Returns:
            Validated weights dictionary
        """
        # Remove negative weights
        weights = {k: max(0.0, v) for k, v in weights.items()}

        # Check sum
        total = sum(weights.values())

        if total > 1.0:
            # Normalize if over 100%
            weights = {k: v / total for k, v in weights.items()}

        # Remove zero weights
        weights = {k: v for k, v in weights.items() if v > 0.001}  # 0.1% minimum

        return weights

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class StaticAllocationStrategy(BaseETFStrategy):
    """
    Static asset allocation strategy with fixed weights.

    This is a simple buy-and-hold strategy with periodic rebalancing
    to maintain target weights.
    """

    def __init__(
        self,
        target_weights: Dict[str, float],
        name: str = "StaticAllocation"
    ):
        """
        Initialize static allocation strategy.

        Args:
            target_weights: Fixed target weights for each ticker
            name: Strategy name

        Example:
            target_weights = {
                "069500": 0.30,  # Domestic equity: 30%
                "360750": 0.40,  # US equity: 40%
                "152380": 0.20,  # Bonds: 20%
                "132030": 0.10   # Gold: 10%
            }
        """
        super().__init__(name)
        self.target_weights = self.validate_weights(target_weights)

    def get_weights(
        self,
        data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        **kwargs
    ) -> Dict[str, float]:
        """
        Return static target weights.

        Args:
            data: Price data (not used in static strategy)
            date: Current date (not used in static strategy)

        Returns:
            Static target weights
        """
        return self.target_weights.copy()


class DynamicAllocationStrategy(BaseETFStrategy):
    """
    Base class for dynamic allocation strategies.

    Dynamic strategies adjust weights based on market conditions,
    momentum, volatility, or other factors.
    """

    def __init__(self, universe: List[str], name: str = "DynamicAllocation"):
        """
        Initialize dynamic allocation strategy.

        Args:
            universe: List of ticker codes in investment universe
            name: Strategy name
        """
        super().__init__(name)
        self.universe = universe

    def calculate_momentum(
        self,
        df: pd.DataFrame,
        lookback_days: int = 60
    ) -> float:
        """
        Calculate momentum for a single ETF.

        Args:
            df: OHLCV DataFrame
            lookback_days: Lookback period in days

        Returns:
            Momentum value (return over lookback period)
        """
        if len(df) < lookback_days:
            return 0.0

        close_prices = df['close'].values
        if len(close_prices) < lookback_days:
            return 0.0

        current_price = close_prices[-1]
        past_price = close_prices[-lookback_days]

        if past_price == 0:
            return 0.0

        return (current_price - past_price) / past_price

    def calculate_volatility(
        self,
        df: pd.DataFrame,
        lookback_days: int = 60
    ) -> float:
        """
        Calculate annualized volatility.

        Args:
            df: OHLCV DataFrame
            lookback_days: Lookback period in days

        Returns:
            Annualized volatility
        """
        if len(df) < lookback_days + 1:
            return 0.0

        returns = df['close'].pct_change().dropna()
        if len(returns) < lookback_days:
            return 0.0

        recent_returns = returns.tail(lookback_days)
        return recent_returns.std() * np.sqrt(252)  # Annualize

    def rank_by_metric(
        self,
        data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        metric_func,
        **metric_kwargs
    ) -> List[tuple]:
        """
        Rank ETFs by a given metric.

        Args:
            data: Price data for all ETFs
            date: Current date
            metric_func: Function to calculate metric
            **metric_kwargs: Arguments for metric function

        Returns:
            List of (ticker, metric_value) tuples, sorted by metric (descending)
        """
        rankings = []

        for ticker, df in data.items():
            if ticker not in self.universe:
                continue

            # Filter data up to current date
            df_filtered = df[df['date'] <= date].copy()

            if df_filtered.empty:
                continue

            try:
                metric_value = metric_func(df_filtered, **metric_kwargs)
                rankings.append((ticker, metric_value))
            except Exception as e:
                print(f"Error calculating metric for {ticker}: {e}")
                continue

        # Sort by metric value (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings


if __name__ == "__main__":
    # Example usage
    print("Base ETF Strategy Classes")
    print("=" * 50)

    # Static allocation example
    static_strategy = StaticAllocationStrategy(
        target_weights={
            "069500": 0.30,
            "360750": 0.40,
            "152380": 0.20,
            "132030": 0.10
        },
        name="60/40 Portfolio"
    )

    print(f"Strategy: {static_strategy}")
    print(f"Target weights: {static_strategy.target_weights}")
