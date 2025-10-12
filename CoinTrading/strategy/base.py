"""
Base Strategy Class

Abstract base class for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies should inherit from this class and implement
    the required methods.

    The strategy lifecycle for each date:
    1. select_universe() - Filter coins to consider
    2. calculate_signals() - Calculate trading signals/scores
    3. generate_weights() - Convert signals to position weights

    Integration with BacktestEngine:
    - The engine calls get_weights(date, data) for each period
    - Strategy returns {symbol: weight} dict
    - Engine handles rebalancing and cost calculation
    """

    def __init__(self, name: str = None, config: Optional[Dict] = None):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}

    @abstractmethod
    def select_universe(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """
        Select universe of coins to trade.

        Args:
            date: Current date
            data: Dict of {symbol: DataFrame} with data up to current date

        Returns:
            List of symbols to consider for this period
        """
        pass

    @abstractmethod
    def calculate_signals(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
        universe: List[str]
    ) -> Dict[str, float]:
        """
        Calculate trading signals for each symbol.

        Args:
            date: Current date
            data: Dict of {symbol: DataFrame}
            universe: List of symbols in universe

        Returns:
            Dict of {symbol: signal_value}
            Signal can be any numeric value (e.g., momentum, score, rank)
        """
        pass

    @abstractmethod
    def generate_weights(
        self,
        date: pd.Timestamp,
        signals: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Convert signals to position weights.

        Args:
            date: Current date
            signals: Dict of {symbol: signal_value}

        Returns:
            Dict of {symbol: weight}
            Weights should sum to desired total exposure (e.g., 1.0 for 100%)
            Negative weights indicate short positions
        """
        pass

    def get_weights(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Main entry point called by backtesting engine.

        This is the interface method that BacktestEngine uses.

        Args:
            date: Current date
            data: Dict of {symbol: DataFrame} with data up to current date

        Returns:
            Dict of {symbol: weight} for position allocation
        """
        try:
            # 1. Select universe
            universe = self.select_universe(date, data)

            if not universe:
                logger.debug(f"{date.date()}: Empty universe, no positions")
                return {}

            # 2. Calculate signals
            signals = self.calculate_signals(date, data, universe)

            if not signals:
                logger.debug(f"{date.date()}: No signals generated")
                return {}

            # 3. Generate weights
            weights = self.generate_weights(date, signals)

            # Validate weights
            if weights:
                total_abs_weight = sum(abs(w) for w in weights.values())
                logger.debug(
                    f"{date.date()}: {len(weights)} positions, "
                    f"total exposure: {total_abs_weight:.2f}"
                )

            return weights

        except Exception as e:
            logger.error(f"{date.date()}: Strategy failed: {e}")
            return {}

    def validate_data(
        self,
        symbol: str,
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> bool:
        """
        Validate that DataFrame has required columns.

        Args:
            symbol: Symbol name (for logging)
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if valid, False otherwise
        """
        if df.empty:
            logger.warning(f"{symbol}: DataFrame is empty")
            return False

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.warning(f"{symbol}: Missing columns: {missing}")
            return False

        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class LongShortStrategy(Strategy):
    """
    Base class for long/short strategies.

    Provides common functionality for strategies that go long top-ranked
    assets and short bottom-ranked assets.
    """

    def __init__(
        self,
        name: str = None,
        config: Optional[Dict] = None,
        long_top_n: int = 5,
        short_bottom_n: int = 5
    ):
        """
        Initialize long/short strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary
            long_top_n: Number of top assets to go long
            short_bottom_n: Number of bottom assets to short
        """
        super().__init__(name, config)
        self.long_top_n = long_top_n
        self.short_bottom_n = short_bottom_n

    def generate_weights(
        self,
        date: pd.Timestamp,
        signals: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Generate weights for long/short strategy.

        Long top N with equal positive weight, short bottom N with equal negative weight.

        Args:
            date: Current date
            signals: Dict of {symbol: signal_value}

        Returns:
            Dict of {symbol: weight}
        """
        if not signals:
            return {}

        # Sort by signal (descending)
        sorted_symbols = sorted(signals.items(), key=lambda x: x[1], reverse=True)

        weights = {}

        # Long top N
        if self.long_top_n > 0:
            top_symbols = [symbol for symbol, _ in sorted_symbols[:self.long_top_n]]
            long_weight = 1.0 / self.long_top_n if top_symbols else 0.0

            for symbol in top_symbols:
                weights[symbol] = long_weight

        # Short bottom N
        if self.short_bottom_n > 0:
            bottom_symbols = [symbol for symbol, _ in sorted_symbols[-self.short_bottom_n:]]
            short_weight = -1.0 / self.short_bottom_n if bottom_symbols else 0.0

            for symbol in bottom_symbols:
                weights[symbol] = short_weight

        return weights


class LongOnlyStrategy(Strategy):
    """
    Base class for long-only strategies.

    Provides common functionality for strategies that only go long.
    """

    def __init__(
        self,
        name: str = None,
        config: Optional[Dict] = None,
        long_top_n: int = 5
    ):
        """
        Initialize long-only strategy.

        Args:
            name: Strategy name
            config: Configuration dictionary
            long_top_n: Number of top assets to go long
        """
        super().__init__(name, config)
        self.long_top_n = long_top_n

    def generate_weights(
        self,
        date: pd.Timestamp,
        signals: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Generate weights for long-only strategy.

        Long top N with equal weight.

        Args:
            date: Current date
            signals: Dict of {symbol: signal_value}

        Returns:
            Dict of {symbol: weight}
        """
        if not signals:
            return {}

        # Sort by signal (descending)
        sorted_symbols = sorted(signals.items(), key=lambda x: x[1], reverse=True)

        # Long top N
        top_symbols = [symbol for symbol, _ in sorted_symbols[:self.long_top_n]]

        if not top_symbols:
            return {}

        weight = 1.0 / len(top_symbols)
        weights = {symbol: weight for symbol in top_symbols}

        return weights


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test LongShortStrategy base class
    class TestStrategy(LongShortStrategy):
        def select_universe(self, date, data):
            return list(data.keys())

        def calculate_signals(self, date, data, universe):
            # Simple example: use last close price as signal
            signals = {}
            for symbol in universe:
                if symbol in data and not data[symbol].empty:
                    signals[symbol] = data[symbol].iloc[-1]['close']
            return signals

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    sample_data = {
        'BTC': pd.DataFrame({'close': [100, 105, 110, 108, 112, 115, 118, 120, 119, 122]}, index=dates),
        'ETH': pd.DataFrame({'close': [50, 52, 51, 53, 55, 54, 56, 58, 57, 59]}, index=dates),
        'SOL': pd.DataFrame({'close': [20, 19, 21, 22, 21, 23, 24, 23, 25, 26]}, index=dates),
        'ADA': pd.DataFrame({'close': [1, 1.1, 1.05, 1.15, 1.2, 1.18, 1.22, 1.25, 1.23, 1.28]}, index=dates),
        'DOT': pd.DataFrame({'close': [5, 5.1, 4.9, 5.2, 5.3, 5.1, 5.4, 5.5, 5.3, 5.6]}, index=dates),
    }

    # Test strategy
    strategy = TestStrategy(long_top_n=2, short_bottom_n=2)
    print(f"Strategy: {strategy}\n")

    # Get weights for a date
    test_date = dates[5]
    data_slice = {symbol: df.loc[:test_date] for symbol, df in sample_data.items()}

    weights = strategy.get_weights(test_date, data_slice)
    print(f"Weights for {test_date.date()}:")
    for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {weight:+.4f}")

    print("\n✓ Base strategy classes working correctly!")
