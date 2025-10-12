"""
Simple Momentum Strategy (Step 1)

Long top 5 / Short bottom 5 strategy based on any indicator.
Replicates Step1-InspectingFactors.ipynb from original notebooks.

Expected results (with mom20 indicator):
- Long top 5: ~5774% total return
- Short bottom 5: ~-70% total return
"""

from typing import Dict, List, Optional
import pandas as pd
import logging

from .base import LongShortStrategy
from .portfolio import select_by_volume, rank_by_indicator

logger = logging.getLogger(__name__)


class MomentumSimpleStrategy(LongShortStrategy):
    """
    Simple momentum long/short strategy.

    Strategy:
    1. Select top 21 coins by TP×volume
    2. Rank by indicator (momentum, RSI, or percentB)
    3. Long top 5, short bottom 5
    4. Equal weight allocation
    5. Rebalance daily

    This replicates the Step1 notebook strategy.
    """

    def __init__(
        self,
        indicator: str = 'mom20',
        long_top_n: int = 5,
        short_bottom_n: int = 5,
        universe_size: int = 21,
        volume_metric: str = 'tp_volume',
        name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize simple momentum strategy.

        Args:
            indicator: Indicator to rank by (e.g., 'mom20', 'RSI', 'percentB')
            long_top_n: Number of top coins to go long
            short_bottom_n: Number of bottom coins to short
            universe_size: Number of coins to select by volume
            volume_metric: Volume metric for selection
            name: Strategy name
            config: Configuration dictionary
        """
        super().__init__(
            name=name or f"MomentumSimple({indicator})",
            config=config,
            long_top_n=long_top_n,
            short_bottom_n=short_bottom_n
        )

        self.indicator = indicator
        self.universe_size = universe_size
        self.volume_metric = volume_metric

        logger.info(
            f"Initialized {self.name}: "
            f"long_top={long_top_n}, short_bottom={short_bottom_n}, "
            f"indicator={indicator}, universe={universe_size}"
        )

    def select_universe(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """
        Select top N coins by trading volume.

        Args:
            date: Current date
            data: Dict of {symbol: DataFrame}

        Returns:
            List of symbols in universe
        """
        return select_by_volume(
            date=date,
            data=data,
            top_n=self.universe_size,
            volume_metric=self.volume_metric
        )

    def calculate_signals(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
        universe: List[str]
    ) -> Dict[str, float]:
        """
        Calculate signals based on indicator.

        Args:
            date: Current date
            data: Dict of {symbol: DataFrame}
            universe: List of symbols in universe

        Returns:
            Dict of {symbol: indicator_value}
        """
        return rank_by_indicator(
            date=date,
            data=data,
            symbols=universe,
            indicator=self.indicator
        )

    def __repr__(self) -> str:
        return (
            f"MomentumSimpleStrategy("
            f"indicator='{self.indicator}', "
            f"long={self.long_top_n}, "
            f"short={self.short_bottom_n})"
        )


class MomentumLongOnlyStrategy(LongShortStrategy):
    """
    Momentum long-only strategy.

    Same as MomentumSimpleStrategy but only goes long (no shorts).
    Can be configured to long either top or bottom ranked coins.
    """

    def __init__(
        self,
        indicator: str = 'mom20',
        long_top_n: int = 5,
        long_bottom: bool = False,
        universe_size: int = 21,
        volume_metric: str = 'tp_volume',
        name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize long-only momentum strategy.

        Args:
            indicator: Indicator to rank by
            long_top_n: Number of coins to go long
            long_bottom: If True, long bottom N coins instead of top N
            universe_size: Number of coins to select by volume
            volume_metric: Volume metric for selection
            name: Strategy name
            config: Configuration dictionary
        """
        suffix = "Bottom" if long_bottom else "Top"
        super().__init__(
            name=name or f"MomentumLongOnly{suffix}({indicator})",
            config=config,
            long_top_n=long_top_n if not long_bottom else 0,
            short_bottom_n=0  # No shorts
        )

        self.indicator = indicator
        self.long_bottom = long_bottom
        self.long_n = long_top_n
        self.universe_size = universe_size
        self.volume_metric = volume_metric

        logger.info(
            f"Initialized {self.name}: "
            f"long_{suffix.lower()}={long_top_n}, indicator={indicator}, universe={universe_size}"
        )

    def select_universe(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Select top N coins by trading volume."""
        return select_by_volume(
            date=date,
            data=data,
            top_n=self.universe_size,
            volume_metric=self.volume_metric
        )

    def calculate_signals(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
        universe: List[str]
    ) -> Dict[str, float]:
        """Calculate signals based on indicator."""
        return rank_by_indicator(
            date=date,
            data=data,
            symbols=universe,
            indicator=self.indicator
        )

    def generate_weights(
        self,
        date: pd.Timestamp,
        signals: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Generate weights for long-only strategy.

        If long_bottom=False: Long top N coins
        If long_bottom=True: Long bottom N coins

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

        if self.long_bottom:
            # Long bottom N coins
            bottom_symbols = [symbol for symbol, _ in sorted_symbols[-self.long_n:]]
            if bottom_symbols:
                long_weight = 1.0 / len(bottom_symbols)
                for symbol in bottom_symbols:
                    weights[symbol] = long_weight
        else:
            # Long top N coins
            top_symbols = [symbol for symbol, _ in sorted_symbols[:self.long_n]]
            if top_symbols:
                long_weight = 1.0 / len(top_symbols)
                for symbol in top_symbols:
                    weights[symbol] = long_weight

        return weights

    def __repr__(self) -> str:
        return (
            f"MomentumLongOnlyStrategy("
            f"indicator='{self.indicator}', "
            f"long={self.long_top_n})"
        )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data with momentum indicator
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    import numpy as np

    np.random.seed(42)

    sample_data = {}
    for symbol in ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'MATIC', 'AVAX', 'LINK']:
        close_prices = (1 + np.random.randn(100) * 0.02).cumprod() * 100
        high_prices = close_prices * 1.01
        low_prices = close_prices * 0.99
        volumes = np.random.randint(1000, 10000, 100)

        df = pd.DataFrame({
            'close': close_prices,
            'high': high_prices,
            'low': low_prices,
            'volume': volumes,
        }, index=dates)

        # Calculate TP
        df['TP'] = (df['high'] + df['low'] + df['close']) / 3.0

        # Calculate momentum
        df['mom20'] = df['close'].pct_change(periods=20)
        df['mom7'] = df['close'].pct_change(periods=7)

        # Drop NaN
        df.dropna(inplace=True)

        sample_data[symbol] = df

    print("Sample data created with momentum indicators")
    print(f"Symbols: {list(sample_data.keys())}")
    print(f"Dates: {sample_data['BTC'].index[0].date()} to {sample_data['BTC'].index[-1].date()}")
    print()

    # Test long/short strategy
    print("=" * 60)
    print("Testing MomentumSimpleStrategy (long/short)")
    print("=" * 60)

    strategy = MomentumSimpleStrategy(
        indicator='mom20',
        long_top_n=3,
        short_bottom_n=2,
        universe_size=8
    )

    test_date = dates[50]
    data_slice = {symbol: df.loc[:test_date] for symbol, df in sample_data.items()}

    weights = strategy.get_weights(test_date, data_slice)

    print(f"\nWeights for {test_date.date()}:")
    for symbol in sorted(weights.keys(), key=lambda x: weights[x], reverse=True):
        weight = weights[symbol]
        position = "LONG" if weight > 0 else "SHORT"
        print(f"  {symbol:6s}: {weight:+.4f}  ({position})")

    print(f"\nTotal long exposure: {sum(w for w in weights.values() if w > 0):.2f}")
    print(f"Total short exposure: {sum(w for w in weights.values() if w < 0):.2f}")
    print()

    # Test long-only strategy
    print("=" * 60)
    print("Testing MomentumLongOnlyStrategy")
    print("=" * 60)

    strategy_long = MomentumLongOnlyStrategy(
        indicator='mom7',
        long_top_n=5,
        universe_size=8
    )

    weights_long = strategy_long.get_weights(test_date, data_slice)

    print(f"\nWeights for {test_date.date()}:")
    for symbol, weight in sorted(weights_long.items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol:6s}: {weight:+.4f}")

    print(f"\nTotal exposure: {sum(weights_long.values()):.2f}")
    print()

    print("✓ Momentum strategies working correctly!")
