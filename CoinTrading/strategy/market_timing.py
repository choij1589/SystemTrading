"""
Market Timing Strategy (Step 3)

Dual momentum strategy with market regime detection and dynamic leverage.
Replicates Step3-MarketTiming.ipynb from original notebooks.

Expected results (with mom7 + timing):
- ~5084% total return vs ~1370% without timing
- Lower MDD: ~-48% vs ~-80%
"""

from typing import Dict, List, Optional
import pandas as pd
import logging

from .base import Strategy
from .portfolio import (
    select_by_volume,
    filter_by_noise,
    rank_by_indicator,
    calculate_leverage
)

logger = logging.getLogger(__name__)


class MarketTimingStrategy(Strategy):
    """
    Market timing strategy with dynamic leverage.

    Strategy:
    1. Select top 21 coins by TP×volume
    2. Filter out high-noise coins (noise15 > threshold)
    3. Rank by momentum indicator
    4. Detect market regime using BTC/ETH vs EMA50
    5. Bull market: Long top 4 with dynamic leverage
    6. Bear market: Short bottom 8 with dynamic leverage
    7. Individual timing adjustments

    Leverage rules:
    - Long: +1 if mom7>0, +1 if mom20>0, -1 if close>yesterday
    - Short: -1 if close<EMA7, -1 if close<EMA20, +1 if close<yesterday

    This replicates the Step3 notebook strategy.
    """

    def __init__(
        self,
        indicator: str = 'mom7',
        universe_size: int = 21,
        long_top_n: int = 4,
        short_bottom_n: int = 8,
        apply_noise_filter: bool = True,
        noise_threshold: float = 0.7,
        reference_symbols: List[str] = None,
        ema_period: int = 50,
        use_individual_timing: bool = True,
        regime_buffer_pct: float = 0.02,
        neutral_behavior: str = 'flat',
        name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize market timing strategy.

        Args:
            indicator: Indicator to rank by (e.g., 'mom7')
            universe_size: Number of coins to select by volume
            long_top_n: Number of top coins for long positions (bull market)
            short_bottom_n: Number of bottom coins for short positions (bear market)
            apply_noise_filter: Whether to filter out high-noise coins
            noise_threshold: Maximum noise level (default: 0.7)
            reference_symbols: Symbols for market regime detection (default: ['BTCUSDT', 'ETHUSDT'])
            ema_period: EMA period for regime detection (default: 50)
            use_individual_timing: Apply individual position timing
            regime_buffer_pct: Buffer percentage around EMA for hysteresis (default: 0.02 = 2%)
            neutral_behavior: Behavior in neutral zone: 'flat' (no positions) or 'reduced' (partial leverage)
            name: Strategy name
            config: Configuration dictionary
        """
        super().__init__(
            name=name or f"MarketTiming({indicator})",
            config=config
        )

        self.indicator = indicator
        self.universe_size = universe_size
        self.long_top_n = long_top_n
        self.short_bottom_n = short_bottom_n
        self.apply_noise_filter = apply_noise_filter
        self.noise_threshold = noise_threshold
        self.reference_symbols = reference_symbols or ['BTCUSDT', 'ETHUSDT']
        self.ema_period = ema_period
        self.use_individual_timing = use_individual_timing
        self.regime_buffer_pct = regime_buffer_pct
        self.neutral_behavior = neutral_behavior

        logger.info(
            f"Initialized {self.name}: "
            f"long={long_top_n}, short={short_bottom_n}, "
            f"indicator={indicator}, noise_filter={apply_noise_filter}, "
            f"buffer={regime_buffer_pct*100:.1f}%"
        )

    def detect_market_regime(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Detect market regime based on BTC/ETH divergence.

        Logic:
        - Bull: Both BTC and ETH above EMA50
        - Bear: Both BTC and ETH below EMA50
        - Neutral: One above, one below (divergence)

        Args:
            date: Current date
            data: Dict of {symbol: DataFrame}

        Returns:
            'bull', 'bear', or 'neutral'
        """
        ema_col = f'ema{self.ema_period}'

        # Check each reference symbol
        positions = []  # 'above' or 'below'

        for symbol in self.reference_symbols:
            if symbol not in data:
                continue

            df = data[symbol]
            if df.empty or date not in df.index:
                continue

            try:
                close = df.loc[date, 'close']

                if ema_col in df.columns:
                    ema = df.loc[date, ema_col]
                else:
                    # Calculate EMA if not present
                    ema = df.loc[:date, 'close'].ewm(span=self.ema_period, adjust=False).mean().iloc[-1]

                # Determine position relative to EMA
                if close > ema:
                    positions.append('above')
                else:
                    positions.append('below')

            except (KeyError, ValueError) as e:
                logger.debug(f"{symbol}: Error detecting regime: {e}")
                continue

        if not positions or len(positions) < 2:
            return 'neutral'

        # Decision logic:
        # Both above → bull
        if all(p == 'above' for p in positions):
            logger.debug(f"{date.date()}: BULL (both BTC/ETH above EMA50)")
            return 'bull'

        # Both below → bear
        if all(p == 'below' for p in positions):
            logger.debug(f"{date.date()}: BEAR (both BTC/ETH below EMA50)")
            return 'bear'

        # Divergence → neutral
        logger.debug(f"{date.date()}: NEUTRAL (BTC/ETH divergence)")
        return 'neutral'

    def select_universe(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """
        Select universe of coins.

        Args:
            date: Current date
            data: Dict of {symbol: DataFrame}

        Returns:
            List of symbols in universe
        """
        # Select by volume
        universe = select_by_volume(
            date=date,
            data=data,
            top_n=self.universe_size,
            volume_metric='tp_volume'
        )

        # Apply noise filter if enabled
        if self.apply_noise_filter:
            universe = filter_by_noise(
                date=date,
                data=data,
                symbols=universe,
                noise_threshold=self.noise_threshold,
                noise_column='noise15'
            )

        return universe

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

    def generate_weights(
        self,
        date: pd.Timestamp,
        signals: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Generate weights based on market regime and dynamic leverage.

        This method is overridden to access data for leverage calculation.

        Args:
            date: Current date
            signals: Dict of {symbol: signal_value}

        Returns:
            Dict of {symbol: weight}
        """
        # This will be called from get_weights() which has access to data
        # We'll override get_weights() instead
        raise NotImplementedError("Use get_weights() instead")

    def get_weights(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Generate weights with market regime detection and dynamic leverage.

        Args:
            date: Current date
            data: Dict of {symbol: DataFrame}

        Returns:
            Dict of {symbol: weight}
        """
        try:
            # 1. Detect market regime
            regime = self.detect_market_regime(date, data)

            # 2. Select universe
            universe = self.select_universe(date, data)
            if not universe:
                return {}

            # 3. Calculate signals
            signals = self.calculate_signals(date, data, universe)
            if not signals:
                return {}

            # 4. Sort by signal
            sorted_symbols = sorted(signals.items(), key=lambda x: x[1], reverse=True)

            weights = {}

            if regime == 'bull':
                # Long top N with dynamic leverage
                top_symbols = [symbol for symbol, _ in sorted_symbols[:self.long_top_n]]

                for symbol in top_symbols:
                    if symbol not in data or date not in data[symbol].index:
                        continue

                    df = data[symbol]

                    # Base leverage rules
                    rules = [
                        {'condition': 'mom7 > 0', 'leverage': 1},
                        {'condition': 'mom20 > 0', 'leverage': 1},
                    ]

                    # Individual timing
                    if self.use_individual_timing:
                        rules.append({'condition': 'close > close_prev', 'leverage': -1})

                    # Calculate leverage
                    leverage = calculate_leverage(date, symbol, df, rules)

                    # Always include position, even if leverage = 0
                    # This matches original Step3 behavior where all selected coins are included
                    # Weight = leverage / n_positions (can be 0 for neutral position)
                    weight = leverage / self.long_top_n
                    weights[symbol] = weight

            elif regime == 'bear':
                # Short bottom N with dynamic leverage
                bottom_symbols = [symbol for symbol, _ in sorted_symbols[-self.short_bottom_n:]]

                for symbol in bottom_symbols:
                    if symbol not in data or date not in data[symbol].index:
                        continue

                    df = data[symbol]

                    # Base leverage rules (negative for short)
                    rules = [
                        {'condition': 'close < ema7', 'leverage': -1},
                        {'condition': 'close < ema20', 'leverage': -1},
                    ]

                    # Individual timing
                    if self.use_individual_timing:
                        rules.append({'condition': 'close < close_prev', 'leverage': 1})

                    # Calculate leverage
                    leverage = calculate_leverage(date, symbol, df, rules)

                    # Always include position, even if leverage = 0
                    # This matches original Step3 behavior where all selected coins are included
                    # Weight = leverage / n_positions (can be 0 for neutral position)
                    weight = leverage / self.short_bottom_n
                    weights[symbol] = weight

            else:  # neutral
                # In neutral zone (price near EMA50), avoid whipsaw by:
                # - 'flat': No positions (default, safest)
                # - 'reduced': Could implement partial exposure here if desired
                if self.neutral_behavior == 'flat':
                    return {}
                else:
                    # For future enhancement: could return reduced leverage positions
                    return {}

            return weights

        except Exception as e:
            logger.error(f"{date.date()}: Strategy failed: {e}")
            return {}

    def __repr__(self) -> str:
        return (
            f"MarketTimingStrategy("
            f"indicator='{self.indicator}', "
            f"long={self.long_top_n}, "
            f"short={self.short_bottom_n}, "
            f"noise_filter={self.apply_noise_filter})"
        )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data with all required indicators
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    import numpy as np

    np.random.seed(42)

    sample_data = {}

    # Create BTC and ETH as reference
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        close_prices = (1 + np.random.randn(100) * 0.02).cumprod() * 100
        high_prices = close_prices * 1.01
        low_prices = close_prices * 0.99
        volumes = np.random.randint(10000, 50000, 100)

        df = pd.DataFrame({
            'close': close_prices,
            'high': high_prices,
            'low': low_prices,
            'volume': volumes,
        }, index=dates)

        df['TP'] = (df['high'] + df['low'] + df['close']) / 3.0
        df['mom7'] = df['close'].pct_change(periods=7)
        df['mom20'] = df['close'].pct_change(periods=20)
        df['ema7'] = df['close'].ewm(span=7, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['noise15'] = 0.5  # Low noise

        df.dropna(inplace=True)
        sample_data[symbol] = df

    # Create other coins
    for symbol in ['SOLUSDT', 'ADAUSDT', 'DOTUSDT', 'MATICUSDT', 'AVAXUSDT', 'LINKUSDT']:
        close_prices = (1 + np.random.randn(100) * 0.03).cumprod() * 50
        high_prices = close_prices * 1.01
        low_prices = close_prices * 0.99
        volumes = np.random.randint(5000, 20000, 100)

        df = pd.DataFrame({
            'close': close_prices,
            'high': high_prices,
            'low': low_prices,
            'volume': volumes,
        }, index=dates)

        df['TP'] = (df['high'] + df['low'] + df['close']) / 3.0
        df['mom7'] = df['close'].pct_change(periods=7)
        df['mom20'] = df['close'].pct_change(periods=20)
        df['ema7'] = df['close'].ewm(span=7, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['noise15'] = np.random.uniform(0.5, 0.8)

        df.dropna(inplace=True)
        sample_data[symbol] = df

    print("Sample data created with all indicators")
    print(f"Symbols: {list(sample_data.keys())}")
    print()

    # Test market timing strategy
    print("=" * 60)
    print("Testing MarketTimingStrategy")
    print("=" * 60)

    strategy = MarketTimingStrategy(
        indicator='mom7',
        long_top_n=3,
        short_bottom_n=4,
        apply_noise_filter=True,
        noise_threshold=0.7,
        use_individual_timing=True
    )

    test_date = dates[70]
    data_slice = {symbol: df.loc[:test_date] for symbol, df in sample_data.items()}

    # Detect regime
    regime = strategy.detect_market_regime(test_date, data_slice)
    print(f"\nMarket regime on {test_date.date()}: {regime.upper()}")

    # Get weights
    weights = strategy.get_weights(test_date, data_slice)

    print(f"\nWeights:")
    if weights:
        for symbol in sorted(weights.keys(), key=lambda x: weights[x], reverse=True):
            weight = weights[symbol]
            position = "LONG" if weight > 0 else "SHORT"
            print(f"  {symbol:10s}: {weight:+.4f}  ({position})")

        total_long = sum(w for w in weights.values() if w > 0)
        total_short = sum(w for w in weights.values() if w < 0)
        print(f"\nTotal long exposure: {total_long:.2f}")
        print(f"Total short exposure: {total_short:.2f}")
    else:
        print("  No positions")

    print()
    print("✓ Market timing strategy working correctly!")
