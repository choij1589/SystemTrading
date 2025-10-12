"""
Regime-Based Adaptive Strategy for BTC/ETH/LTC

Hybrid approach:
- Each coin (BTC, ETH, LTC) gets individual direction based on its position vs EMA50
- Portfolio leverage is adjusted based on regime (how many coins above EMA50)
"""

from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RegimeBasedStrategy:
    """
    Regime-based strategy trading BTC, ETH, LTC only.

    Individual Coin Direction:
    - Coin above EMA50 → Long (+1/3 base weight)
    - Coin below EMA50 → Short (-1/3 base weight)

    Regime Leverage Multiplier (count above EMA50):
    - Harsh Bull (3/3): 1.5x leverage
    - Bull (2/3): 1.0x leverage
    - Bear (1/3): 0.5x leverage
    - Harsh Bear (0/3): 1.5x leverage

    Final weight = individual_direction × (1/3) × regime_multiplier
    """

    def __init__(
        self,
        symbols: List[str] = None,
        ema_period: int = 50,
        harsh_bull_leverage: float = 1.5,
        bull_leverage: float = 1.0,
        bear_leverage: float = 0.5,
        harsh_bear_leverage: float = 1.5,
        name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize regime-based strategy.

        Args:
            symbols: List of symbols to trade (default: BTC, ETH, LTC)
            ema_period: EMA period for regime detection (default: 50)
            harsh_bull_leverage: Leverage when all 3 above EMA (default: 1.5)
            bull_leverage: Leverage when 2/3 above EMA (default: 1.0)
            bear_leverage: Leverage when 1/3 above EMA (default: 0.5)
            harsh_bear_leverage: Leverage when 0/3 above EMA (default: 1.5)
            name: Strategy name
            config: Configuration dictionary
        """
        self.name = name or "RegimeBased(BTC/ETH/LTC)"
        self.config = config
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'LTCUSDT']
        self.ema_period = ema_period
        self.n_coins = len(self.symbols)

        # Regime leverage multipliers
        self.regime_leverage = {
            'harsh_bull': harsh_bull_leverage,
            'bull': bull_leverage,
            'bear': bear_leverage,
            'harsh_bear': harsh_bear_leverage
        }

        logger.info(
            f"Initialized {self.name}: "
            f"coins={self.symbols}, ema_period={ema_period}, "
            f"leverage={self.regime_leverage}"
        )

    def detect_regime(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
        coin_positions: Dict[str, str]
    ) -> str:
        """
        Detect market regime based on how many coins are above EMA.

        Args:
            date: Current date
            data: Dict of {symbol: DataFrame}
            coin_positions: Dict of {symbol: 'above' or 'below'}

        Returns:
            'harsh_bull', 'bull', 'bear', or 'harsh_bear'
        """
        above_count = sum(1 for pos in coin_positions.values() if pos == 'above')

        if above_count == 3:
            regime = 'harsh_bull'
        elif above_count == 2:
            regime = 'bull'
        elif above_count == 1:
            regime = 'bear'
        else:  # 0
            regime = 'harsh_bear'

        logger.debug(f"{date.date()}: {regime.upper()} ({above_count}/3 above EMA{self.ema_period})")
        return regime

    def get_weights(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Generate portfolio weights.

        Args:
            date: Current date
            data: Dict of {symbol: DataFrame}

        Returns:
            Dict of {symbol: weight}
        """
        try:
            ema_col = f'ema{self.ema_period}'
            weights = {}
            coin_positions = {}

            # Step 1: Determine individual coin directions
            for symbol in self.symbols:
                if symbol not in data:
                    logger.warning(f"{date.date()}: {symbol} not in data")
                    continue

                df = data[symbol]
                if df.empty or date not in df.index:
                    logger.warning(f"{date.date()}: {symbol} no data at date")
                    continue

                try:
                    close = df.loc[date, 'close']

                    if ema_col in df.columns:
                        ema = df.loc[date, ema_col]
                    else:
                        # Calculate EMA if not present
                        ema = df.loc[:date, 'close'].ewm(
                            span=self.ema_period, adjust=False
                        ).mean().iloc[-1]

                    # Determine position: above = long, below = short
                    if close > ema:
                        coin_positions[symbol] = 'above'
                        direction = +1
                    else:
                        coin_positions[symbol] = 'below'
                        direction = -1

                    # Base weight: equal weight across coins
                    base_weight = 1.0 / self.n_coins
                    weights[symbol] = direction * base_weight

                except (KeyError, ValueError, IndexError) as e:
                    logger.warning(f"{date.date()}: {symbol} error: {e}")
                    continue

            if not weights:
                logger.warning(f"{date.date()}: No weights generated")
                return {}

            # Step 2: Detect regime and get leverage multiplier
            regime = self.detect_regime(date, data, coin_positions)
            leverage_multiplier = self.regime_leverage[regime]

            # Step 3: Apply regime leverage to all weights
            final_weights = {
                symbol: weight * leverage_multiplier
                for symbol, weight in weights.items()
            }

            logger.debug(
                f"{date.date()}: {regime} regime, leverage={leverage_multiplier:.2f}, "
                f"positions={list(final_weights.keys())}"
            )

            return final_weights

        except Exception as e:
            logger.error(f"{date.date()}: Strategy failed: {e}")
            return {}

    def __repr__(self) -> str:
        return (
            f"RegimeBasedStrategy("
            f"symbols={self.symbols}, "
            f"ema_period={self.ema_period}, "
            f"leverage={self.regime_leverage})"
        )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    strategy = RegimeBasedStrategy(
        symbols=['BTCUSDT', 'ETHUSDT', 'LTCUSDT'],
        ema_period=50,
        harsh_bull_leverage=1.5,
        bull_leverage=1.0,
        bear_leverage=0.5,
        harsh_bear_leverage=1.5
    )

    print(f"Created strategy: {strategy}")
    print(f"\nRegime leverages:")
    for regime, lev in strategy.regime_leverage.items():
        print(f"  {regime}: {lev}x")
