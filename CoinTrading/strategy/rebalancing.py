"""
Periodic Rebalancing Strategy

Simple strategy that maintains fixed allocation across multiple assets
and rebalances periodically to target weights.
"""

from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RebalancingStrategy:
    """
    Periodic rebalancing strategy with optional EMA timing filter.

    Maintains fixed target allocation and rebalances every N days.
    Between rebalancing dates, positions drift naturally with price changes.

    Example (basic):
        strategy = RebalancingStrategy(
            target_weights={'BTCUSDT': 0.33, 'ETHUSDT': 0.33, 'LTCUSDT': 0.34},
            rebalance_period=7
        )

    Example (with EMA timing):
        strategy = RebalancingStrategy(
            target_weights={'USDT': 0.25, 'BTCUSDT': 0.25, 'ETHUSDT': 0.25, 'LTCUSDT': 0.25},
            rebalance_period=7,
            use_ema_timing=True,
            ema_period=50
        )

    Features:
    - Configurable target weights for each asset
    - Configurable rebalancing period (days)
    - Optional EMA trend filter (checked only on rebalancing days)
    - Supports USDT as cash (treated as constant value 1.0)
    - Tracks position units to calculate drift between rebalances
    - Returns empty dict on non-rebalancing days (no trading)
    - Returns adjusted weights on rebalancing days (triggers trades)

    EMA Timing Behavior:
    - On each rebalancing day, check if coin price >= EMA
    - Coins below EMA are excluded, their weight goes to USDT
    - Between rebalancing days, positions held unchanged (no daily checks)
    """

    def __init__(
        self,
        target_weights: Dict[str, float] = None,
        rebalance_period: int = 7,
        use_ema_timing: bool = False,
        ema_period: int = 50,
        name: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize rebalancing strategy.

        Args:
            target_weights: Target allocation for each asset.
                          Default: {'USDT': 0.25, 'BTCUSDT': 0.25,
                                   'ETHUSDT': 0.25, 'LTCUSDT': 0.25}
            rebalance_period: Days between rebalancing (default: 7)
            use_ema_timing: Enable EMA trend filter (default: False)
            ema_period: EMA period for trend detection (default: 50)
            name: Strategy name
            config: Configuration dictionary
        """
        self.name = name or "Rebalancing"
        self.config = config

        # Default allocation: 25% each across USDT, BTC, ETH, LTC
        if target_weights is None:
            self.target_weights = {
                'USDT': 0.25,
                'BTCUSDT': 0.25,
                'ETHUSDT': 0.25,
                'LTCUSDT': 0.25
            }
        else:
            self.target_weights = target_weights

        # Validate weights sum to 1.0
        total_weight = sum(self.target_weights.values())
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point error
            logger.warning(
                f"Target weights sum to {total_weight:.4f}, not 1.0. "
                f"Normalizing to 100%."
            )
            # Normalize weights
            self.target_weights = {
                k: v / total_weight
                for k, v in self.target_weights.items()
            }

        self.rebalance_period = rebalance_period
        self.use_ema_timing = use_ema_timing
        self.ema_period = ema_period
        self.last_rebalance_date = None

        # Track position units (shares/coins held) for calculating drift
        self.position_units: Dict[str, float] = {}

        logger.info(
            f"Initialized {self.name}: "
            f"weights={self.target_weights}, "
            f"rebalance_period={rebalance_period} days, "
            f"ema_timing={use_ema_timing}"
        )

    def should_rebalance(self, date: pd.Timestamp) -> bool:
        """
        Determine if we should rebalance on this date.

        Args:
            date: Current date

        Returns:
            True if should rebalance, False otherwise
        """
        if self.last_rebalance_date is None:
            # First trade - always rebalance
            return True

        days_since_rebalance = (date - self.last_rebalance_date).days
        return days_since_rebalance >= self.rebalance_period

    def _apply_ema_filter(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
        target_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply EMA trend filter to target weights.

        For each crypto asset:
        - If price >= EMA: Keep in portfolio
        - If price < EMA: Exclude (weight = 0)

        Excluded coins' weights are redistributed to USDT.

        Args:
            date: Current date
            data: Price data with EMA indicators
            target_weights: Original target weights

        Returns:
            Adjusted weights with EMA filter applied
        """
        ema_col = f'ema{self.ema_period}'

        # Identify active coins (price >= EMA)
        active_coins = {}
        excluded_weight = 0.0

        for symbol, weight in target_weights.items():
            if symbol == 'USDT':
                # USDT always included (cash)
                active_coins[symbol] = weight
                continue

            # Check if coin has EMA data
            if symbol not in data:
                logger.warning(f"{symbol}: Not in data, excluding")
                excluded_weight += weight
                continue

            df = data[symbol]

            if ema_col not in df.columns:
                logger.warning(f"{symbol}: No {ema_col} data, excluding")
                excluded_weight += weight
                continue

            if date not in df.index:
                logger.warning(f"{symbol}: No data for {date.date()}, excluding")
                excluded_weight += weight
                continue

            price = df.loc[date, 'close']
            ema = df.loc[date, ema_col]

            if price >= ema:
                # Coin passes filter
                active_coins[symbol] = weight
                logger.debug(f"{symbol}: Active (price={price:.2f} >= EMA={ema:.2f})")
            else:
                # Coin filtered out
                excluded_weight += weight
                logger.info(f"{symbol}: Filtered out (price={price:.2f} < EMA={ema:.2f})")

        # Add excluded weight to USDT
        if 'USDT' in active_coins:
            active_coins['USDT'] += excluded_weight
        else:
            active_coins['USDT'] = excluded_weight

        logger.info(
            f"{date.date()}: Active coins: {list(active_coins.keys())}, "
            f"USDT: {active_coins.get('USDT', 0):.1%}"
        )

        return active_coins

    def get_weights(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Get portfolio weights for the given date.

        With EMA timing enabled:
        - Checks EMA filter daily and trades accordingly (Option 1)
        - On rebalancing days: Full rebalance to target weights
        - On non-rebalancing days: Adjust holdings based on EMA filter

        Without EMA timing:
        - On rebalancing days: Returns target weights (triggers trades)
        - On non-rebalancing days: Returns empty dict (no trades, positions drift)

        Args:
            date: Current date
            data: Dictionary of DataFrames with OHLCV data

        Returns:
            Dictionary mapping symbol to target weight (or empty dict)
        """
        # Get current prices for all assets
        current_prices = {}
        for symbol in self.target_weights.keys():
            if symbol == 'USDT':
                current_prices[symbol] = 1.0
            elif symbol in data:
                df = data[symbol]
                if date in df.index:
                    current_prices[symbol] = df.loc[date, 'close']
                else:
                    logger.warning(f"{symbol} has no price data for {date.date()}")
            else:
                logger.warning(f"{symbol} not in data on {date.date()}")

        # First call - initialize positions
        if not self.position_units:
            self.last_rebalance_date = date

            # Apply EMA filter if enabled
            if self.use_ema_timing:
                adjusted_weights = self._apply_ema_filter(date, data, self.target_weights)
            else:
                adjusted_weights = self.target_weights.copy()

            # Initialize position units proportional to target weight / price
            for symbol, target_weight in adjusted_weights.items():
                if symbol in current_prices:
                    self.position_units[symbol] = target_weight / current_prices[symbol]

            logger.info(f"{date.date()}: Initial allocation to adjusted weights")
            return adjusted_weights

        # Calculate current portfolio value from position units
        current_values = {}
        for symbol, units in self.position_units.items():
            if symbol in current_prices:
                current_values[symbol] = units * current_prices[symbol]

        total_value = sum(current_values.values())
        if total_value == 0:
            logger.warning(f"{date.date()}: Portfolio value is zero")
            return {}

        # Calculate current weights after drift
        current_weights = {s: v / total_value for s, v in current_values.items()}

        # Apply EMA filter if enabled (daily check for Option 1)
        if self.use_ema_timing:
            adjusted_weights = self._apply_ema_filter(date, data, self.target_weights)
        else:
            adjusted_weights = self.target_weights.copy()

        # Check if we should rebalance
        if self.should_rebalance(date):
            self.last_rebalance_date = date

            # Update position units to match adjusted weights
            # New units = (adjusted_weight * total_value) / price
            for symbol, target_weight in adjusted_weights.items():
                if symbol in current_prices:
                    self.position_units[symbol] = (target_weight * total_value) / current_prices[symbol]

            logger.info(f"{date.date()}: Rebalancing to adjusted weights")
            return adjusted_weights

        else:
            # Non-rebalancing day
            if self.use_ema_timing:
                # Daily EMA timing (Option 1): Return adjusted weights to trigger trades
                # This allows coins to be sold when they drop below EMA50,
                # and bought back when they go above EMA50

                # Update position units to match adjusted weights
                for symbol, target_weight in adjusted_weights.items():
                    if symbol in current_prices:
                        self.position_units[symbol] = (target_weight * total_value) / current_prices[symbol]

                logger.debug(f"{date.date()}: Daily EMA check")
                return adjusted_weights
            else:
                # No EMA timing - hold positions (drift naturally)
                logger.debug(
                    f"{date.date()}: Maintaining positions "
                    f"(days since rebalance: {(date - self.last_rebalance_date).days})"
                )
                return {}  # Empty dict signals "don't trade"

    def __repr__(self):
        """String representation of the strategy."""
        weights_str = ', '.join([f'{k}: {v:.1%}' for k, v in self.target_weights.items()])
        timing_str = f", EMA{self.ema_period}_timing" if self.use_ema_timing else ""
        return (
            f"{self.name}(weights={{{weights_str}}}, "
            f"rebalance_every={self.rebalance_period}d{timing_str})"
        )
