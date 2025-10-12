"""
Trader

Main interface for live trading with strategy integration.
"""

from typing import Dict, Optional, Any, Callable
import pandas as pd
import logging
import time

from .order_manager import OrderManager, Order
from .risk_manager import RiskManager, RiskLimits

logger = logging.getLogger(__name__)


class Trader:
    """
    Main trading interface.

    Integrates strategy, risk management, and order execution.
    """

    def __init__(
        self,
        strategy: Any,
        data_loader: Any,
        order_manager: OrderManager,
        risk_manager: RiskManager,
        initial_capital: float = 10000.0,
        rebalance_interval_hours: int = 24,
        dry_run: bool = True
    ):
        """
        Initialize trader.

        Args:
            strategy: Strategy instance with get_weights() method
            data_loader: DataLoader instance for fetching market data
            order_manager: OrderManager instance
            risk_manager: RiskManager instance
            initial_capital: Initial capital (USDT)
            rebalance_interval_hours: Hours between rebalances
            dry_run: If True, log trades but don't execute
        """
        self.strategy = strategy
        self.data_loader = data_loader
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.rebalance_interval_hours = rebalance_interval_hours
        self.dry_run = dry_run

        # State
        self.current_equity = initial_capital
        self.start_of_day_equity = initial_capital
        self.last_rebalance: Optional[pd.Timestamp] = None

        # Performance tracking
        self.equity_history: pd.Series = pd.Series(dtype=float)
        self.trade_history: list = []

        logger.info(
            f"Initialized Trader: capital=${initial_capital:.2f}, "
            f"rebalance_interval={rebalance_interval_hours}h, dry_run={dry_run}"
        )

    def update_equity(
        self,
        current_date: pd.Timestamp
    ) -> float:
        """
        Update current portfolio equity.

        Args:
            current_date: Current date

        Returns:
            Current equity value
        """
        positions = self.order_manager.get_current_positions()

        if not positions:
            self.current_equity = self.initial_capital
            return self.current_equity

        # Get current prices
        symbols = list(positions.keys())
        prices = self.order_manager.get_current_prices(symbols)

        # Calculate equity
        equity = 0.0
        for symbol, quantity in positions.items():
            if symbol in prices:
                equity += quantity * prices[symbol]
            else:
                logger.warning(f"No price for {symbol}, using last known value")

        self.current_equity = equity

        # Update risk manager
        self.risk_manager.update_daily_pnl(equity, current_date)

        # Track equity history
        self.equity_history[current_date] = equity

        return equity

    def should_rebalance(
        self,
        current_date: pd.Timestamp
    ) -> bool:
        """
        Check if it's time to rebalance.

        Args:
            current_date: Current date

        Returns:
            True if should rebalance
        """
        if self.last_rebalance is None:
            return True

        hours_since_last = (current_date - self.last_rebalance).total_seconds() / 3600

        return hours_since_last >= self.rebalance_interval_hours

    def rebalance(
        self,
        current_date: pd.Timestamp,
        data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Rebalance portfolio to strategy target weights.

        Args:
            current_date: Current date
            data: Market data dict {symbol: DataFrame}

        Returns:
            True if rebalance successful
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Rebalancing portfolio at {current_date}")
        logger.info(f"{'='*80}")

        # Update equity
        self.update_equity(current_date)
        logger.info(f"Current equity: ${self.current_equity:.2f}")

        # Risk checks
        is_valid, reason = self.risk_manager.check_daily_loss_limit(
            self.current_equity,
            self.start_of_day_equity
        )
        if not is_valid:
            logger.error(f"Daily loss limit check failed: {reason}")
            return False

        is_valid, reason = self.risk_manager.check_drawdown_limit(
            self.current_equity
        )
        if not is_valid:
            logger.warning(f"Drawdown limit check failed: {reason}")
            # Continue but log warning

        # Get strategy target weights
        try:
            target_weights = self.strategy.get_weights(current_date, data)
            logger.info(f"Strategy target weights: {target_weights}")
        except Exception as e:
            logger.error(f"Error getting strategy weights: {e}")
            return False

        # Validate and adjust weights
        is_valid, reason, adjusted_weights = self.risk_manager.validate_position_weights(
            target_weights,
            self.current_equity
        )

        if not is_valid:
            logger.error(f"Weight validation failed: {reason}")
            return False

        if reason:
            logger.warning(f"Weights adjusted: {reason}")
            logger.info(f"Adjusted weights: {adjusted_weights}")

        # Get current prices
        symbols = list(adjusted_weights.keys())
        prices = self.order_manager.get_current_prices(symbols)

        if not prices:
            logger.error("Failed to get current prices")
            return False

        # Execute rebalance
        if self.dry_run:
            logger.info("DRY RUN - No orders will be executed")
            self._log_target_positions(adjusted_weights, prices)
            success = True
        else:
            orders = self.order_manager.rebalance_to_target_weights(
                adjusted_weights,
                self.current_equity,
                prices
            )
            self.trade_history.extend(orders)
            success = len(orders) > 0

        if success:
            self.last_rebalance = current_date
            logger.info("Rebalance completed successfully")
        else:
            logger.error("Rebalance failed")

        return success

    def _log_target_positions(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float]
    ) -> None:
        """
        Log target positions (for dry run).

        Args:
            target_weights: Target weights
            prices: Current prices
        """
        logger.info("\nTarget positions:")
        for symbol, weight in sorted(target_weights.items(), key=lambda x: abs(x[1]), reverse=True):
            if symbol in prices:
                notional = weight * self.current_equity
                quantity = notional / prices[symbol] if prices[symbol] > 0 else 0
                logger.info(
                    f"  {symbol:12s} | Weight: {weight:6.2f} | "
                    f"Notional: ${notional:9.2f} | Qty: {quantity:10.4f}"
                )

    def run_live(
        self,
        start_date: Optional[pd.Timestamp] = None,
        check_interval_seconds: int = 60
    ) -> None:
        """
        Run live trading loop.

        Args:
            start_date: Start date (default: now)
            check_interval_seconds: Seconds between checks
        """
        if start_date is None:
            start_date = pd.Timestamp.now()

        logger.info(f"\n{'='*80}")
        logger.info(f"Starting live trading from {start_date}")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE TRADING'}")
        logger.info(f"{'='*80}\n")

        self.start_of_day_equity = self.current_equity

        try:
            while True:
                current_date = pd.Timestamp.now()

                # Update equity
                self.update_equity(current_date)

                # Check if need to rebalance
                if self.should_rebalance(current_date):
                    # Load latest market data
                    try:
                        # Get universe symbols from strategy
                        universe = self.data_loader.load_multiple(
                            symbols=None,  # Load all available
                            skip_errors=True
                        )

                        # Rebalance
                        self.rebalance(current_date, universe)

                    except Exception as e:
                        logger.error(f"Error during rebalance: {e}")

                # Sleep until next check
                time.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            logger.info("\nTrading stopped by user")
            self._print_summary()

    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Run backtest simulation.

        Args:
            data: Historical market data
            start_date: Start date
            end_date: End date

        Returns:
            Equity curve
        """
        logger.info(f"\n{'='*80}")
        logger.info("Running backtest")
        logger.info(f"{'='*80}\n")

        # Get common dates
        all_dates = None
        for df in data.values():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates &= set(df.index)

        dates = pd.DatetimeIndex(sorted(all_dates))

        if start_date:
            dates = dates[dates >= start_date]
        if end_date:
            dates = dates[dates <= end_date]

        # Run through history
        for date in dates:
            if self.should_rebalance(date):
                self.rebalance(date, data)

        self._print_summary()

        return self.equity_history

    def _print_summary(self) -> None:
        """Print trading summary."""
        logger.info(f"\n{'='*80}")
        logger.info("Trading Summary")
        logger.info(f"{'='*80}")
        logger.info(f"Initial capital: ${self.initial_capital:.2f}")
        logger.info(f"Final equity: ${self.current_equity:.2f}")
        logger.info(f"Total return: {(self.current_equity/self.initial_capital - 1)*100:.2f}%")
        logger.info(f"Number of trades: {len(self.trade_history)}")

        if len(self.equity_history) > 1:
            from ..backtesting.metrics import calculate_cagr, calculate_mdd, calculate_sharpe

            returns = self.equity_history.pct_change().dropna()
            cagr = calculate_cagr(self.equity_history) * 100
            mdd = calculate_mdd(self.equity_history) * 100
            sharpe = calculate_sharpe(returns)

            logger.info(f"CAGR: {cagr:.2f}%")
            logger.info(f"MDD: {mdd:.2f}%")
            logger.info(f"Sharpe: {sharpe:.2f}")

        logger.info(f"{'='*80}\n")

    def get_equity_curve(self) -> pd.Series:
        """Get equity curve."""
        return self.equity_history.copy()

    def get_trade_history(self) -> list:
        """Get trade history."""
        return self.trade_history.copy()

    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.order_manager.get_current_positions()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Trader example - see notebooks for full demonstration")
    print()
    print("This module provides:")
    print("  - Trader: Main trading interface")
    print("  - run_live(): Live trading loop")
    print("  - run_backtest(): Backtest simulation")
    print("  - Integrated risk management and order execution")
    print()
    print("✓ Trader module ready!")
