"""
Upbit Order Manager

Handles order execution and portfolio rebalancing for Upbit exchange.
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RebalanceOrder:
    """Order details for rebalancing."""
    ticker: str
    action: str  # 'BUY' or 'SELL'
    current_value: float  # Current position value in KRW
    target_value: float  # Target position value in KRW
    diff_value: float  # Difference in KRW
    current_weight: float  # Current weight (0-1)
    target_weight: float  # Target weight (0-1)
    amount: Optional[float] = None  # Amount to trade (KRW for buy, coin units for sell)
    executed: bool = False
    error: Optional[str] = None


class UpbitOrderManager:
    """
    Manages portfolio rebalancing for Upbit exchange.

    Handles both paper trading (simulation) and live trading.
    """

    def __init__(
        self,
        upbit_client,
        paper_trading: bool = True,
        min_order_krw: float = 5000.0,
        transaction_fee: float = 0.0005
    ):
        """
        Initialize order manager.

        Args:
            upbit_client: UpbitClient instance (with or without authentication)
            paper_trading: If True, simulate orders without execution
            min_order_krw: Minimum order size in KRW (Upbit minimum is 5,000 KRW)
            transaction_fee: Transaction fee rate (default: 0.05%)
        """
        self.client = upbit_client
        self.paper_trading = paper_trading
        self.min_order_krw = min_order_krw
        self.transaction_fee = transaction_fee

        logger.info(
            f"Initialized UpbitOrderManager: "
            f"paper_trading={paper_trading}, "
            f"min_order={min_order_krw} KRW, "
            f"fee={transaction_fee*100:.2f}%"
        )

    def get_portfolio_value(self) -> Tuple[float, Dict[str, float]]:
        """
        Get total portfolio value and individual positions.

        Returns:
            Tuple of (total_value_krw, positions_dict)
            positions_dict: {ticker: value_in_krw}
        """
        # Get all balances
        balances_df = self.client.get_balances()

        if balances_df is None or balances_df.empty:
            logger.error("Failed to fetch balances")
            return 0.0, {}

        positions = {}
        total_value = 0.0

        for _, row in balances_df.iterrows():
            currency = row['currency']
            balance = float(row['balance'])
            locked = float(row.get('locked', 0))
            total_balance = balance + locked

            if total_balance < 1e-8:
                continue

            # Handle KRW separately
            if currency == 'KRW':
                positions['KRW'] = total_balance
                total_value += total_balance
            else:
                # Get current price for this currency
                ticker = f'KRW-{currency}'
                price = self.client.get_current_price(ticker)

                if price is None:
                    logger.warning(f"{ticker}: Could not get price, skipping")
                    continue

                value_krw = total_balance * price
                positions[ticker] = value_krw
                total_value += value_krw

                logger.debug(
                    f"{ticker}: {total_balance:.8f} units @ {price:,.0f} KRW = {value_krw:,.0f} KRW"
                )

        logger.info(f"Total portfolio value: {total_value:,.0f} KRW")
        return total_value, positions

    def get_current_weights(
        self,
        total_value: float,
        positions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate current portfolio weights.

        Args:
            total_value: Total portfolio value in KRW
            positions: Dict of {ticker: value_in_krw}

        Returns:
            Dict of {ticker: weight} where weight is 0-1
        """
        if total_value == 0:
            return {}

        weights = {
            ticker: value / total_value
            for ticker, value in positions.items()
        }

        return weights

    def calculate_rebalance_orders(
        self,
        target_weights: Dict[str, float],
        total_value: float,
        current_positions: Dict[str, float]
    ) -> List[RebalanceOrder]:
        """
        Calculate orders needed to rebalance to target weights.

        Args:
            target_weights: Dict of {ticker: target_weight} (0-1)
            total_value: Total portfolio value in KRW
            current_positions: Dict of {ticker: current_value_krw}

        Returns:
            List of RebalanceOrder objects
        """
        orders = []

        # Get all unique tickers
        all_tickers = set(list(target_weights.keys()) + list(current_positions.keys()))

        for ticker in all_tickers:
            current_value = current_positions.get(ticker, 0.0)
            target_weight = target_weights.get(ticker, 0.0)
            target_value = target_weight * total_value

            current_weight = current_value / total_value if total_value > 0 else 0.0
            diff_value = target_value - current_value

            # Determine action
            if abs(diff_value) < self.min_order_krw:
                # Skip if difference is too small
                continue

            if diff_value > 0:
                action = 'BUY'
            else:
                action = 'SELL'

            order = RebalanceOrder(
                ticker=ticker,
                action=action,
                current_value=current_value,
                target_value=target_value,
                diff_value=diff_value,
                current_weight=current_weight,
                target_weight=target_weight
            )

            orders.append(order)

        # Sort orders: sells first, then buys (to free up KRW)
        orders.sort(key=lambda x: (x.action == 'BUY', abs(x.diff_value)), reverse=True)

        return orders

    def execute_rebalance_orders(
        self,
        orders: List[RebalanceOrder],
        dry_run: bool = False
    ) -> Tuple[int, int, float]:
        """
        Execute rebalancing orders.

        Executes in two phases:
        1. Execute all SELL orders first (to free up KRW)
        2. Wait briefly for settlement
        3. Execute all BUY orders (using freed KRW)

        Args:
            orders: List of RebalanceOrder objects (should be pre-sorted: SELLs first)
            dry_run: If True, only log without executing

        Returns:
            Tuple of (successful_count, failed_count, total_fees_krw)
        """
        import time

        successful = 0
        failed = 0
        total_fees = 0.0

        # Get current prices for all tickers
        tickers_to_price = [o.ticker for o in orders if o.ticker != 'KRW']
        prices = self.client.get_current_prices(tickers_to_price) if tickers_to_price else {}

        # Separate SELL and BUY orders
        sell_orders = [o for o in orders if o.action == 'SELL']
        buy_orders = [o for o in orders if o.action == 'BUY']

        logger.info(f"Execution plan: {len(sell_orders)} SELLs first, then {len(buy_orders)} BUYs")

        # Phase 1: Execute all SELL orders
        logger.info("Phase 1: Executing SELL orders...")
        for order in sell_orders:
            try:
                if order.ticker == 'KRW':
                    # Can't trade KRW directly, skip
                    logger.debug("Skipping KRW (not tradeable)")
                    continue

                # Get current price
                current_price = prices.get(order.ticker)
                if current_price is None:
                    order.error = f"No price available for {order.ticker}"
                    logger.error(order.error)
                    failed += 1
                    continue

                # Calculate trade amount
                if order.action == 'BUY':
                    # Buy with KRW
                    krw_amount = abs(order.diff_value)

                    # For live trading, check actual available KRW balance
                    if not self.paper_trading and not dry_run:
                        available_krw = self.client.get_balance('KRW')
                        logger.info(f"{order.ticker}: Available KRW: {available_krw:,.0f}, Target: {krw_amount:,.0f}")
                        # Use the minimum of target amount and available KRW (minus small buffer)
                        if krw_amount > available_krw * 0.999:
                            krw_amount = available_krw * 0.999  # 0.1% buffer for safety
                            logger.warning(f"{order.ticker}: Adjusted BUY amount to available KRW: {krw_amount:,.0f}")

                    # Account for fees
                    krw_amount_with_fee = krw_amount / (1 + self.transaction_fee)

                    if krw_amount_with_fee < self.min_order_krw:
                        order.error = f"Order too small: {krw_amount_with_fee:,.0f} KRW < {self.min_order_krw} KRW"
                        logger.warning(order.error)
                        failed += 1
                        continue

                    order.amount = krw_amount_with_fee

                    if dry_run:
                        logger.info(
                            f"[DRY RUN] BUY {order.ticker}: "
                            f"{krw_amount_with_fee:,.0f} KRW "
                            f"(~{krw_amount_with_fee/current_price:.6f} units @ {current_price:,.0f} KRW)"
                        )
                        order.executed = True
                        successful += 1
                    elif not self.paper_trading:
                        # Execute live order
                        result = self.client.buy_market_order(order.ticker, krw_amount_with_fee)
                        if result and 'uuid' in result:
                            order.executed = True
                            successful += 1
                            fee = krw_amount_with_fee * self.transaction_fee
                            total_fees += fee
                            logger.info(
                                f"✓ BUY {order.ticker}: "
                                f"{krw_amount_with_fee:,.0f} KRW (fee: {fee:,.0f} KRW)"
                            )
                        else:
                            order.error = f"Order failed: {result}"
                            logger.error(order.error)
                            failed += 1
                    else:
                        # Paper trading
                        order.executed = True
                        successful += 1
                        fee = krw_amount_with_fee * self.transaction_fee
                        total_fees += fee
                        logger.info(
                            f"[PAPER] BUY {order.ticker}: "
                            f"{krw_amount_with_fee:,.0f} KRW "
                            f"(~{krw_amount_with_fee/current_price:.6f} units @ {current_price:,.0f} KRW)"
                        )

                else:  # SELL
                    # Sell coins for KRW
                    coin_amount = abs(order.diff_value) / current_price

                    # Get actual balance to verify
                    currency = order.ticker.replace('KRW-', '')
                    available_balance = self.client.get_balance(currency)

                    if coin_amount > available_balance:
                        coin_amount = available_balance
                        logger.warning(
                            f"{order.ticker}: Adjusted sell amount to available balance: {coin_amount:.8f}"
                        )

                    # Check minimum KRW value
                    krw_value = coin_amount * current_price
                    if krw_value < self.min_order_krw:
                        order.error = f"Order too small: {krw_value:,.0f} KRW < {self.min_order_krw} KRW"
                        logger.warning(order.error)
                        failed += 1
                        continue

                    order.amount = coin_amount

                    if dry_run:
                        logger.info(
                            f"[DRY RUN] SELL {order.ticker}: "
                            f"{coin_amount:.6f} units @ {current_price:,.0f} KRW "
                            f"(~{krw_value:,.0f} KRW)"
                        )
                        order.executed = True
                        successful += 1
                    elif not self.paper_trading:
                        # Execute live order
                        result = self.client.sell_market_order(order.ticker, coin_amount)
                        if result and 'uuid' in result:
                            order.executed = True
                            successful += 1
                            fee = krw_value * self.transaction_fee
                            total_fees += fee
                            logger.info(
                                f"✓ SELL {order.ticker}: "
                                f"{coin_amount:.6f} units (fee: {fee:,.0f} KRW)"
                            )
                        else:
                            order.error = f"Order failed: {result}"
                            logger.error(order.error)
                            failed += 1
                    else:
                        # Paper trading
                        order.executed = True
                        successful += 1
                        fee = krw_value * self.transaction_fee
                        total_fees += fee
                        logger.info(
                            f"[PAPER] SELL {order.ticker}: "
                            f"{coin_amount:.6f} units @ {current_price:,.0f} KRW "
                            f"(~{krw_value:,.0f} KRW)"
                        )

            except Exception as e:
                order.error = f"Exception: {str(e)}"
                logger.error(f"Error executing order for {order.ticker}: {e}")
                failed += 1

        # Phase 2: Wait for SELL orders to settle, then execute BUY orders
        if buy_orders and sell_orders and not dry_run and not self.paper_trading:
            logger.info(f"Waiting 2 seconds for SELL orders to settle...")
            time.sleep(2)
            logger.info("Phase 2: Executing BUY orders...")

        for order in buy_orders:
            try:
                if order.ticker == 'KRW':
                    continue

                # Get current price
                current_price = prices.get(order.ticker)
                if current_price is None:
                    order.error = f"No price available for {order.ticker}"
                    logger.error(order.error)
                    failed += 1
                    continue

                # Buy with KRW
                krw_amount = abs(order.diff_value)

                # For live trading, check actual available KRW balance
                if not self.paper_trading and not dry_run:
                    available_krw = self.client.get_balance('KRW')
                    logger.info(f"{order.ticker}: Available KRW: {available_krw:,.0f}, Target: {krw_amount:,.0f}")
                    # Use the minimum of target amount and available KRW (minus small buffer)
                    if krw_amount > available_krw * 0.999:
                        krw_amount = available_krw * 0.999  # 0.1% buffer for safety
                        logger.warning(f"{order.ticker}: Adjusted BUY amount to available KRW: {krw_amount:,.0f}")

                # Account for fees
                krw_amount_with_fee = krw_amount / (1 + self.transaction_fee)

                if krw_amount_with_fee < self.min_order_krw:
                    order.error = f"Order too small: {krw_amount_with_fee:,.0f} KRW < {self.min_order_krw} KRW"
                    logger.warning(order.error)
                    failed += 1
                    continue

                order.amount = krw_amount_with_fee

                if dry_run:
                    logger.info(
                        f"[DRY RUN] BUY {order.ticker}: "
                        f"{krw_amount_with_fee:,.0f} KRW "
                        f"(~{krw_amount_with_fee/current_price:.6f} units @ {current_price:,.0f} KRW)"
                    )
                    order.executed = True
                    successful += 1
                elif not self.paper_trading:
                    # Execute live order
                    result = self.client.buy_market_order(order.ticker, krw_amount_with_fee)
                    if result and 'uuid' in result:
                        order.executed = True
                        successful += 1
                        fee = krw_amount_with_fee * self.transaction_fee
                        total_fees += fee
                        logger.info(
                            f"✓ BUY {order.ticker}: "
                            f"{krw_amount_with_fee:,.0f} KRW (fee: {fee:,.0f} KRW)"
                        )
                    else:
                        order.error = f"Order failed: {result}"
                        logger.error(order.error)
                        failed += 1
                else:
                    # Paper trading
                    order.executed = True
                    successful += 1
                    fee = krw_amount_with_fee * self.transaction_fee
                    total_fees += fee
                    logger.info(
                        f"[PAPER] BUY {order.ticker}: "
                        f"{krw_amount_with_fee:,.0f} KRW "
                        f"(~{krw_amount_with_fee/current_price:.6f} units @ {current_price:,.0f} KRW)"
                    )

            except Exception as e:
                order.error = f"Exception: {str(e)}"
                logger.error(f"Error executing BUY order for {order.ticker}: {e}")
                failed += 1

        return successful, failed, total_fees

    def print_rebalance_plan(
        self,
        orders: List[RebalanceOrder],
        total_value: float
    ) -> None:
        """
        Print rebalancing plan in a formatted table.

        Args:
            orders: List of RebalanceOrder objects
            total_value: Total portfolio value in KRW
        """
        print("\n" + "=" * 100)
        print("REBALANCING PLAN")
        print("=" * 100)
        print(f"Total Portfolio Value: {total_value:,.0f} KRW")
        print(f"Number of trades: {len(orders)}")
        print()

        if not orders:
            print("No rebalancing needed - all positions within tolerance")
            print("=" * 100)
            return

        # Print orders
        print(f"{'Ticker':<12} {'Action':<6} {'Current':<12} {'Target':<12} {'Diff (KRW)':<15} {'Current%':<10} {'Target%':<10}")
        print("-" * 100)

        for order in orders:
            print(
                f"{order.ticker:<12} "
                f"{order.action:<6} "
                f"{order.current_value:>11,.0f} "
                f"{order.target_value:>11,.0f} "
                f"{order.diff_value:>+14,.0f} "
                f"{order.current_weight:>9.1%} "
                f"{order.target_weight:>9.1%}"
            )

        print("=" * 100)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    from CoinTrading.data.upbit_client import UpbitClient

    print("UpbitOrderManager Example (Paper Trading)")
    print("=" * 80)

    # Create client (no authentication needed for paper trading)
    client = UpbitClient()

    # Create order manager in paper trading mode
    order_manager = UpbitOrderManager(
        upbit_client=client,
        paper_trading=True,
        min_order_krw=5000.0,
        transaction_fee=0.0005
    )

    # Simulate portfolio (for demonstration)
    print("\nSimulated Portfolio:")
    total_value = 1000000.0  # 1M KRW
    current_positions = {
        'KRW': 300000,  # 30% cash
        'KRW-BTC': 400000,  # 40% BTC
        'KRW-ETH': 200000,  # 20% ETH
        'KRW-USDT': 100000  # 10% USDT
    }

    current_weights = order_manager.get_current_weights(total_value, current_positions)
    print(f"Total value: {total_value:,.0f} KRW")
    for ticker, weight in current_weights.items():
        print(f"  {ticker}: {weight:.1%}")

    # Target weights (25/25/25/25)
    target_weights = {
        'KRW-USDT': 0.25,
        'KRW-BTC': 0.25,
        'KRW-ETH': 0.25,
        'KRW-LTC': 0.25
    }

    print("\nTarget Allocation:")
    for ticker, weight in target_weights.items():
        print(f"  {ticker}: {weight:.1%}")

    # Calculate rebalance orders
    orders = order_manager.calculate_rebalance_orders(
        target_weights,
        total_value,
        current_positions
    )

    # Print plan
    order_manager.print_rebalance_plan(orders, total_value)

    print("\n✓ UpbitOrderManager ready!")
