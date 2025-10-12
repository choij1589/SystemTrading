"""
Order Manager

Handles order execution and position tracking.
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class Order:
    """
    Order details.

    Attributes:
        symbol: Trading symbol
        side: BUY or SELL
        order_type: MARKET or LIMIT
        quantity: Order quantity
        price: Limit price (None for market orders)
        status: Order status
        filled_quantity: Quantity filled
        average_price: Average fill price
        order_id: Exchange order ID
        timestamp: Order creation time
        error: Error message (if failed)
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    order_id: Optional[str] = None
    timestamp: Optional[pd.Timestamp] = None
    error: Optional[str] = None


class OrderManager:
    """
    Manages order execution and position tracking.

    Supports both live trading and paper trading modes.
    """

    def __init__(
        self,
        binance_client: Optional[any] = None,
        paper_trading: bool = True,
        transaction_fee: float = 0.003
    ):
        """
        Initialize order manager.

        Args:
            binance_client: Binance client for live trading (optional)
            paper_trading: If True, simulate orders without execution
            transaction_fee: Transaction fee per trade
        """
        self.client = binance_client
        self.paper_trading = paper_trading
        self.transaction_fee = transaction_fee

        # Order tracking
        self.orders: List[Order] = []
        self.positions: Dict[str, float] = {}  # {symbol: quantity}

        logger.info(
            f"Initialized OrderManager: paper_trading={paper_trading}, "
            f"fee={transaction_fee*100:.2f}%"
        )

    def get_current_positions(self) -> Dict[str, float]:
        """
        Get current positions.

        Returns:
            Dict of {symbol: quantity}
        """
        if self.paper_trading:
            return self.positions.copy()
        else:
            # Query exchange for live positions
            if self.client is None:
                logger.error("No Binance client provided for live trading")
                return {}

            try:
                positions = {}
                account = self.client.futures_account()

                for position in account.get('positions', []):
                    symbol = position['symbol']
                    quantity = float(position['positionAmt'])

                    if quantity != 0:
                        positions[symbol] = quantity

                self.positions = positions
                return positions

            except Exception as e:
                logger.error(f"Error fetching positions: {e}")
                return self.positions.copy()

    def get_current_prices(
        self,
        symbols: List[str]
    ) -> Dict[str, float]:
        """
        Get current market prices.

        Args:
            symbols: List of symbols

        Returns:
            Dict of {symbol: price}
        """
        if self.client is None:
            logger.error("No Binance client provided")
            return {}

        try:
            tickers = self.client.futures_mark_price()
            prices = {}

            for ticker in tickers:
                symbol = ticker['symbol']
                if symbol in symbols:
                    prices[symbol] = float(ticker['markPrice'])

            return prices

        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return {}

    def calculate_order_quantity(
        self,
        symbol: str,
        target_notional: float,
        current_price: float,
        current_quantity: float = 0.0
    ) -> Tuple[OrderSide, float]:
        """
        Calculate order quantity needed to reach target.

        Args:
            symbol: Trading symbol
            target_notional: Target position value (USDT)
            current_price: Current asset price
            current_quantity: Current position quantity

        Returns:
            Tuple of (side, quantity)
        """
        # Target quantity
        target_quantity = target_notional / current_price if current_price > 0 else 0

        # Quantity to trade
        quantity_diff = target_quantity - current_quantity

        # Determine side
        if quantity_diff > 0:
            side = OrderSide.BUY
            quantity = abs(quantity_diff)
        else:
            side = OrderSide.SELL
            quantity = abs(quantity_diff)

        return side, quantity

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None
    ) -> Order:
        """
        Create an order.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            order_type: MARKET or LIMIT
            price: Limit price (for limit orders)

        Returns:
            Order object
        """
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=pd.Timestamp.now()
        )

        self.orders.append(order)
        return order

    def execute_order(
        self,
        order: Order,
        current_price: Optional[float] = None
    ) -> bool:
        """
        Execute an order.

        Args:
            order: Order to execute
            current_price: Current market price (for paper trading)

        Returns:
            True if successful
        """
        if self.paper_trading:
            return self._execute_paper_order(order, current_price)
        else:
            return self._execute_live_order(order)

    def _execute_paper_order(
        self,
        order: Order,
        current_price: Optional[float]
    ) -> bool:
        """
        Simulate order execution (paper trading).

        Args:
            order: Order to execute
            current_price: Current market price

        Returns:
            True if successful
        """
        if current_price is None:
            order.status = OrderStatus.FAILED
            order.error = "No current price provided for paper trading"
            logger.error(order.error)
            return False

        try:
            # Simulate fill
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_price = current_price
            order.order_id = f"PAPER_{order.timestamp.strftime('%Y%m%d%H%M%S')}_{order.symbol}"

            # Update positions
            current_qty = self.positions.get(order.symbol, 0.0)

            if order.side == OrderSide.BUY:
                new_qty = current_qty + order.quantity
            else:  # SELL
                new_qty = current_qty - order.quantity

            if abs(new_qty) < 1e-8:
                self.positions.pop(order.symbol, None)
            else:
                self.positions[order.symbol] = new_qty

            logger.info(
                f"Paper order filled: {order.symbol} {order.side.value} "
                f"{order.quantity:.4f} @ {current_price:.2f}"
            )

            return True

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error = str(e)
            logger.error(f"Paper order failed: {e}")
            return False

    def _execute_live_order(
        self,
        order: Order
    ) -> bool:
        """
        Execute live order on exchange.

        Args:
            order: Order to execute

        Returns:
            True if successful
        """
        if self.client is None:
            order.status = OrderStatus.FAILED
            order.error = "No Binance client provided"
            logger.error(order.error)
            return False

        try:
            # Place order on exchange
            if order.order_type == OrderType.MARKET:
                response = self.client.futures_create_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    type='MARKET',
                    quantity=order.quantity
                )
            else:  # LIMIT
                response = self.client.futures_create_order(
                    symbol=order.symbol,
                    side=order.side.value,
                    type='LIMIT',
                    quantity=order.quantity,
                    price=order.price,
                    timeInForce='GTC'
                )

            # Update order status
            order.order_id = response['orderId']
            order.status = OrderStatus.FILLED
            order.filled_quantity = float(response.get('executedQty', order.quantity))
            order.average_price = float(response.get('avgPrice', 0))

            # Update positions
            self.get_current_positions()  # Refresh from exchange

            logger.info(
                f"Live order filled: {order.symbol} {order.side.value} "
                f"{order.filled_quantity:.4f} @ {order.average_price:.2f}"
            )

            return True

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.error = str(e)
            logger.error(f"Live order failed: {e}")
            return False

    def rebalance_to_target_weights(
        self,
        target_weights: Dict[str, float],
        current_equity: float,
        current_prices: Dict[str, float]
    ) -> List[Order]:
        """
        Rebalance portfolio to target weights.

        Args:
            target_weights: Dict of {symbol: weight}
            current_equity: Current portfolio value
            current_prices: Dict of {symbol: price}

        Returns:
            List of executed orders
        """
        executed_orders = []
        current_positions = self.get_current_positions()

        for symbol, target_weight in target_weights.items():
            if symbol not in current_prices:
                logger.warning(f"No price available for {symbol}, skipping")
                continue

            current_price = current_prices[symbol]
            current_quantity = current_positions.get(symbol, 0.0)
            target_notional = target_weight * current_equity

            # Calculate order
            side, quantity = self.calculate_order_quantity(
                symbol,
                target_notional,
                current_price,
                current_quantity
            )

            # Skip if quantity too small
            if quantity < 1e-8:
                continue

            # Create and execute order
            order = self.create_order(symbol, side, quantity)
            success = self.execute_order(order, current_price)

            if success:
                executed_orders.append(order)

        # Close positions not in target
        for symbol in current_positions:
            if symbol not in target_weights:
                current_quantity = current_positions[symbol]

                if abs(current_quantity) < 1e-8:
                    continue

                if symbol not in current_prices:
                    logger.warning(f"No price available for {symbol}, cannot close")
                    continue

                current_price = current_prices[symbol]
                side = OrderSide.SELL if current_quantity > 0 else OrderSide.BUY

                order = self.create_order(symbol, side, abs(current_quantity))
                success = self.execute_order(order, current_price)

                if success:
                    executed_orders.append(order)

        return executed_orders

    def get_order_history(
        self,
        symbol: Optional[str] = None
    ) -> List[Order]:
        """
        Get order history.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of orders
        """
        if symbol is None:
            return self.orders.copy()
        else:
            return [o for o in self.orders if o.symbol == symbol]

    def clear_history(self) -> None:
        """Clear order history."""
        self.orders.clear()
        logger.info("Order history cleared")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Order Manager example - Paper Trading Mode")
    print()

    # Create order manager (paper trading)
    order_manager = OrderManager(
        paper_trading=True,
        transaction_fee=0.003
    )

    # Simulate some orders
    current_prices = {
        'BTCUSDT': 50000.0,
        'ETHUSDT': 3000.0,
    }

    target_weights = {
        'BTCUSDT': 0.5,   # 50% of portfolio
        'ETHUSDT': 0.3,   # 30% of portfolio
    }

    current_equity = 10000.0

    print(f"Current equity: ${current_equity:.2f}")
    print(f"Target weights: {target_weights}")
    print()

    # Rebalance
    orders = order_manager.rebalance_to_target_weights(
        target_weights,
        current_equity,
        current_prices
    )

    print(f"\nExecuted {len(orders)} orders:")
    for order in orders:
        print(f"  {order.symbol} {order.side.value} {order.quantity:.4f} @ ${order.average_price:.2f}")

    print(f"\nCurrent positions: {order_manager.get_current_positions()}")
    print()
    print("✓ Order manager ready!")
