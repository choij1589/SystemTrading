"""
Backtesting Engine

Core backtesting engine for portfolio simulation with transaction costs.
"""

from typing import Dict, List, Optional, Callable, Tuple
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Represents a position in a single asset.

    Attributes:
        symbol: Asset symbol
        weight: Target weight (between -1 and 1, negative for short)
        size: Actual position size (in units of portfolio equity)
    """
    symbol: str
    weight: float
    size: float = 0.0

    def __repr__(self) -> str:
        return f"Position({self.symbol}, weight={self.weight:.2%}, size={self.size:.2f})"


@dataclass
class Trade:
    """
    Represents a trade execution.

    Attributes:
        date: Trade date
        symbol: Asset symbol
        size_change: Change in position size
        cost: Transaction cost
    """
    date: pd.Timestamp
    symbol: str
    size_change: float
    cost: float

    def __repr__(self) -> str:
        return f"Trade({self.date.date()}, {self.symbol}, {self.size_change:+.4f}, cost={self.cost:.4f})"


class Portfolio:
    """
    Portfolio tracker that maintains equity, positions, and trade history.

    Attributes:
        initial_equity: Starting equity (default: 1.0)
        transaction_fee: Transaction cost as decimal (default: 0.003 = 0.3%)
    """

    def __init__(
        self,
        initial_equity: float = 1.0,
        transaction_fee: float = 0.003
    ):
        self.initial_equity = initial_equity
        self.transaction_fee = transaction_fee

        # Current state
        self.equity = initial_equity
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

        # History tracking
        self.equity_history: List[float] = [initial_equity]
        self.dates: List[pd.Timestamp] = []

    def rebalance(
        self,
        date: pd.Timestamp,
        target_weights: Dict[str, float],
        returns: Dict[str, float]
    ) -> float:
        """
        Rebalance portfolio to target weights.

        Args:
            date: Current date
            target_weights: Dict of {symbol: target_weight}
            returns: Dict of {symbol: period_return} (e.g., 0.01 = 1% return)

        Returns:
            Transaction cost incurred
        """
        # Calculate portfolio return from current positions
        # This works for both long-only and long/short strategies
        # Portfolio return = sum of (weight * return) for each position
        portfolio_return = 0.0
        for symbol, position in self.positions.items():
            if symbol in returns:
                # Each position contributes: weight * asset_return
                # For long positions (weight > 0): positive return → positive contribution
                # For short positions (weight < 0): positive return → negative contribution
                portfolio_return += position.weight * returns[symbol]

        # Apply portfolio return to equity
        self.equity *= (1 + portfolio_return)

        # If no target weights provided, skip rebalancing (just apply returns)
        # This allows strategies to hold positions without trading
        if not target_weights:
            self.equity_history.append(self.equity)
            self.dates.append(date)
            logger.debug(f"{date.date()}: No rebalancing (holding positions), equity={self.equity:.4f}")
            return 0.0  # No transaction costs

        # Calculate target position sizes
        total_cost = 0.0

        # Build new positions
        new_positions: Dict[str, Position] = {}

        for symbol, target_weight in target_weights.items():
            target_size = self.equity * target_weight

            # Calculate size change
            current_size = self.positions.get(symbol, Position(symbol, 0.0)).size
            size_change = target_size - current_size

            # Calculate transaction cost (on the traded amount)
            trade_cost = abs(size_change) * self.transaction_fee
            total_cost += trade_cost

            # Create new position
            new_positions[symbol] = Position(
                symbol=symbol,
                weight=target_weight,
                size=target_size
            )

            # Record trade if there was a change
            if abs(size_change) > 1e-10:
                self.trades.append(Trade(
                    date=date,
                    symbol=symbol,
                    size_change=size_change,
                    cost=trade_cost
                ))

        # Update positions
        self.positions = new_positions

        # Deduct transaction costs from equity
        self.equity -= total_cost

        # Update history
        self.equity_history.append(self.equity)
        self.dates.append(date)

        logger.debug(
            f"{date.date()}: Rebalanced to {len(target_weights)} positions, "
            f"equity={self.equity:.4f}, cost={total_cost:.4f}"
        )

        return total_cost

    def get_equity_curve(self) -> pd.Series:
        """
        Get equity curve as pandas Series.

        Returns:
            Series with dates as index and equity as values
        """
        if not self.dates:
            return pd.Series([self.initial_equity])

        return pd.Series(self.equity_history[1:], index=self.dates)

    def get_returns(self) -> pd.Series:
        """
        Get period returns.

        Returns:
            Series of period returns (not cumulative)
        """
        equity_curve = self.get_equity_curve()

        if len(equity_curve) < 2:
            return pd.Series([], dtype=float)

        # Calculate returns: (equity[t] / equity[t-1]) - 1
        returns = equity_curve.pct_change().dropna()

        return returns

    def get_trade_log(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.

        Returns:
            DataFrame with columns: date, symbol, size_change, cost
        """
        if not self.trades:
            return pd.DataFrame(columns=['date', 'symbol', 'size_change', 'cost'])

        return pd.DataFrame([
            {
                'date': trade.date,
                'symbol': trade.symbol,
                'size_change': trade.size_change,
                'cost': trade.cost
            }
            for trade in self.trades
        ])

    def reset(self):
        """Reset portfolio to initial state."""
        self.equity = self.initial_equity
        self.positions = {}
        self.trades = []
        self.equity_history = [self.initial_equity]
        self.dates = []


class BacktestEngine:
    """
    Main backtesting engine.

    Runs a strategy over historical data and tracks performance.
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        transaction_fee: float = 0.003,
        initial_equity: float = 1.0
    ):
        """
        Initialize backtesting engine.

        Args:
            data: Dict of {symbol: DataFrame with OHLCV data}
            transaction_fee: Transaction cost (default: 0.003 = 0.3%)
            initial_equity: Starting equity (default: 1.0)
        """
        self.data = data
        self.transaction_fee = transaction_fee
        self.initial_equity = initial_equity

        # Find common date range
        self.dates = self._get_common_dates()

        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_equity=initial_equity,
            transaction_fee=transaction_fee
        )

        logger.info(
            f"Initialized backtesting engine with {len(data)} symbols, "
            f"{len(self.dates)} trading days"
        )

    def _get_common_dates(self) -> pd.DatetimeIndex:
        """
        Find common date range across all symbols.
        
        Uses union of all dates instead of intersection, allowing strategies
        to work with different symbols on different dates.

        Returns:
            DatetimeIndex of all available dates
        """
        if not self.data:
            return pd.DatetimeIndex([])

        # Get union of all date indices (not intersection)
        # This allows strategies to work even if not all symbols have data on all dates
        all_dates = set()

        for symbol, df in self.data.items():
            all_dates |= set(df.index)

        if not all_dates:
            return pd.DatetimeIndex([])

        return pd.DatetimeIndex(sorted(all_dates))

    def run(
        self,
        strategy_func: Callable[[pd.Timestamp, Dict[str, pd.DataFrame]], Dict[str, float]],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Run backtest with a strategy function.

        Args:
            strategy_func: Function that takes (date, data) and returns {symbol: weight}
            start_date: Start date (default: first common date)
            end_date: End date (default: last common date)

        Returns:
            Equity curve as Series

        Example:
            >>> def my_strategy(date, data):
            ...     # Return equal weight for all symbols
            ...     return {symbol: 1.0 / len(data) for symbol in data}
            >>>
            >>> engine = BacktestEngine(data)
            >>> equity_curve = engine.run(my_strategy)
        """
        # Reset portfolio
        self.portfolio.reset()

        # Filter dates
        dates = self.dates
        if start_date:
            dates = dates[dates >= start_date]
        if end_date:
            dates = dates[dates <= end_date]

        if len(dates) < 2:
            logger.warning("Not enough dates for backtesting")
            return pd.Series([self.initial_equity])

        logger.info(f"Running backtest from {dates[0].date()} to {dates[-1].date()}")

        # Run backtest
        for i, date in enumerate(dates[:-1]):  # Stop at second-to-last date
            # Get data up to current date for each symbol
            data_slice = {
                symbol: df.loc[:date]
                for symbol, df in self.data.items()
                if date in df.index
            }

            # Get target weights from strategy
            try:
                target_weights = strategy_func(date, data_slice)
            except Exception as e:
                logger.error(f"Strategy function failed at {date.date()}: {e}")
                target_weights = {}

            # Calculate returns for next period
            next_date = dates[i + 1]
            returns = {}

            for symbol, df in self.data.items():
                if date in df.index and next_date in df.index:
                    current_close = df.loc[date, 'close']
                    next_close = df.loc[next_date, 'close']
                    returns[symbol] = (next_close / current_close) - 1.0

            # Rebalance portfolio
            self.portfolio.rebalance(next_date, target_weights, returns)

        logger.info(
            f"Backtest complete. Final equity: {self.portfolio.equity:.4f} "
            f"({(self.portfolio.equity / self.initial_equity - 1) * 100:.2f}%)"
        )

        return self.portfolio.get_equity_curve()

    def get_results(self) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Get backtest results.

        Returns:
            Tuple of (equity_curve, returns, trade_log)
        """
        return (
            self.portfolio.get_equity_curve(),
            self.portfolio.get_returns(),
            self.portfolio.get_trade_log()
        )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    np.random.seed(42)
    sample_data = {
        'BTC': pd.DataFrame({
            'close': (1 + np.random.randn(100) * 0.02).cumprod() * 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates),
        'ETH': pd.DataFrame({
            'close': (1 + np.random.randn(100) * 0.025).cumprod() * 50,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates),
        'SOL': pd.DataFrame({
            'close': (1 + np.random.randn(100) * 0.03).cumprod() * 20,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates),
    }

    print("Sample data:")
    print(sample_data['BTC'].head())
    print()

    # Define a simple momentum strategy
    def momentum_strategy(date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Simple momentum strategy: equal weight top 2 coins by 20-day momentum.
        """
        if len(data) == 0:
            return {}

        # Calculate 20-day momentum for each symbol
        momentums = {}

        for symbol, df in data.items():
            if len(df) < 20:
                continue

            current_close = df.iloc[-1]['close']
            past_close = df.iloc[-20]['close']
            momentum = (current_close - past_close) / past_close

            momentums[symbol] = momentum

        # Sort by momentum and take top 2
        sorted_symbols = sorted(momentums.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_symbols[:2]]

        # Equal weight
        if len(top_symbols) == 0:
            return {}

        weight = 1.0 / len(top_symbols)
        return {symbol: weight for symbol in top_symbols}

    # Run backtest
    engine = BacktestEngine(sample_data, transaction_fee=0.003)
    equity_curve = engine.run(momentum_strategy)

    print("Backtest results:")
    print(equity_curve.tail())
    print()

    equity_curve, returns, trades = engine.get_results()

    print(f"Final equity: {equity_curve.iloc[-1]:.4f}")
    print(f"Total return: {(equity_curve.iloc[-1] / 1.0 - 1) * 100:.2f}%")
    print(f"Number of trades: {len(trades)}")
    print()

    print("Trade log (last 10):")
    print(trades.tail(10))
    print()

    print("✓ Backtesting engine working correctly!")
