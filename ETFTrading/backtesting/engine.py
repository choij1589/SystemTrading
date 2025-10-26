"""
ETF Backtesting Engine

Simulates ETF portfolio with:
- Regular monthly deposits
- Periodic rebalancing
- Transaction costs (commission + tax)
- Integer share constraints
"""

import os
import yaml
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..strategy.base import BaseETFStrategy


class ETFBacktestEngine:
    """
    Backtesting engine for ETF investment strategies.

    Features:
    - Monthly deposits (Dollar Cost Averaging)
    - Periodic rebalancing (weekly or monthly)
    - Transaction costs (commission + tax)
    - Integer share constraints
    - Cash management
    """

    def __init__(
        self,
        initial_capital: float = 4_200_000,
        monthly_deposit: float = 300_000,
        commission_rate: float = 0.00015,
        tax_rate: float = 0.0023,
        slippage: float = 0.0001,
        rebalance_frequency: str = "monthly",
        deposit_day: int = 1,
        config_path: Optional[str] = None
    ):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Initial investment (KRW)
            monthly_deposit: Monthly deposit amount (KRW)
            commission_rate: Commission rate (0.015%)
            tax_rate: Tax rate for sell orders (0.23%)
            slippage: Slippage rate (0.01%)
            rebalance_frequency: "weekly" or "monthly"
            deposit_day: Day of month for deposits (1-28)
            config_path: Path to config.yaml (optional)
        """
        # Load from config if provided
        if config_path:
            self._load_config(config_path)
        else:
            self.initial_capital = initial_capital
            self.monthly_deposit = monthly_deposit
            self.commission_rate = commission_rate
            self.tax_rate = tax_rate
            self.slippage = slippage
            self.rebalance_frequency = rebalance_frequency
            self.deposit_day = deposit_day

        # Portfolio state
        self.cash = self.initial_capital
        self.holdings = {}  # {ticker: num_shares}
        self.equity_curve = []
        self.trade_log = []

    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        inv = config['investment']
        self.initial_capital = inv['initial_capital']
        self.monthly_deposit = inv['monthly_deposit']
        self.deposit_day = inv['deposit_day']

        txn = config['transaction']
        self.commission_rate = txn['commission_rate']
        self.tax_rate = txn['tax_rate']
        self.slippage = txn['slippage']

        reb = config['rebalancing']
        self.rebalance_frequency = reb['frequency']

    def _is_rebalance_day(self, date: pd.Timestamp, prev_date: pd.Timestamp) -> bool:
        """Check if this date is a rebalancing day."""
        if self.rebalance_frequency == "monthly":
            # Rebalance on first trading day of month
            return date.month != prev_date.month
        elif self.rebalance_frequency == "weekly":
            # Rebalance on Mondays (weekday=0)
            return date.weekday() == 0 and date != prev_date
        else:
            return False

    def _is_deposit_day(self, date: pd.Timestamp, prev_date: pd.Timestamp) -> bool:
        """Check if this date is a deposit day."""
        # Deposit on first trading day of month
        return date.month != prev_date.month

    def _get_portfolio_value(
        self,
        date: pd.Timestamp,
        prices: Dict[str, float]
    ) -> float:
        """
        Calculate current portfolio value.

        Args:
            date: Current date
            prices: Current prices {ticker: price}

        Returns:
            Total portfolio value (cash + holdings)
        """
        holdings_value = 0.0

        for ticker, shares in self.holdings.items():
            if ticker in prices:
                holdings_value += shares * prices[ticker]

        return self.cash + holdings_value

    def _get_current_weights(
        self,
        prices: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate current portfolio weights.

        Args:
            prices: Current prices {ticker: price}

        Returns:
            Current weights {ticker: weight}
        """
        total_value = self._get_portfolio_value(None, prices)

        if total_value == 0:
            return {}

        weights = {}
        for ticker, shares in self.holdings.items():
            if ticker in prices:
                value = shares * prices[ticker]
                weights[ticker] = value / total_value

        return weights

    def _rebalance(
        self,
        date: pd.Timestamp,
        target_weights: Dict[str, float],
        prices: Dict[str, float]
    ):
        """
        Rebalance portfolio to target weights.

        Args:
            date: Current date
            target_weights: Target weights {ticker: weight}
            prices: Current prices {ticker: price}
        """
        total_value = self._get_portfolio_value(date, prices)

        if total_value <= 0:
            return

        # Calculate target shares for each position
        target_shares = {}
        for ticker, weight in target_weights.items():
            if ticker not in prices:
                continue

            target_value = total_value * weight
            shares = int(target_value / prices[ticker])  # Integer shares only
            target_shares[ticker] = shares

        # Current holdings
        current_shares = self.holdings.copy()

        # Determine sells (do sells first to free up cash)
        sells = []
        for ticker, current in current_shares.items():
            target = target_shares.get(ticker, 0)
            if target < current:
                sells.append((ticker, current - target))

        # Execute sells
        for ticker, shares_to_sell in sells:
            self._execute_trade(
                date=date,
                ticker=ticker,
                shares=shares_to_sell,
                price=prices[ticker],
                side="sell"
            )

        # Determine buys
        buys = []
        for ticker, target in target_shares.items():
            current = self.holdings.get(ticker, 0)
            if target > current:
                buys.append((ticker, target - current))

        # Execute buys
        for ticker, shares_to_buy in buys:
            self._execute_trade(
                date=date,
                ticker=ticker,
                shares=shares_to_buy,
                price=prices[ticker],
                side="buy"
            )

    def _execute_trade(
        self,
        date: pd.Timestamp,
        ticker: str,
        shares: int,
        price: float,
        side: str
    ):
        """
        Execute a single trade with costs.

        Args:
            date: Trade date
            ticker: ETF ticker
            shares: Number of shares
            price: Trade price
            side: "buy" or "sell"
        """
        if shares <= 0:
            return

        # Apply slippage
        if side == "buy":
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)

        # Calculate trade value
        trade_value = shares * execution_price

        # Calculate costs
        commission = trade_value * self.commission_rate

        if side == "buy":
            total_cost = trade_value + commission

            # Check if we have enough cash
            if total_cost > self.cash:
                # Adjust shares to fit available cash
                available_after_commission = self.cash / (1 + self.commission_rate)
                shares = int(available_after_commission / execution_price)

                if shares <= 0:
                    return

                trade_value = shares * execution_price
                commission = trade_value * self.commission_rate
                total_cost = trade_value + commission

            # Execute buy
            self.cash -= total_cost
            self.holdings[ticker] = self.holdings.get(ticker, 0) + shares

            # Log trade
            self.trade_log.append({
                'date': date,
                'ticker': ticker,
                'side': 'buy',
                'shares': shares,
                'price': execution_price,
                'value': trade_value,
                'commission': commission,
                'tax': 0,
                'total_cost': total_cost
            })

        else:  # sell
            tax = trade_value * self.tax_rate
            total_proceeds = trade_value - commission - tax

            # Execute sell
            self.cash += total_proceeds
            self.holdings[ticker] = self.holdings.get(ticker, 0) - shares

            # Remove if zero
            if self.holdings[ticker] <= 0:
                del self.holdings[ticker]

            # Log trade
            self.trade_log.append({
                'date': date,
                'ticker': ticker,
                'side': 'sell',
                'shares': shares,
                'price': execution_price,
                'value': trade_value,
                'commission': commission,
                'tax': tax,
                'total_proceeds': total_proceeds
            })

    def run(
        self,
        strategy: BaseETFStrategy,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run backtest for a strategy.

        Args:
            strategy: Trading strategy
            data: Price data {ticker: DataFrame}
            start_date: Start date (None = earliest available)
            end_date: End date (None = latest available)

        Returns:
            DataFrame with equity curve
        """
        # Reset state
        self.cash = self.initial_capital
        self.holdings = {}
        self.equity_curve = []
        self.trade_log = []

        # Get all dates
        all_dates = set()
        for df in data.values():
            all_dates.update(df['date'].values)

        dates = sorted(all_dates)
        dates = pd.to_datetime(dates)

        # Filter by date range
        if start_date:
            dates = [d for d in dates if d >= pd.to_datetime(start_date)]
        if end_date:
            dates = [d for d in dates if d <= pd.to_datetime(end_date)]

        if len(dates) < 2:
            raise ValueError("Need at least 2 dates for backtesting")

        # Run simulation
        prev_date = dates[0]

        for i, date in enumerate(dates):
            # Get current prices
            prices = {}
            for ticker, df in data.items():
                ticker_data = df[df['date'] == date]
                if not ticker_data.empty:
                    prices[ticker] = ticker_data['close'].values[0]

            if not prices:
                continue

            # Monthly deposit
            if i > 0 and self._is_deposit_day(date, prev_date):
                self.cash += self.monthly_deposit

            # Rebalancing
            if i == 0 or self._is_rebalance_day(date, prev_date):
                target_weights = strategy.get_weights(data, date)
                self._rebalance(date, target_weights, prices)

            # Record equity curve
            portfolio_value = self._get_portfolio_value(date, prices)
            self.equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'holdings_value': portfolio_value - self.cash
            })

            prev_date = date

        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        return equity_df

    def get_trade_log(self) -> pd.DataFrame:
        """
        Get trade log as DataFrame.

        Returns:
            DataFrame with all trades
        """
        if not self.trade_log:
            return pd.DataFrame()

        return pd.DataFrame(self.trade_log)

    def get_summary_stats(self, equity_df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics.

        Args:
            equity_df: Equity curve DataFrame

        Returns:
            Dictionary with performance metrics
        """
        if equity_df.empty:
            return {}

        # Calculate returns
        equity_df = equity_df.copy()
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()

        # Total return
        initial_value = equity_df['portfolio_value'].iloc[0]
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value

        # CAGR
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = days / 365.25
        cagr = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else 0

        # Volatility (annualized)
        daily_returns = equity_df['returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0

        # Maximum drawdown
        cummax = equity_df['portfolio_value'].cummax()
        drawdown = (equity_df['portfolio_value'] - cummax) / cummax
        max_drawdown = drawdown.min()

        # Win rate
        winning_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_rate = winning_days / total_days if total_days > 0 else 0

        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_days': total_days
        }


if __name__ == "__main__":
    print("ETF Backtesting Engine")
    print("=" * 50)
    print("This engine simulates ETF portfolio with:")
    print("- Monthly deposits (Dollar Cost Averaging)")
    print("- Periodic rebalancing")
    print("- Transaction costs (commission + tax)")
    print("- Integer share constraints")
