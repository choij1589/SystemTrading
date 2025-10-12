"""
Walk-Forward Analysis

Time-series validation to prevent look-ahead bias.

Trains on a window, tests on the next period, then moves forward.
More realistic than single train/test split for time-series data.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field

from ..backtesting.engine import BacktestEngine
from ..backtesting.metrics import calculate_cagr, calculate_mdd, calculate_sharpe

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardPeriod:
    """
    Result from a single walk-forward period.

    Attributes:
        period_num: Period number
        train_start: Training start date
        train_end: Training end date
        test_start: Test start date
        test_end: Test end date
        best_params: Best parameters from optimization
        train_metrics: Metrics on training set
        test_metrics: Metrics on test set (out-of-sample)
        equity_curve: Test period equity curve
    """
    period_num: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: Dict[str, Any] = field(default_factory=dict)
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    equity_curve: Optional[pd.Series] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'period': self.period_num,
            'train_start': self.train_start.date(),
            'train_end': self.train_end.date(),
            'test_start': self.test_start.date(),
            'test_end': self.test_end.date(),
        }

        # Add parameters
        for key, value in self.best_params.items():
            result[f'param_{key}'] = value

        # Add metrics
        for key, value in self.train_metrics.items():
            result[f'train_{key}'] = value

        for key, value in self.test_metrics.items():
            result[f'test_{key}'] = value

        return result


class WalkForwardAnalysis:
    """
    Walk-forward analysis with expanding or rolling window.

    Example:
        Period 1: Train [0:90],   Test [90:120]
        Period 2: Train [0:120],  Test [120:150]  (expanding)
        Period 2: Train [30:120], Test [120:150]  (rolling)
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        train_window_days: int = 90,
        test_window_days: int = 30,
        step_days: int = 30,
        window_type: str = 'expanding',
        optimization_metric: str = 'sharpe',
        transaction_fee: float = 0.003,
        min_train_days: int = 200
    ):
        """
        Initialize walk-forward analysis.

        Args:
            data: Dict of {symbol: DataFrame}
            strategy_class: Strategy class to test
            param_grid: Parameter grid for optimization
            train_window_days: Training window size
            test_window_days: Test window size
            step_days: Step size between periods
            window_type: 'expanding' or 'rolling'
            optimization_metric: Metric to optimize
            transaction_fee: Transaction cost
            min_train_days: Minimum training days required
        """
        self.data = data
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.window_type = window_type
        self.optimization_metric = optimization_metric.lower()
        self.transaction_fee = transaction_fee
        self.min_train_days = min_train_days

        # Get common dates
        self.dates = self._get_common_dates()

        # Results storage
        self.periods: List[WalkForwardPeriod] = []

        logger.info(
            f"Initialized Walk-Forward Analysis: {window_type} window, "
            f"train={train_window_days}d, test={test_window_days}d, step={step_days}d"
        )
        logger.info(f"Date range: {self.dates[0].date()} to {self.dates[-1].date()}")

    def _get_common_dates(self) -> pd.DatetimeIndex:
        """Get union of all dates (allow different symbols on different dates)."""
        all_dates = set()
        for symbol, df in self.data.items():
            all_dates |= set(df.index)

        if not all_dates:
            raise ValueError("No dates found in data")

        return pd.DatetimeIndex(sorted(all_dates))

    def _generate_periods(self) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """
        Generate train/test period pairs.

        Returns:
            List of (train_dates, test_dates) tuples
        """
        periods = []
        current_test_start_idx = self.min_train_days

        while current_test_start_idx < len(self.dates):
            # Test period
            test_end_idx = min(
                current_test_start_idx + self.test_window_days,
                len(self.dates)
            )

            if test_end_idx <= current_test_start_idx:
                break

            test_dates = self.dates[current_test_start_idx:test_end_idx]

            # Train period
            if self.window_type == 'expanding':
                # Expanding: use all data from start
                train_start_idx = 0
            else:  # rolling
                # Rolling: fixed-size window
                train_start_idx = max(0, current_test_start_idx - self.train_window_days)

            train_dates = self.dates[train_start_idx:current_test_start_idx]

            # Ensure minimum training size
            if len(train_dates) >= self.min_train_days:
                periods.append((train_dates, test_dates))

            # Move forward
            current_test_start_idx += self.step_days

        return periods

    def _optimize_on_period(
        self,
        train_dates: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """
        Find best parameters on training period.

        Args:
            train_dates: Training dates

        Returns:
            Best parameters
        """
        from .grid_search import ParameterGrid

        param_grid = ParameterGrid(self.param_grid)
        best_params = None
        best_score = -np.inf

        for params in param_grid:
            try:
                # Create strategy
                strategy = self.strategy_class(**params)

                # Backtest on train period
                data_slice = {
                    symbol: df.loc[:train_dates[-1]]
                    for symbol, df in self.data.items()
                }

                engine = BacktestEngine(data_slice, transaction_fee=self.transaction_fee)
                equity_curve = engine.run(
                    strategy.get_weights,
                    start_date=train_dates[0],
                    end_date=train_dates[-1]
                )

                _, returns, _ = engine.get_results()

                # Calculate metric
                if self.optimization_metric == 'cagr':
                    score = calculate_cagr(equity_curve)
                elif self.optimization_metric == 'sharpe':
                    score = calculate_sharpe(returns)
                elif self.optimization_metric == 'mdd':
                    score = -calculate_mdd(equity_curve)  # Negative because we minimize MDD
                else:
                    score = calculate_sharpe(returns)

                # Track best
                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.debug(f"Error optimizing {params}: {e}")
                continue

        return best_params or {}

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        if len(equity_curve) < 2:
            return {'cagr': 0.0, 'mdd': 0.0, 'sharpe': 0.0}

        return {
            'cagr': calculate_cagr(equity_curve) * 100,
            'mdd': calculate_mdd(equity_curve) * 100,
            'sharpe': calculate_sharpe(returns),
        }

    def run(self) -> pd.DataFrame:
        """
        Run walk-forward analysis.

        Returns:
            DataFrame with results for each period
        """
        periods = self._generate_periods()
        logger.info(f"Generated {len(periods)} walk-forward periods")

        for i, (train_dates, test_dates) in enumerate(periods, 1):
            logger.info(f"\n[Period {i}/{len(periods)}]")
            logger.info(f"  Train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} days)")
            logger.info(f"  Test:  {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")

            # Optimize on training period
            logger.info("  Optimizing parameters...")
            best_params = self._optimize_on_period(train_dates)
            logger.info(f"  Best params: {best_params}")

            if not best_params:
                logger.warning("  No valid parameters found, skipping period")
                continue

            # Test on training period (in-sample)
            try:
                strategy = self.strategy_class(**best_params)

                data_slice = {
                    symbol: df.loc[:train_dates[-1]]
                    for symbol, df in self.data.items()
                }

                engine = BacktestEngine(data_slice, transaction_fee=self.transaction_fee)
                train_equity = engine.run(
                    strategy.get_weights,
                    start_date=train_dates[0],
                    end_date=train_dates[-1]
                )
                _, train_returns, _ = engine.get_results()
                train_metrics = self._calculate_metrics(train_equity, train_returns)

                logger.info(
                    f"  Train: CAGR={train_metrics['cagr']:.1f}% "
                    f"MDD={train_metrics['mdd']:.1f}% Sharpe={train_metrics['sharpe']:.2f}"
                )

            except Exception as e:
                logger.error(f"  Error on train: {e}")
                train_metrics = {}

            # Test on test period (out-of-sample)
            try:
                strategy = self.strategy_class(**best_params)

                data_slice = {
                    symbol: df.loc[:test_dates[-1]]
                    for symbol, df in self.data.items()
                }

                engine = BacktestEngine(data_slice, transaction_fee=self.transaction_fee)
                test_equity = engine.run(
                    strategy.get_weights,
                    start_date=test_dates[0],
                    end_date=test_dates[-1]
                )
                _, test_returns, _ = engine.get_results()
                test_metrics = self._calculate_metrics(test_equity, test_returns)

                logger.info(
                    f"  Test:  CAGR={test_metrics['cagr']:.1f}% "
                    f"MDD={test_metrics['mdd']:.1f}% Sharpe={test_metrics['sharpe']:.2f}"
                )

            except Exception as e:
                logger.error(f"  Error on test: {e}")
                test_metrics = {}
                test_equity = None

            # Store period result
            period = WalkForwardPeriod(
                period_num=i,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                best_params=best_params,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                equity_curve=test_equity
            )
            self.periods.append(period)

        logger.info(f"\nWalk-forward analysis complete! {len(self.periods)} periods tested")

        return self.get_results_df()

    def get_results_df(self) -> pd.DataFrame:
        """
        Get results as DataFrame.

        Returns:
            DataFrame with all period results
        """
        if not self.periods:
            return pd.DataFrame()

        rows = [period.to_dict() for period in self.periods]
        return pd.DataFrame(rows)

    def get_combined_equity_curve(self) -> pd.Series:
        """
        Combine test period equity curves into single curve.

        Returns:
            Combined equity curve (out-of-sample only)
        """
        if not self.periods:
            return pd.Series([], dtype=float)

        # Concatenate all test equity curves
        curves = []
        for period in self.periods:
            if period.equity_curve is not None and len(period.equity_curve) > 0:
                curves.append(period.equity_curve)

        if not curves:
            return pd.Series([], dtype=float)

        # Chain them together
        combined = pd.concat(curves)
        return combined


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Walk-Forward Analysis example - see notebooks for full demonstration")
    print()
    print("This module provides:")
    print("  - WalkForwardAnalysis: Time-series validation")
    print("  - WalkForwardPeriod: Results for each period")
    print("  - Expanding/Rolling window support")
    print()
    print("✓ Walk-forward module ready!")
