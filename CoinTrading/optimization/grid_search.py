"""
Grid Search Optimization

Parameter optimization with proper train/validation/test splits.

This fixes a critical bug in the original notebooks: testing on the same
data used for optimization (overfitting).
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
import pandas as pd
import numpy as np
import logging
from itertools import product
from dataclasses import dataclass, field
import time

from ..backtesting.engine import BacktestEngine
from ..backtesting.metrics import calculate_cagr, calculate_mdd, calculate_sharpe

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """
    Result from a single parameter combination.

    Attributes:
        parameters: Parameter values tested
        train_metrics: Metrics on training set
        val_metrics: Metrics on validation set
        test_metrics: Metrics on test set (if available)
        equity_curve: Final equity curve
        runtime: Time taken to run backtest
    """
    parameters: Dict[str, Any]
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    equity_curve: Optional[pd.Series] = None
    runtime: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for DataFrame."""
        result = {'runtime': self.runtime}

        # Add parameters
        for key, value in self.parameters.items():
            result[f'param_{key}'] = value

        # Add train metrics
        for key, value in self.train_metrics.items():
            result[f'train_{key}'] = value

        # Add validation metrics
        for key, value in self.val_metrics.items():
            result[f'val_{key}'] = value

        # Add test metrics (if available)
        for key, value in self.test_metrics.items():
            result[f'test_{key}'] = value

        return result


class ParameterGrid:
    """
    Generate all combinations of parameters.

    Example:
        >>> grid = ParameterGrid({
        ...     'period': [7, 14, 20],
        ...     'threshold': [0.6, 0.7, 0.8]
        ... })
        >>> list(grid)
        [{'period': 7, 'threshold': 0.6}, ...]
    """

    def __init__(self, param_grid: Dict[str, List[Any]]):
        """
        Initialize parameter grid.

        Args:
            param_grid: Dict of {param_name: [values]}
        """
        self.param_grid = param_grid
        self.param_names = list(param_grid.keys())
        self.param_values = [param_grid[name] for name in self.param_names]

    def __iter__(self):
        """Iterate over all parameter combinations."""
        for values in product(*self.param_values):
            yield dict(zip(self.param_names, values))

    def __len__(self) -> int:
        """Number of combinations."""
        return np.prod([len(values) for values in self.param_values])


class GridSearch:
    """
    Grid search with train/validation/test splits.

    Prevents overfitting by optimizing on train set, selecting best
    parameters on validation set, and reporting final performance on test set.
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        optimization_metric: str = 'sharpe',
        transaction_fee: float = 0.003
    ):
        """
        Initialize grid search.

        Args:
            data: Dict of {symbol: DataFrame}
            strategy_class: Strategy class to test
            param_grid: Parameter grid
            train_ratio: Training set ratio (default: 0.6)
            val_ratio: Validation set ratio (default: 0.2)
            test_ratio: Test set ratio (default: 0.2)
            optimization_metric: Metric to optimize ('cagr', 'sharpe', 'mdd')
            transaction_fee: Transaction cost
        """
        self.data = data
        self.strategy_class = strategy_class
        self.param_grid = ParameterGrid(param_grid)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.optimization_metric = optimization_metric.lower()
        self.transaction_fee = transaction_fee

        # Split dates
        self.train_dates, self.val_dates, self.test_dates = self._split_dates()

        # Results storage
        self.results: List[OptimizationResult] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = -np.inf

        logger.info(
            f"Initialized GridSearch: {len(self.param_grid)} combinations, "
            f"optimize on {optimization_metric}"
        )
        logger.info(f"Train: {self.train_dates[0].date()} to {self.train_dates[-1].date()}")
        logger.info(f"Val:   {self.val_dates[0].date()} to {self.val_dates[-1].date()}")
        logger.info(f"Test:  {self.test_dates[0].date()} to {self.test_dates[-1].date()}")

    def _split_dates(self) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
        """
        Split dates into train/validation/test.

        Returns:
            Tuple of (train_dates, val_dates, test_dates)
        """
        # Get union of all dates (allow different symbols on different dates)
        # This matches BacktestEngine behavior
        all_dates = set()
        for symbol, df in self.data.items():
            all_dates |= set(df.index)

        if not all_dates:
            raise ValueError("No dates found in data")

        dates = pd.DatetimeIndex(sorted(all_dates))

        # Calculate split indices
        n = len(dates)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]

        return train_dates, val_dates, test_dates

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            equity_curve: Equity curve
            returns: Period returns

        Returns:
            Dict of metrics
        """
        if len(equity_curve) < 2:
            return {'cagr': 0.0, 'mdd': 0.0, 'sharpe': 0.0}

        return {
            'cagr': calculate_cagr(equity_curve) * 100,
            'mdd': calculate_mdd(equity_curve) * 100,
            'sharpe': calculate_sharpe(returns),
        }

    def _run_backtest(
        self,
        strategy: Any,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Run backtest for a date range.

        Args:
            strategy: Strategy instance
            start_date: Start date
            end_date: End date

        Returns:
            Tuple of (equity_curve, returns)
        """
        # Create data slice
        data_slice = {
            symbol: df.loc[:end_date]  # Include all history up to end_date
            for symbol, df in self.data.items()
            if not df.empty
        }

        # Create engine
        engine = BacktestEngine(
            data=data_slice,
            transaction_fee=self.transaction_fee
        )

        # Run backtest
        equity_curve = engine.run(
            strategy_func=strategy.get_weights,
            start_date=start_date,
            end_date=end_date
        )

        _, returns, _ = engine.get_results()

        return equity_curve, returns

    def run(self) -> pd.DataFrame:
        """
        Run grid search.

        Returns:
            DataFrame with results for all parameter combinations
        """
        logger.info(f"Starting grid search: {len(self.param_grid)} combinations")

        for i, params in enumerate(self.param_grid, 1):
            start_time = time.time()

            logger.info(f"[{i}/{len(self.param_grid)}] Testing: {params}")

            try:
                # Create strategy with these parameters
                strategy = self.strategy_class(**params)

                # Run on train set
                train_equity, train_returns = self._run_backtest(
                    strategy,
                    self.train_dates[0],
                    self.train_dates[-1]
                )
                train_metrics = self._calculate_metrics(train_equity, train_returns)

                # Run on validation set
                val_equity, val_returns = self._run_backtest(
                    strategy,
                    self.val_dates[0],
                    self.val_dates[-1]
                )
                val_metrics = self._calculate_metrics(val_equity, val_returns)

                # Run on test set
                test_equity, test_returns = self._run_backtest(
                    strategy,
                    self.test_dates[0],
                    self.test_dates[-1]
                )
                test_metrics = self._calculate_metrics(test_equity, test_returns)

                runtime = time.time() - start_time

                # Store result
                result = OptimizationResult(
                    parameters=params,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                    equity_curve=test_equity,  # Store test equity curve
                    runtime=runtime
                )
                self.results.append(result)

                # Check if this is the best on validation set
                val_score = val_metrics.get(self.optimization_metric, -np.inf)
                if val_score > self.best_score:
                    self.best_score = val_score
                    self.best_params = params
                    logger.info(f"  → New best! Val {self.optimization_metric}: {val_score:.2f}")

                logger.info(
                    f"  Train: CAGR={train_metrics['cagr']:.1f}% MDD={train_metrics['mdd']:.1f}% "
                    f"Sharpe={train_metrics['sharpe']:.2f}"
                )
                logger.info(
                    f"  Val:   CAGR={val_metrics['cagr']:.1f}% MDD={val_metrics['mdd']:.1f}% "
                    f"Sharpe={val_metrics['sharpe']:.2f}"
                )
                logger.info(
                    f"  Test:  CAGR={test_metrics['cagr']:.1f}% MDD={test_metrics['mdd']:.1f}% "
                    f"Sharpe={test_metrics['sharpe']:.2f}"
                )

            except Exception as e:
                logger.error(f"  Error: {e}")
                continue

        logger.info(f"\nGrid search complete!")
        logger.info(f"Best parameters (by val {self.optimization_metric}): {self.best_params}")

        # Convert to DataFrame
        return self.get_results_df()

    def get_results_df(self) -> pd.DataFrame:
        """
        Get results as DataFrame.

        Returns:
            DataFrame with all results
        """
        if not self.results:
            return pd.DataFrame()

        rows = [result.to_dict() for result in self.results]
        df = pd.DataFrame(rows)

        # Sort by validation metric (descending for CAGR/Sharpe, ascending for MDD)
        ascending = (self.optimization_metric == 'mdd')
        df = df.sort_values(f'val_{self.optimization_metric}', ascending=ascending)

        return df

    def get_best_strategy(self) -> Any:
        """
        Get strategy with best parameters.

        Returns:
            Strategy instance with best parameters
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run grid search first.")

        return self.strategy_class(**self.best_params)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Grid Search example - see notebooks for full demonstration")
    print()
    print("This module provides:")
    print("  - ParameterGrid: Generate parameter combinations")
    print("  - GridSearch: Optimize with train/val/test splits")
    print("  - OptimizationResult: Store results for each combination")
    print()
    print("✓ Grid search module ready!")
