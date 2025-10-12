"""
Strategy Comparison

Compare multiple strategies or parameter sets side-by-side.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from ..backtesting.engine import BacktestEngine
from ..backtesting.metrics import generate_report, PerformanceReport
from ..backtesting.visualization import plot_comparison

logger = logging.getLogger(__name__)


def compare_strategies(
    data: Dict[str, pd.DataFrame],
    strategies: Dict[str, Any],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    transaction_fee: float = 0.003
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Compare multiple strategies.

    Args:
        data: Dict of {symbol: DataFrame}
        strategies: Dict of {strategy_name: strategy_instance}
        start_date: Start date (optional)
        end_date: End date (optional)
        transaction_fee: Transaction cost

    Returns:
        Tuple of (comparison_df, equity_curves_dict)

    Example:
        >>> strategies = {
        ...     'Mom7': MomentumSimpleStrategy(indicator='mom7'),
        ...     'Mom20': MomentumSimpleStrategy(indicator='mom20'),
        ...     'Timing': MarketTimingStrategy()
        ... }
        >>> df, curves = compare_strategies(data, strategies)
    """
    results = {}
    equity_curves = {}

    for name, strategy in strategies.items():
        logger.info(f"Running strategy: {name}")

        try:
            # Create engine
            engine = BacktestEngine(data, transaction_fee=transaction_fee)

            # Run backtest
            equity_curve = engine.run(
                strategy.get_weights,
                start_date=start_date,
                end_date=end_date
            )

            # Get results
            _, returns, _ = engine.get_results()

            # Generate report
            report = generate_report(equity_curve, returns)

            # Store results
            results[name] = {
                'Total Return (%)': report.total_return,
                'CAGR (%)': report.cagr,
                'MDD (%)': report.mdd,
                'Volatility (%)': report.volatility,
                'Sharpe': report.sharpe,
                'Win Rate (%)': report.win_rate,
                'Profit Factor': report.profit_factor,
            }

            equity_curves[name] = equity_curve

            logger.info(
                f"  {name}: Return={report.total_return:.1f}% "
                f"CAGR={report.cagr:.1f}% Sharpe={report.sharpe:.2f}"
            )

        except Exception as e:
            logger.error(f"  Error running {name}: {e}")
            continue

    # Convert to DataFrame
    comparison_df = pd.DataFrame(results).T

    return comparison_df, equity_curves


def generate_comparison_report(
    comparison_df: pd.DataFrame,
    title: str = "Strategy Comparison"
) -> str:
    """
    Generate formatted comparison report.

    Args:
        comparison_df: DataFrame from compare_strategies()
        title: Report title

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 80,
        title.center(80),
        "=" * 80,
        "",
    ]

    # Sort by total return (descending)
    df = comparison_df.sort_values('Total Return (%)', ascending=False)

    # Format table
    for strategy_name in df.index:
        lines.append(f"{strategy_name}:")
        lines.append("-" * 40)

        for metric, value in df.loc[strategy_name].items():
            if isinstance(value, (int, float)):
                if 'Ratio' in metric or metric == 'Sharpe' or 'Factor' in metric:
                    formatted = f"{value:.2f}"
                else:
                    formatted = f"{value:.2f}%"
            else:
                formatted = str(value)

            lines.append(f"  {metric:.<30} {formatted:>10}")

        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def plot_strategy_comparison(
    equity_curves: Dict[str, pd.Series],
    title: str = "Strategy Comparison",
    log_scale: bool = True,
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Plot equity curves for multiple strategies.

    Args:
        equity_curves: Dict of {strategy_name: equity_curve}
        title: Plot title
        log_scale: Use log scale
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    return plot_comparison(
        equity_curves,
        title=title,
        log_scale=log_scale,
        figsize=figsize
    )


def calculate_degradation(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate performance degradation from train to test.

    Args:
        train_metrics: Training metrics
        test_metrics: Test metrics

    Returns:
        Dict of degradation percentages

    Example:
        >>> train = {'cagr': 100.0, 'sharpe': 2.0}
        >>> test = {'cagr': 80.0, 'sharpe': 1.6}
        >>> degradation = calculate_degradation(train, test)
        >>> degradation['cagr']  # -20.0 (20% degradation)
    """
    degradation = {}

    for metric in train_metrics:
        if metric in test_metrics:
            train_val = train_metrics[metric]
            test_val = test_metrics[metric]

            if train_val != 0:
                deg = ((test_val - train_val) / abs(train_val)) * 100
                degradation[metric] = deg

    return degradation


def rank_strategies(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Rank strategies by multiple metrics.

    Args:
        comparison_df: Comparison DataFrame
        metrics: List of metrics to rank by (default: all)

    Returns:
        DataFrame with ranks for each metric

    Example:
        >>> ranks = rank_strategies(comparison_df, metrics=['CAGR (%)', 'Sharpe'])
        >>> ranks['CAGR (%)']  # [1, 2, 3, ...] (1 = best)
    """
    if metrics is None:
        metrics = list(comparison_df.columns)

    ranks = pd.DataFrame(index=comparison_df.index)

    for metric in metrics:
        if metric not in comparison_df.columns:
            continue

        # For MDD, lower is better (rank ascending)
        # For others, higher is better (rank descending)
        ascending = ('MDD' in metric or 'Drawdown' in metric)

        ranks[metric] = comparison_df[metric].rank(ascending=ascending, method='min').astype(int)

    # Calculate average rank
    ranks['Avg Rank'] = ranks.mean(axis=1)

    # Sort by average rank
    ranks = ranks.sort_values('Avg Rank')

    return ranks


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Metrics Comparison example - see notebooks for full demonstration")
    print()
    print("This module provides:")
    print("  - compare_strategies(): Run and compare multiple strategies")
    print("  - generate_comparison_report(): Formatted text report")
    print("  - plot_strategy_comparison(): Visual comparison")
    print("  - calculate_degradation(): Train/test performance degradation")
    print("  - rank_strategies(): Rank strategies by multiple metrics")
    print()
    print("✓ Metrics comparison module ready!")
