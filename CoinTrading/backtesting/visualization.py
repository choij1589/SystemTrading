"""
Visualization Utilities

Plotting functions for backtesting results.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import logging

logger = logging.getLogger(__name__)


def plot_equity_curve(
    equity_curve: pd.Series,
    title: str = "Equity Curve",
    log_scale: bool = True,
    figsize: Tuple[int, int] = (15, 6),
    benchmark: Optional[pd.Series] = None
) -> Figure:
    """
    Plot equity curve over time.

    Args:
        equity_curve: Series of cumulative equity
        title: Plot title
        log_scale: Use log scale for y-axis (default: True)
        figsize: Figure size
        benchmark: Optional benchmark series to compare against

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve.values, label='Strategy', linewidth=2)

    # Plot benchmark if provided
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, label='Benchmark',
                linewidth=2, alpha=0.7, linestyle='--')

    # Formatting
    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    plt.tight_layout()

    return fig


def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown",
    figsize: Tuple[int, int] = (15, 6)
) -> Figure:
    """
    Plot drawdown over time.

    Args:
        equity_curve: Series of cumulative equity
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max

    # Plot drawdown
    ax.fill_between(drawdown.index, drawdown.values, 0,
                     alpha=0.3, color='red', label='Drawdown')
    ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)

    # Mark maximum drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    ax.scatter([max_dd_idx], [max_dd_val], color='red', s=100, zorder=5,
               label=f'Max DD: {max_dd_val:.2%}')

    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    plt.tight_layout()

    return fig


def plot_monthly_returns(
    returns: pd.Series,
    title: str = "Monthly Returns Heatmap",
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Plot monthly returns as a heatmap.

    Args:
        returns: Series of period returns
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    # Resample to monthly returns
    monthly = (1 + returns).resample('M').prod() - 1

    # Create pivot table (years as rows, months as columns)
    monthly_df = pd.DataFrame({
        'Year': monthly.index.year,
        'Month': monthly.index.month,
        'Return': monthly.values
    })

    pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Use a diverging colormap centered at 0
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    vmin = -vmax

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                   vmin=vmin, vmax=vmax)

    # Set ticks and labels
    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot)):
        for j in range(12):
            value = pivot.iloc[i, j]
            if not pd.isna(value):
                text = ax.text(j, i, f'{value:.1%}',
                             ha="center", va="center", color="black", fontsize=8)

    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Return', rotation=270, labelpad=15)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()

    return fig


def plot_comparison(
    equity_curves: Dict[str, pd.Series],
    title: str = "Strategy Comparison",
    log_scale: bool = True,
    figsize: Tuple[int, int] = (15, 8)
) -> Figure:
    """
    Plot multiple equity curves for comparison.

    Args:
        equity_curves: Dict of {strategy_name: equity_series}
        title: Plot title
        log_scale: Use log scale for y-axis
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each equity curve
    for name, curve in equity_curves.items():
        ax.plot(curve.index, curve.values, label=name, linewidth=2)

    # Formatting
    if log_scale:
        ax.set_yscale('log')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    plt.tight_layout()

    return fig


def plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    bins: int = 50,
    figsize: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Plot returns distribution histogram.

    Args:
        returns: Series of period returns
        title: Plot title
        bins: Number of histogram bins
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(returns.values, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(returns.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {returns.mean():.2%}')
    ax1.axvline(returns.median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {returns.median():.2%}')

    ax1.set_xlabel('Return', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Histogram', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    # Q-Q plot (normal distribution)
    from scipy import stats
    stats.probplot(returns.values, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 60,
    title: str = "Rolling Performance Metrics",
    figsize: Tuple[int, int] = (15, 10)
) -> Figure:
    """
    Plot rolling performance metrics.

    Args:
        returns: Series of period returns
        window: Rolling window size (days)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Calculate rolling metrics
    rolling_mean = returns.rolling(window).mean() * 365  # Annualized
    rolling_vol = returns.rolling(window).std() * np.sqrt(365)  # Annualized
    rolling_sharpe = rolling_mean / rolling_vol

    # Plot rolling return
    axes[0].plot(rolling_mean.index, rolling_mean.values, color='blue', linewidth=2)
    axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('Annualized Return', fontsize=10)
    axes[0].set_title(f'{window}-Day Rolling Return', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Plot rolling volatility
    axes[1].plot(rolling_vol.index, rolling_vol.values, color='red', linewidth=2)
    axes[1].set_ylabel('Annualized Volatility', fontsize=10)
    axes[1].set_title(f'{window}-Day Rolling Volatility', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Plot rolling Sharpe
    axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, color='green', linewidth=2)
    axes[2].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('Sharpe Ratio', fontsize=10)
    axes[2].set_title(f'{window}-Day Rolling Sharpe', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].grid(True, alpha=0.3)

    # Format x-axis dates
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    np.random.seed(42)

    # Simulate returns
    daily_returns = pd.Series(
        np.random.normal(0.001, 0.02, 365),
        index=dates
    )
    equity_curve = (1 + daily_returns).cumprod()

    # Simulate benchmark
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.015, 365),
        index=dates
    )
    benchmark_curve = (1 + benchmark_returns).cumprod()

    print("Creating sample plots...")
    print()

    # Plot equity curve
    fig1 = plot_equity_curve(equity_curve, benchmark=benchmark_curve)
    print("✓ Equity curve plotted")

    # Plot drawdown
    fig2 = plot_drawdown(equity_curve)
    print("✓ Drawdown plotted")

    # Plot monthly returns heatmap
    fig3 = plot_monthly_returns(daily_returns)
    print("✓ Monthly returns heatmap plotted")

    # Plot comparison
    fig4 = plot_comparison({
        'Strategy': equity_curve,
        'Benchmark': benchmark_curve
    })
    print("✓ Comparison plot created")

    # Plot returns distribution
    fig5 = plot_returns_distribution(daily_returns)
    print("✓ Returns distribution plotted")

    # Plot rolling metrics
    fig6 = plot_rolling_metrics(daily_returns, window=60)
    print("✓ Rolling metrics plotted")

    print()
    print("All visualizations created successfully!")
    print("Note: Plots are not displayed in headless mode. Use plt.show() to display.")
