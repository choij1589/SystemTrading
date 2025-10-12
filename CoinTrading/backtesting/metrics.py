"""
Performance Metrics

Calculates backtesting performance metrics (CAGR, MDD, Sharpe, etc.)
All calculations are vectorized using pandas.
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def calculate_cagr(
    equity_curve: pd.Series,
    periods_per_year: int = 365
) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Formula: CAGR = (final_value / initial_value)^(periods_per_year / n_periods) - 1

    Args:
        equity_curve: Series of cumulative returns (equity)
        periods_per_year: Number of periods in a year (365 for daily)

    Returns:
        CAGR as decimal (e.g., 0.25 = 25% annual return)
    """
    if len(equity_curve) < 2:
        return 0.0

    final_value = equity_curve.iloc[-1]
    initial_value = equity_curve.iloc[0]
    n_periods = len(equity_curve)

    if initial_value <= 0 or final_value <= 0:
        return 0.0

    cagr = (final_value / initial_value) ** (periods_per_year / n_periods) - 1.0
    return float(cagr)


def calculate_mdd(equity_curve: pd.Series) -> float:
    """
    Calculate Maximum Drawdown.

    Formula: MDD = min((equity - running_max) / running_max)

    Args:
        equity_curve: Series of cumulative returns (equity)

    Returns:
        Maximum drawdown as decimal (e.g., -0.50 = -50% drawdown)
    """
    if len(equity_curve) < 2:
        return 0.0

    # Calculate running maximum
    running_max = equity_curve.cummax()

    # Calculate drawdown at each point
    drawdown = (equity_curve - running_max) / running_max

    # Maximum drawdown is the minimum value
    mdd = float(drawdown.min())

    return mdd


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = 365
) -> float:
    """
    Calculate annualized volatility.

    Formula: VOL = std(returns) * sqrt(periods_per_year)

    Args:
        returns: Series of period returns (NOT cumulative)
        periods_per_year: Number of periods in a year

    Returns:
        Annualized volatility as decimal
    """
    if len(returns) < 2:
        return 0.0

    vol = float(returns.std() * np.sqrt(periods_per_year))
    return vol


def calculate_sharpe(
    returns: pd.Series,
    periods_per_year: int = 365,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe ratio.

    Formula: Sharpe = (mean(returns) - rf) / std(returns) * sqrt(periods_per_year)

    Args:
        returns: Series of period returns (NOT cumulative)
        periods_per_year: Number of periods in a year
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    std = returns.std()
    if std == 0:
        return 0.0

    # Convert annualized risk-free rate to period rate
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    sharpe = (returns.mean() - rf_period) / std * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (percentage of winning periods).

    Args:
        returns: Series of period returns

    Returns:
        Win rate as decimal (e.g., 0.55 = 55% win rate)
    """
    if len(returns) == 0:
        return 0.0

    # Count winning periods (returns > 0)
    wins = (returns > 0).sum()
    total = len(returns)

    return float(wins / total)


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor (sum of gains / sum of losses).

    Args:
        returns: Series of period returns

    Returns:
        Profit factor (values > 1 indicate profitability)
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float('inf') if gains > 0 else 0.0

    return float(gains / losses)


@dataclass
class PerformanceReport:
    """
    Container for all performance metrics.

    Attributes:
        total_return: Total return (%)
        cagr: Compound annual growth rate (%)
        mdd: Maximum drawdown (%)
        volatility: Annualized volatility (%)
        sharpe: Sharpe ratio
        win_rate: Win rate (%)
        profit_factor: Profit factor
        n_periods: Number of periods
        start_date: Start date
        end_date: End date
    """
    total_return: float
    cagr: float
    mdd: float
    volatility: float
    sharpe: float
    win_rate: float
    profit_factor: float
    n_periods: int
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'Total Return (%)': f"{self.total_return:.2f}",
            'CAGR (%)': f"{self.cagr:.2f}",
            'Max Drawdown (%)': f"{self.mdd:.2f}",
            'Volatility (%)': f"{self.volatility:.2f}",
            'Sharpe Ratio': f"{self.sharpe:.2f}",
            'Win Rate (%)': f"{self.win_rate:.2f}",
            'Profit Factor': f"{self.profit_factor:.2f}",
            'Periods': self.n_periods,
            'Start': str(self.start_date.date()) if self.start_date else 'N/A',
            'End': str(self.end_date.date()) if self.end_date else 'N/A',
        }

    def __str__(self) -> str:
        """Pretty print the report."""
        lines = [
            "=" * 50,
            "PERFORMANCE REPORT",
            "=" * 50,
        ]

        for key, value in self.to_dict().items():
            lines.append(f"{key:.<30} {value:>18}")

        lines.append("=" * 50)
        return "\n".join(lines)


def generate_report(
    equity_curve: pd.Series,
    returns: pd.Series,
    periods_per_year: int = 365
) -> PerformanceReport:
    """
    Generate comprehensive performance report.

    Args:
        equity_curve: Series of cumulative equity
        returns: Series of period returns
        periods_per_year: Number of periods per year

    Returns:
        PerformanceReport with all metrics
    """
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    cagr = calculate_cagr(equity_curve, periods_per_year) * 100
    mdd = calculate_mdd(equity_curve) * 100
    volatility = calculate_volatility(returns, periods_per_year) * 100
    sharpe = calculate_sharpe(returns, periods_per_year)
    win_rate = calculate_win_rate(returns) * 100
    profit_factor = calculate_profit_factor(returns)

    return PerformanceReport(
        total_return=total_return,
        cagr=cagr,
        mdd=mdd,
        volatility=volatility,
        sharpe=sharpe,
        win_rate=win_rate,
        profit_factor=profit_factor,
        n_periods=len(returns),
        start_date=equity_curve.index[0] if hasattr(equity_curve.index[0], 'date') else None,
        end_date=equity_curve.index[-1] if hasattr(equity_curve.index[-1], 'date') else None,
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample equity curve
    dates = pd.date_range('2024-01-01', periods=365, freq='D')

    # Simulate a strategy with 50% total return and some drawdowns
    np.random.seed(42)
    daily_returns = pd.Series(
        np.random.normal(0.001, 0.02, 365),  # Mean 0.1% daily, 2% volatility
        index=dates
    )

    equity_curve = (1 + daily_returns).cumprod()

    print("Sample equity curve:")
    print(equity_curve.tail())
    print()

    # Calculate individual metrics
    print(f"CAGR: {calculate_cagr(equity_curve) * 100:.2f}%")
    print(f"MDD: {calculate_mdd(equity_curve) * 100:.2f}%")
    print(f"Volatility: {calculate_volatility(daily_returns) * 100:.2f}%")
    print(f"Sharpe: {calculate_sharpe(daily_returns):.2f}")
    print(f"Win Rate: {calculate_win_rate(daily_returns) * 100:.2f}%")
    print(f"Profit Factor: {calculate_profit_factor(daily_returns):.2f}")
    print()

    # Generate comprehensive report
    report = generate_report(equity_curve, daily_returns)
    print(report)

    # Test edge cases
    print("\n" + "=" * 50)
    print("Testing edge cases:")
    print("=" * 50)

    # Empty series
    empty_series = pd.Series([], dtype=float)
    print(f"Empty series CAGR: {calculate_cagr(empty_series)}")

    # Single value
    single_value = pd.Series([1.0])
    print(f"Single value MDD: {calculate_mdd(single_value)}")

    # Zero volatility
    constant = pd.Series([0.0] * 100)
    print(f"Zero volatility Sharpe: {calculate_sharpe(constant)}")

    print("\n✓ All metrics calculated successfully!")
