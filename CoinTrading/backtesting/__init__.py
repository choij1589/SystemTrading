"""
Backtesting Framework

Provides tools for backtesting trading strategies with realistic transaction costs.

Usage:
    from CoinTrading.backtesting import BacktestEngine, generate_report, plot_equity_curve

Example:
    # Initialize engine
    engine = BacktestEngine(data, transaction_fee=0.003)

    # Define strategy
    def my_strategy(date, data):
        # Return {symbol: weight} dict
        return {'BTC': 0.5, 'ETH': 0.5}

    # Run backtest
    equity_curve = engine.run(my_strategy)

    # Get results
    equity_curve, returns, trades = engine.get_results()

    # Generate report
    report = generate_report(equity_curve, returns)
    print(report)

    # Visualize
    plot_equity_curve(equity_curve)
"""

from .engine import BacktestEngine, Portfolio, Position, Trade
from .metrics import (
    calculate_cagr,
    calculate_mdd,
    calculate_sharpe,
    calculate_volatility,
    calculate_win_rate,
    calculate_profit_factor,
    generate_report,
    PerformanceReport
)
from .visualization import (
    plot_equity_curve,
    plot_drawdown,
    plot_monthly_returns,
    plot_comparison,
    plot_returns_distribution,
    plot_rolling_metrics
)

__all__ = [
    # Engine
    'BacktestEngine',
    'Portfolio',
    'Position',
    'Trade',

    # Metrics
    'calculate_cagr',
    'calculate_mdd',
    'calculate_sharpe',
    'calculate_volatility',
    'calculate_win_rate',
    'calculate_profit_factor',
    'generate_report',
    'PerformanceReport',

    # Visualization
    'plot_equity_curve',
    'plot_drawdown',
    'plot_monthly_returns',
    'plot_comparison',
    'plot_returns_distribution',
    'plot_rolling_metrics',
]
