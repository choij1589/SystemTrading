"""
Optimization & Validation Module

Tools for parameter optimization and strategy validation.

Usage:
    from CoinTrading.optimization import GridSearch, WalkForwardAnalysis, compare_strategies

Example - Grid Search:
    >>> param_grid = {
    ...     'indicator': ['mom7', 'mom14', 'mom20'],
    ...     'long_top_n': [3, 4, 5],
    ...     'short_bottom_n': [5, 8, 10]
    ... }
    >>>
    >>> grid_search = GridSearch(
    ...     data=data,
    ...     strategy_class=MomentumSimpleStrategy,
    ...     param_grid=param_grid,
    ...     optimization_metric='sharpe'
    ... )
    >>>
    >>> results_df = grid_search.run()
    >>> best_strategy = grid_search.get_best_strategy()

Example - Walk-Forward:
    >>> wfa = WalkForwardAnalysis(
    ...     data=data,
    ...     strategy_class=MomentumSimpleStrategy,
    ...     param_grid=param_grid,
    ...     train_window_days=90,
    ...     test_window_days=30
    ... )
    >>>
    >>> results_df = wfa.run()
    >>> equity_curve = wfa.get_combined_equity_curve()

Example - Strategy Comparison:
    >>> strategies = {
    ...     'Mom7': MomentumSimpleStrategy(indicator='mom7'),
    ...     'Mom20': MomentumSimpleStrategy(indicator='mom20'),
    ...     'Timing': MarketTimingStrategy()
    ... }
    >>>
    >>> comparison_df, equity_curves = compare_strategies(data, strategies)
    >>> print(generate_comparison_report(comparison_df))
    >>> plot_strategy_comparison(equity_curves)
"""

from .grid_search import (
    GridSearch,
    ParameterGrid,
    OptimizationResult
)
from .walk_forward import (
    WalkForwardAnalysis,
    WalkForwardPeriod
)
from .metrics_comparison import (
    compare_strategies,
    generate_comparison_report,
    plot_strategy_comparison,
    calculate_degradation,
    rank_strategies
)

__all__ = [
    # Grid Search
    'GridSearch',
    'ParameterGrid',
    'OptimizationResult',

    # Walk-Forward
    'WalkForwardAnalysis',
    'WalkForwardPeriod',

    # Comparison
    'compare_strategies',
    'generate_comparison_report',
    'plot_strategy_comparison',
    'calculate_degradation',
    'rank_strategies',
]
