"""
Buy-and-Hold Benchmark Strategies

Simple buy-and-hold strategies for comparison with active strategies.
"""

from typing import Dict, List
import pandas as pd

from .base import StaticAllocationStrategy


class BuyHoldSPYStrategy(StaticAllocationStrategy):
    """
    100% SPY Buy-and-Hold

    Simple S&P 500 index fund strategy.
    No rebalancing, just buy and hold.
    """

    def __init__(self):
        """Initialize 100% SPY strategy."""
        target_weights = {
            "SPY": 1.0,  # 100% S&P 500
        }

        super().__init__(
            target_weights=target_weights,
            name="100% SPY Buy-Hold"
        )

    def get_universe(self) -> List[str]:
        """Get list of ETFs in this strategy."""
        return list(self.target_weights.keys())

    def describe(self) -> str:
        """Get strategy description."""
        return "100% S&P 500 (SPY) Buy-and-Hold Benchmark"


class BuyHoldQQQStrategy(StaticAllocationStrategy):
    """
    100% QQQ Buy-and-Hold

    Simple Nasdaq 100 index fund strategy.
    High growth, tech-heavy exposure.
    """

    def __init__(self):
        """Initialize 100% QQQ strategy."""
        target_weights = {
            "QQQ": 1.0,  # 100% Nasdaq 100
        }

        super().__init__(
            target_weights=target_weights,
            name="100% QQQ Buy-Hold"
        )

    def get_universe(self) -> List[str]:
        """Get list of ETFs in this strategy."""
        return list(self.target_weights.keys())

    def describe(self) -> str:
        """Get strategy description."""
        return "100% Nasdaq 100 (QQQ) Buy-and-Hold Benchmark"


class BuyHoldTLTStrategy(StaticAllocationStrategy):
    """
    100% TLT Buy-and-Hold

    Simple long-term treasury bond strategy.
    Safe haven asset.
    """

    def __init__(self):
        """Initialize 100% TLT strategy."""
        target_weights = {
            "TLT": 1.0,  # 100% Long-term Bonds
        }

        super().__init__(
            target_weights=target_weights,
            name="100% TLT Buy-Hold"
        )

    def get_universe(self) -> List[str]:
        """Get list of ETFs in this strategy."""
        return list(self.target_weights.keys())

    def describe(self) -> str:
        """Get strategy description."""
        return "100% Long-term Bonds (TLT) Buy-and-Hold Benchmark"


class Classic6040Strategy(StaticAllocationStrategy):
    """
    Classic 60/40 Portfolio

    60% Stocks / 40% Bonds
    Traditional balanced portfolio.
    """

    def __init__(self):
        """Initialize 60/40 strategy."""
        target_weights = {
            "SPY": 0.60,  # 60% Stocks
            "BND": 0.40,  # 40% Bonds
        }

        super().__init__(
            target_weights=target_weights,
            name="60/40 Stock/Bond"
        )

    def get_universe(self) -> List[str]:
        """Get list of ETFs in this strategy."""
        return list(self.target_weights.keys())

    def describe(self) -> str:
        """Get strategy description."""
        return "Classic 60/40 Stock/Bond Portfolio"
