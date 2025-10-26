"""
Global Asset Allocation Strategy

Implements a static asset allocation strategy similar to All Weather Portfolio.
Provides diversification across asset classes with fixed target weights.
"""

from typing import Dict, List
import pandas as pd

from .base import StaticAllocationStrategy


class GlobalAssetAllocationStrategy(StaticAllocationStrategy):
    """
    Global Asset Allocation Strategy (All Weather Style)

    Asset allocation:
    - Domestic Equity (KODEX 200): 30%
    - US Equity (TIGER S&P500): 40%
    - Bonds (KODEX 국고채3년): 20%
    - Commodities (KODEX 골드선물): 10%

    This is a classic 60/40 variant with gold for inflation protection.
    Rebalanced periodically to maintain target weights.
    """

    def __init__(self):
        """Initialize global asset allocation strategy."""
        target_weights = {
            "069500": 0.30,  # KODEX 200 (Korean equity)
            "360750": 0.40,  # TIGER 미국S&P500 (US equity)
            "152380": 0.20,  # KODEX 국고채3년 (Short-term bonds)
            "132030": 0.10   # KODEX 골드선물(H) (Gold)
        }

        super().__init__(
            target_weights=target_weights,
            name="GlobalAssetAllocation"
        )

    def get_universe(self) -> List[str]:
        """Get list of ETFs in this strategy."""
        return list(self.target_weights.keys())

    def describe(self) -> str:
        """Get strategy description."""
        desc = f"""
Global Asset Allocation Strategy
{'=' * 50}
Type: Static allocation (Buy & Hold with rebalancing)

Asset Allocation:
- 30% Domestic Equity (KODEX 200)
- 40% US Equity (TIGER S&P500)
- 20% Bonds (KODEX 국고채3년)
- 10% Gold (KODEX 골드선물)

Philosophy:
- Diversification across asset classes
- Risk parity approach
- All-weather protection
- Low turnover

Rebalancing:
- Monthly or when weights deviate >5%
        """
        return desc.strip()


if __name__ == "__main__":
    # Example usage
    strategy = GlobalAssetAllocationStrategy()

    print(strategy.describe())
    print()
    print(f"Universe: {strategy.get_universe()}")
    print(f"Target weights: {strategy.target_weights}")
