"""
Dividend + Growth Mix Strategy

Balanced strategy combining dividend income and growth potential.
Simple 50/50 allocation between dividend and growth ETFs.
"""

from typing import Dict, List
import pandas as pd

from .base import StaticAllocationStrategy


class DividendGrowthMixStrategy(StaticAllocationStrategy):
    """
    Dividend + Growth Mix Strategy

    Asset allocation:
    - US Dividend ETF (TIGER 미국배당다우존스): 50%
    - US Growth ETF (TIGER 나스닥100): 50%

    This strategy combines:
    - Dividend income from established companies
    - Growth potential from tech stocks
    - Simple 50/50 balance for risk management
    """

    def __init__(self):
        """Initialize dividend + growth mix strategy."""
        target_weights = {
            "458730": 0.50,  # TIGER 미국배당다우존스 (Dividend)
            "133690": 0.50   # TIGER 미국나스닥100 (Growth)
        }

        super().__init__(
            target_weights=target_weights,
            name="DividendGrowthMix"
        )

    def get_universe(self) -> List[str]:
        """Get list of ETFs in this strategy."""
        return list(self.target_weights.keys())

    def describe(self) -> str:
        """Get strategy description."""
        desc = f"""
Dividend + Growth Mix Strategy
{'=' * 50}
Type: Static allocation (Buy & Hold with rebalancing)

Asset Allocation:
- 50% Dividend Income (TIGER 미국배당다우존스)
- 50% Growth (TIGER 미국나스닥100)

Philosophy:
- Balance between income and growth
- Dividend aristocrats for stability
- Tech growth for capital appreciation
- Lower volatility than 100% growth
- Higher returns than 100% dividend

ETF Details:
1. TIGER 미국배당다우존스
   - Dow Jones U.S. Dividend 100 Index
   - High dividend yield (~3-4%)
   - Quality dividend payers

2. TIGER 미국나스닥100
   - NASDAQ-100 Index
   - Large-cap tech growth
   - Higher volatility, higher potential returns

Rebalancing:
- Monthly or when weights deviate >5%
- Maintain 50/50 balance

Risk Profile:
- Moderate risk/return
- Diversification across US market
- Currency risk (KRW/USD)
        """
        return desc.strip()


if __name__ == "__main__":
    # Example usage
    strategy = DividendGrowthMixStrategy()

    print(strategy.describe())
    print()
    print(f"Universe: {strategy.get_universe()}")
    print(f"Target weights: {strategy.target_weights}")
