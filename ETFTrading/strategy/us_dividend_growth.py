"""
US Dividend + Growth Strategy

Combines dividend income with growth potential using US ETFs.
Balanced approach for income and capital appreciation.
"""

from typing import Dict, List
import pandas as pd

from .base import StaticAllocationStrategy


class USDividendGrowthStrategy(StaticAllocationStrategy):
    """
    US Dividend + Growth Strategy

    Asset allocation:
    - Dividend Growth (SCHD): 50%
    - Tech Growth (QQQ): 50%

    This strategy combines:
    - Dividend income from quality dividend growers (SCHD)
    - Capital appreciation from tech leaders (QQQ)

    Benefits:
    - Regular dividend income
    - Growth potential from tech sector
    - Balanced risk/return profile
    - Lower volatility than 100% growth
    """

    def __init__(self):
        """Initialize US Dividend + Growth strategy."""
        target_weights = {
            "SCHD": 0.50,  # Schwab US Dividend Equity ETF
            "QQQ": 0.50    # Invesco QQQ Trust (Nasdaq 100)
        }

        super().__init__(
            target_weights=target_weights,
            name="US Dividend + Growth"
        )

    def get_universe(self) -> List[str]:
        """Get list of ETFs in this strategy."""
        return list(self.target_weights.keys())

    def describe(self) -> str:
        """Get strategy description."""
        desc = f"""
US Dividend + Growth Strategy
{'=' * 60}
Type: Static allocation (Income + Growth)

Asset Allocation:
- 50% Dividend Growth (SCHD)
- 50% Tech Growth (QQQ)

Philosophy:
- Balanced income and growth
- Quality dividend payers + tech leaders
- Lower volatility than pure growth
- Regular dividend income
- Long-term wealth building

SCHD (Schwab US Dividend Equity):
- High-quality dividend growers
- Low expense ratio (0.06%)
- Focus on dividend sustainability
- Lower volatility

QQQ (Invesco QQQ Trust):
- Nasdaq 100 tracking
- Tech-heavy portfolio
- Growth-oriented
- Higher volatility

Expected Performance:
- Moderate growth with income
- Lower volatility than 100% QQQ
- Dividend yield: ~2-3%
- Suitable for balanced investors

Rebalancing:
- Monthly or when weights deviate >5%
        """
        return desc.strip()


if __name__ == "__main__":
    # Example usage
    strategy = USDividendGrowthStrategy()

    print(strategy.describe())
    print()
    print(f"Universe: {strategy.get_universe()}")
    print(f"Target weights: {strategy.target_weights}")
