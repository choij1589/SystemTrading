"""
US All-Weather Asset Allocation Strategy

Implements Ray Dalio's All-Weather Portfolio concept with US ETFs.
Balanced allocation across different economic environments.
"""

from typing import Dict, List
import pandas as pd

from .base import StaticAllocationStrategy


class USAllWeatherStrategy(StaticAllocationStrategy):
    """
    US All-Weather Asset Allocation Strategy

    Based on Ray Dalio's All-Weather Portfolio concept:
    - Diversified across asset classes
    - Balanced for different economic scenarios
    - Risk parity approach

    Asset allocation:
    - US Stocks (SPY): 30%
    - Long-term Bonds (TLT): 40%
    - Intermediate Bonds (IEF): 15%
    - Gold (GLD): 7.5%
    - Commodities (DBC): 7.5%

    This allocation aims to perform well across various economic conditions:
    - Growth + Low Inflation: Stocks benefit
    - Growth + High Inflation: Commodities benefit
    - Recession + Low Inflation: Bonds benefit
    - Recession + High Inflation: Gold benefits
    """

    def __init__(self):
        """Initialize US All-Weather strategy."""
        target_weights = {
            "SPY": 0.30,   # US Large Cap Stocks
            "TLT": 0.40,   # 20+ Year Treasury Bonds
            "IEF": 0.15,   # 7-10 Year Treasury Bonds
            "GLD": 0.075,  # Gold
            "DBC": 0.075   # Commodities
        }

        super().__init__(
            target_weights=target_weights,
            name="US All-Weather"
        )

    def get_universe(self) -> List[str]:
        """Get list of ETFs in this strategy."""
        return list(self.target_weights.keys())

    def describe(self) -> str:
        """Get strategy description."""
        desc = f"""
US All-Weather Asset Allocation Strategy
{'=' * 60}
Type: Static allocation (Risk Parity inspired)

Asset Allocation:
- 30% US Stocks (SPY)
- 40% Long-term Bonds (TLT)
- 15% Intermediate Bonds (IEF)
- 7.5% Gold (GLD)
- 7.5% Commodities (DBC)

Philosophy:
- Based on Ray Dalio's All-Weather Portfolio
- Balanced for all economic environments
- Risk parity approach (equal risk contribution)
- Low correlation between assets
- Defensive positioning with growth potential

Expected Performance:
- Lower volatility than 100% stocks
- Positive returns in most environments
- Protection against inflation
- Downside protection through bonds

Rebalancing:
- Monthly or when weights deviate >5%
        """
        return desc.strip()


if __name__ == "__main__":
    # Example usage
    strategy = USAllWeatherStrategy()

    print(strategy.describe())
    print()
    print(f"Universe: {strategy.get_universe()}")
    print(f"Target weights: {strategy.target_weights}")
