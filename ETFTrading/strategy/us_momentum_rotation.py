"""
US Momentum Sector Rotation Strategy

Dynamically rotates between US sector ETFs based on momentum.
Captures sector rotation trends in the US market.
"""

from typing import Dict, List
import pandas as pd
import numpy as np

from .base import DynamicAllocationStrategy


class USMomentumSectorRotationStrategy(DynamicAllocationStrategy):
    """
    US Momentum-based Sector Rotation Strategy

    Strategy logic:
    1. Calculate 3-month and 6-month momentum for all US sector ETFs
    2. Rank sectors by combined momentum score
    3. Invest equally in top N sectors (default: 3)
    4. Require positive momentum to enter
    5. Rebalance monthly

    This captures US sector rotation and trend-following effects.
    """

    def __init__(
        self,
        top_n: int = 3,
        momentum_3m_weight: float = 0.5,
        momentum_6m_weight: float = 0.5,
        min_momentum: float = 0.0
    ):
        """
        Initialize US momentum sector rotation strategy.

        Args:
            top_n: Number of top sectors to hold
            momentum_3m_weight: Weight for 3-month momentum
            momentum_6m_weight: Weight for 6-month momentum
            min_momentum: Minimum momentum to enter position
        """
        # Universe: S&P 500 sector ETFs
        universe = [
            "XLK",   # Technology
            "XLF",   # Financial
            "XLE",   # Energy
            "XLV",   # Healthcare
            "XLI",   # Industrial
            "XLY",   # Consumer Discretionary
            "XLP",   # Consumer Staples
            "XLU",   # Utilities
            "XLRE",  # Real Estate
        ]

        super().__init__(universe=universe, name="US Momentum Sector Rotation")

        self.top_n = top_n
        self.momentum_3m_weight = momentum_3m_weight
        self.momentum_6m_weight = momentum_6m_weight
        self.min_momentum = min_momentum

    def get_weights(
        self,
        data: Dict[str, pd.DataFrame],
        date: pd.Timestamp,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate target weights based on momentum.

        Args:
            data: Price data for all ETFs
            date: Current date for weight calculation

        Returns:
            Target weights dictionary
        """
        momentum_scores = {}

        for ticker in self.universe:
            if ticker not in data:
                continue

            df = data[ticker]
            df_filtered = df[df['date'] <= date].copy()

            if len(df_filtered) < 130:  # Need ~6 months of data
                continue

            try:
                # Calculate 3-month momentum (~60 trading days)
                mom_3m = self.calculate_momentum(df_filtered, lookback_days=60)

                # Calculate 6-month momentum (~120 trading days)
                mom_6m = self.calculate_momentum(df_filtered, lookback_days=120)

                # Combined momentum score
                score = (
                    self.momentum_3m_weight * mom_3m +
                    self.momentum_6m_weight * mom_6m
                )

                momentum_scores[ticker] = score

            except Exception as e:
                print(f"Error calculating momentum for {ticker}: {e}")
                continue

        # Rank by momentum
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)

        # Select top N with positive momentum
        selected = [
            ticker for ticker, score in ranked[:self.top_n]
            if score >= self.min_momentum
        ]

        if not selected:
            # No positive momentum - hold cash
            return {}

        # Equal weight among selected
        weight_per_etf = 1.0 / len(selected)
        weights = {ticker: weight_per_etf for ticker in selected}

        return self.validate_weights(weights)

    def get_universe(self) -> List[str]:
        """Get list of ETFs in this strategy."""
        return self.universe

    def describe(self) -> str:
        """Get strategy description."""
        desc = f"""
US Momentum Sector Rotation Strategy
{'=' * 60}
Type: Dynamic allocation (Trend following)

Strategy Logic:
1. Calculate 3-month and 6-month momentum for all US sectors
2. Rank sectors by combined momentum score
3. Invest equally in top {self.top_n} sectors
4. Only invest if momentum > {self.min_momentum:.1%}
5. Rebalance monthly

Sector Universe (S&P 500 Sectors):
- Technology (XLK)
- Financial (XLF)
- Energy (XLE)
- Healthcare (XLV)
- Industrial (XLI)
- Consumer Discretionary (XLY)
- Consumer Staples (XLP)
- Utilities (XLU)
- Real Estate (XLRE)

Philosophy:
- Capture US sector rotation trends
- Momentum-based allocation
- Diversify across top performers
- Monthly rebalancing
- Stay out during negative momentum

Risk:
- Higher turnover than static allocation
- Sector concentration risk
- Momentum reversal risk
- May underperform in choppy markets

Expected Performance:
- Outperform in trending markets
- Higher volatility than balanced portfolios
- Potential for significant drawdowns
        """
        return desc.strip()


if __name__ == "__main__":
    # Example usage
    strategy = USMomentumSectorRotationStrategy(top_n=3)

    print(strategy.describe())
    print()
    print(f"Universe: {strategy.get_universe()}")
    print(f"Top N: {strategy.top_n}")
