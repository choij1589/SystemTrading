"""
Momentum-based Sector Rotation Strategy

Dynamically rotates between sector ETFs based on recent momentum.
Selects top performing sectors and rebalances monthly.
"""

from typing import Dict, List
import pandas as pd
import numpy as np

from .base import DynamicAllocationStrategy


class MomentumSectorRotationStrategy(DynamicAllocationStrategy):
    """
    Momentum-based Sector Rotation Strategy

    Strategy logic:
    1. Calculate 3-month and 6-month momentum for all sector ETFs
    2. Rank sectors by average momentum
    3. Invest equally in top N sectors (default: 3)
    4. Rebalance monthly

    This captures sector rotation and trend-following effects.
    """

    def __init__(
        self,
        top_n: int = 3,
        momentum_3m_weight: float = 0.5,
        momentum_6m_weight: float = 0.5,
        min_momentum: float = 0.0
    ):
        """
        Initialize momentum sector rotation strategy.

        Args:
            top_n: Number of top sectors to hold
            momentum_3m_weight: Weight for 3-month momentum
            momentum_6m_weight: Weight for 6-month momentum
            min_momentum: Minimum momentum to enter position
        """
        # Universe: various sector ETFs
        universe = [
            "091180",  # KODEX 반도체
            "157450",  # TIGER 2차전지테마
            "227540",  # TIGER 200 IT
            "139230",  # TIGER 200 건설
            "139260",  # TIGER 200 에너지화학
            "139250",  # TIGER 200 금융
            "228790",  # TIGER 200 헬스케어
            "360750",  # TIGER 미국S&P500 (fallback)
        ]

        super().__init__(universe=universe, name="MomentumSectorRotation")

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
Momentum Sector Rotation Strategy
{'=' * 50}
Type: Dynamic allocation (Trend following)

Strategy Logic:
1. Calculate 3-month and 6-month momentum for all sectors
2. Rank sectors by combined momentum score
3. Invest equally in top {self.top_n} sectors
4. Only invest if momentum > {self.min_momentum:.1%}
5. Rebalance monthly

Sector Universe:
- Semiconductors (KODEX 반도체)
- Battery (TIGER 2차전지테마)
- IT (TIGER 200 IT)
- Construction (TIGER 200 건설)
- Energy/Chemical (TIGER 200 에너지화학)
- Finance (TIGER 200 금융)
- Healthcare (TIGER 200 헬스케어)
- US Equity (TIGER S&P500)

Philosophy:
- Capture sector rotation trends
- Momentum-based allocation
- Diversify across top performers
- Monthly rebalancing

Risk:
- Higher turnover than static allocation
- Sector concentration risk
- Momentum reversal risk
        """
        return desc.strip()


if __name__ == "__main__":
    # Example usage
    strategy = MomentumSectorRotationStrategy(top_n=3)

    print(strategy.describe())
    print()
    print(f"Universe: {strategy.get_universe()}")
    print(f"Top N: {strategy.top_n}")
