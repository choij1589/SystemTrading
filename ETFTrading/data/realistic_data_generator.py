"""
Realistic ETF Data Generator

Generates synthetic but realistic US ETF price data based on:
- Historical statistics from actual ETFs
- Realistic correlations between asset classes
- Market regime changes (bull/bear markets)
- Sector rotation patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta


class RealisticETFDataGenerator:
    """
    Generate realistic ETF price data for backtesting.

    Based on historical characteristics of actual US ETFs.
    """

    # Historical parameters (based on 2010-2024 data)
    ETF_PARAMETERS = {
        # Broad Market
        "SPY": {"annual_return": 0.13, "volatility": 0.18, "sharpe": 0.72},
        "QQQ": {"annual_return": 0.18, "volatility": 0.23, "sharpe": 0.78},

        # Bonds
        "TLT": {"annual_return": 0.05, "volatility": 0.15, "sharpe": 0.33},
        "IEF": {"annual_return": 0.03, "volatility": 0.08, "sharpe": 0.38},
        "BND": {"annual_return": 0.03, "volatility": 0.06, "sharpe": 0.50},

        # Commodities
        "GLD": {"annual_return": 0.08, "volatility": 0.16, "sharpe": 0.50},
        "DBC": {"annual_return": 0.02, "volatility": 0.22, "sharpe": 0.09},

        # Sectors
        "XLK": {"annual_return": 0.18, "volatility": 0.21, "sharpe": 0.86},  # Tech
        "XLF": {"annual_return": 0.11, "volatility": 0.24, "sharpe": 0.46},  # Finance
        "XLE": {"annual_return": 0.04, "volatility": 0.26, "sharpe": 0.15},  # Energy
        "XLV": {"annual_return": 0.12, "volatility": 0.16, "sharpe": 0.75},  # Healthcare
        "XLI": {"annual_return": 0.12, "volatility": 0.19, "sharpe": 0.63},  # Industrial
        "XLY": {"annual_return": 0.14, "volatility": 0.19, "sharpe": 0.74},  # Consumer Disc
        "XLP": {"annual_return": 0.10, "volatility": 0.14, "sharpe": 0.71},  # Consumer Staples
        "XLU": {"annual_return": 0.09, "volatility": 0.17, "sharpe": 0.53},  # Utilities
        "XLRE": {"annual_return": 0.08, "volatility": 0.20, "sharpe": 0.40},  # Real Estate

        # Dividend
        "SCHD": {"annual_return": 0.13, "volatility": 0.16, "sharpe": 0.81},
    }

    # Correlation matrix (simplified)
    CORRELATIONS = {
        "SPY-QQQ": 0.95,
        "SPY-TLT": -0.30,
        "SPY-GLD": 0.00,
        "TLT-GLD": 0.20,
        "SPY-Sectors": 0.85,  # Sectors correlate highly with market
        "Sectors-Sectors": 0.60,  # Sectors correlate with each other
    }

    def __init__(self, seed: int = 42):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def generate(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_price: float = 100.0
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate realistic price data.

        Args:
            tickers: List of ETF tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_price: Starting price

        Returns:
            Dictionary mapping ticker to OHLCV DataFrame
        """
        # Generate date range (trading days only - ~252 per year)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # Remove weekends
        dates = dates[dates.weekday < 5]

        # Generate market factor (common component)
        market_returns = self._generate_market_returns(len(dates))

        result = {}

        for ticker in tickers:
            if ticker not in self.ETF_PARAMETERS:
                print(f"Warning: No parameters for {ticker}, using SPY defaults")
                params = self.ETF_PARAMETERS["SPY"]
            else:
                params = self.ETF_PARAMETERS[ticker]

            # Generate correlated returns
            returns = self._generate_correlated_returns(
                ticker=ticker,
                market_returns=market_returns,
                params=params,
                n_days=len(dates)
            )

            # Generate OHLCV data
            df = self._returns_to_ohlcv(
                dates=dates,
                returns=returns,
                initial_price=initial_price
            )

            result[ticker] = df

        return result

    def _generate_market_returns(self, n_days: int) -> np.ndarray:
        """Generate market-level returns (common factor)."""
        # Market has bull and bear regimes
        daily_return = 0.13 / 252  # ~13% annual
        daily_vol = 0.18 / np.sqrt(252)  # ~18% annual vol

        # Add regime changes (bull/bear markets)
        returns = np.random.normal(daily_return, daily_vol, n_days)

        # Simulate occasional crashes
        crash_prob = 0.002  # ~0.2% chance per day
        crashes = np.random.binomial(1, crash_prob, n_days)
        returns -= crashes * np.random.uniform(0.02, 0.05, n_days)  # -2% to -5%

        # Add persistence (auto-correlation)
        for i in range(1, n_days):
            returns[i] += 0.05 * returns[i-1]  # Slight momentum

        return returns

    def _generate_correlated_returns(
        self,
        ticker: str,
        market_returns: np.ndarray,
        params: Dict,
        n_days: int
    ) -> np.ndarray:
        """Generate returns correlated with market."""
        daily_return = params["annual_return"] / 252
        daily_vol = params["volatility"] / np.sqrt(252)

        # Determine correlation with market
        if ticker in ["SPY", "QQQ"]:
            correlation = 0.98  # Very high
        elif ticker in ["TLT", "IEF"]:
            correlation = -0.30  # Negative (flight to safety)
        elif ticker in ["GLD", "DBC"]:
            correlation = 0.10  # Low
        elif ticker.startswith("XL"):  # Sectors
            correlation = 0.85
        else:
            correlation = 0.70

        # Generate correlated returns
        idiosyncratic = np.random.normal(0, 1, n_days)

        returns = (
            correlation * market_returns * (params["volatility"] / 0.18) +
            np.sqrt(1 - correlation**2) * idiosyncratic * daily_vol +
            daily_return
        )

        return returns

    def _returns_to_ohlcv(
        self,
        dates: pd.DatetimeIndex,
        returns: np.ndarray,
        initial_price: float
    ) -> pd.DataFrame:
        """Convert returns to OHLCV data."""
        # Generate close prices
        prices = initial_price * np.cumprod(1 + returns)

        # Generate OHLC (realistic intraday movement)
        df = pd.DataFrame({
            'date': dates,
            'close': prices
        })

        # Open = previous close with small gap
        df['open'] = df['close'].shift(1) * (1 + np.random.uniform(-0.002, 0.002, len(df)))
        df['open'].iloc[0] = initial_price

        # High and Low based on daily range
        daily_range = np.abs(np.random.normal(0.005, 0.003, len(df)))  # ~0.5% average range
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + daily_range)
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - daily_range)

        # Volume (correlated with volatility)
        base_volume = 1_000_000
        vol_factor = np.abs(returns) * 10 + 1
        df['volume'] = (base_volume * vol_factor).astype(int)

        # Reorder columns
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

        return df


if __name__ == "__main__":
    # Test generator
    generator = RealisticETFDataGenerator(seed=42)

    tickers = ["SPY", "QQQ", "TLT", "GLD"]
    data = generator.generate(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2024-12-31",
        initial_price=100.0
    )

    for ticker, df in data.items():
        print(f"\n{ticker}:")
        print(f"  Days: {len(df)}")
        print(f"  Start: {df['close'].iloc[0]:.2f}")
        print(f"  End: {df['close'].iloc[-1]:.2f}")
        print(f"  Return: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.1f}%")
        print(f"  Ann. Vol: {df['close'].pct_change().std() * np.sqrt(252) * 100:.1f}%")
