"""
Yahoo Finance Data Loader

Alternative data source for backtesting when KIS API is not available.
Uses yfinance library to fetch Korean ETF data.
"""

from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Install with: pip install yfinance")


class YahooETFLoader:
    """
    ETF data loader using Yahoo Finance.

    Korean ETFs are listed on Yahoo with .KS suffix (e.g., 069500.KS)
    """

    # Mapping from Korean ticker to Yahoo ticker
    TICKER_MAP = {
        "069500": "069500.KS",  # KODEX 200
        "102110": "102110.KS",  # TIGER 200
        "360750": "360750.KS",  # TIGER 미국S&P500
        "133690": "133690.KS",  # TIGER 미국나스닥100
        "360200": "360200.KS",  # TIGER 미국테크TOP10
        "458730": "458730.KS",  # TIGER 미국배당다우존스
        "152380": "152380.KS",  # KODEX 국고채3년
        "114260": "114260.KS",  # KODEX 국고채10년
        "132030": "132030.KS",  # KODEX 골드선물
        "091180": "091180.KS",  # KODEX 반도체
        "157450": "157450.KS",  # TIGER 2차전지테마
        "227540": "227540.KS",  # TIGER 200 IT
        "139230": "139230.KS",  # TIGER 200 건설
        "139260": "139260.KS",  # TIGER 200 에너지화학
        "139250": "139250.KS",  # TIGER 200 금융
        "228790": "228790.KS",  # TIGER 200 헬스케어
    }

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize Yahoo Finance loader.

        Args:
            cache_dir: Directory for cached data
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _korean_to_yahoo_ticker(self, ticker: str) -> str:
        """Convert Korean ticker to Yahoo ticker."""
        return self.TICKER_MAP.get(ticker, f"{ticker}.KS")

    def load_single(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load data for a single ETF.

        Args:
            ticker: Korean ETF ticker (e.g., "069500")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date (None = today)

        Returns:
            DataFrame with OHLCV data
        """
        yahoo_ticker = self._korean_to_yahoo_ticker(ticker)

        # Check cache
        cache_file = self.cache_dir / f"{ticker}_daily.parquet"
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                # Filter by date
                df = df[(df['date'] >= pd.to_datetime(start_date))]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
                if not df.empty:
                    return df
            except:
                pass

        # Fetch from Yahoo Finance
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        try:
            yf_ticker = yf.Ticker(yahoo_ticker)
            df = yf_ticker.history(start=start_date, end=end_date)

            if df.empty:
                print(f"No data available for {ticker} ({yahoo_ticker})")
                return pd.DataFrame()

            # Rename and restructure
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Select columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

            # Remove timezone info
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

            # Save to cache
            if use_cache:
                df.to_parquet(cache_file)

            return df

        except Exception as e:
            print(f"Error fetching {ticker} from Yahoo Finance: {e}")
            return pd.DataFrame()

    def load_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple ETFs.

        Args:
            tickers: List of Korean ETF tickers
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date (None = today)

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        result = {}

        for ticker in tickers:
            df = self.load_single(ticker, start_date, end_date, use_cache)
            if not df.empty:
                result[ticker] = df

        return result


def generate_sample_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    initial_price: float = 10000.0,
    volatility: float = 0.02
) -> Dict[str, pd.DataFrame]:
    """
    Generate sample price data for testing.

    Args:
        tickers: List of ticker codes
        start_date: Start date
        end_date: End date
        initial_price: Starting price
        volatility: Daily volatility

    Returns:
        Dictionary mapping ticker to sample DataFrame
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    result = {}

    for ticker in tickers:
        # Generate random returns
        np.random.seed(hash(ticker) % (2**32))
        returns = np.random.normal(0.0003, volatility, len(dates))

        # Generate price series
        prices = initial_price * np.cumprod(1 + returns)

        # Create OHLCV data
        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates))
        })

        result[ticker] = df

    return result


if __name__ == "__main__":
    print("Yahoo Finance ETF Loader")
    print("=" * 50)

    if YFINANCE_AVAILABLE:
        loader = YahooETFLoader()

        # Test loading
        df = loader.load_single(
            ticker="069500",  # KODEX 200
            start_date="2024-01-01"
        )

        print(f"\nLoaded {len(df)} rows for KODEX 200")
        print(df.head())
    else:
        print("yfinance not available - using sample data")
        data = generate_sample_data(
            tickers=["069500", "360750"],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        print(f"Generated sample data for {len(data)} tickers")
