"""
Korean ETF Data Loader - Multiple Source Support

This module provides a unified interface for loading Korean ETF data
from various sources (pykrx, FinanceDataReader, KIS API, or cached data).
"""

import os
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import warnings

class KoreanETFLoader:
    """
    Unified Korean ETF Data Loader

    Supports multiple data sources:
    1. pykrx (KRX public data - FREE, no auth required)
    2. FinanceDataReader (Multiple sources - FREE)
    3. KIS API (Korea Investment Securities - requires API key)
    4. Cached data (parquet files)
    """

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize loader

        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Check available data sources
        self.available_sources = self._check_sources()

    def _check_sources(self) -> Dict[str, bool]:
        """Check which data sources are available"""
        sources = {}

        # Check pykrx
        try:
            import pykrx
            sources['pykrx'] = True
        except ImportError:
            sources['pykrx'] = False

        # Check FinanceDataReader
        try:
            import FinanceDataReader
            sources['fdr'] = True
        except ImportError:
            sources['fdr'] = False

        # Check KIS API
        kis_secrets = os.path.join(
            os.path.dirname(__file__),
            "../config/secrets.yaml"
        )
        sources['kis'] = os.path.exists(kis_secrets)

        return sources

    def get_etf_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        source: str = "auto",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a Korean ETF

        Args:
            ticker: ETF code (e.g., "069500")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Data source - "auto", "pykrx", "fdr", "kis", or "cache"
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(ticker, start_date, end_date)
            if cached is not None:
                return cached

        # Determine source
        if source == "auto":
            source = self._select_best_source()

        # Fetch data
        df = None

        if source == "pykrx":
            df = self._fetch_pykrx(ticker, start_date, end_date)
        elif source == "fdr":
            df = self._fetch_fdr(ticker, start_date, end_date)
        elif source == "kis":
            df = self._fetch_kis(ticker, start_date, end_date)
        else:
            raise ValueError(f"Unknown source: {source}")

        # Cache if successful
        if df is not None and not df.empty and use_cache:
            self._save_to_cache(ticker, df)

        return df if df is not None else pd.DataFrame()

    def load_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        source: str = "auto",
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple ETFs

        Args:
            tickers: List of ETF codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Data source
            use_cache: Whether to use cache

        Returns:
            Dictionary {ticker: DataFrame}
        """
        results = {}

        for ticker in tickers:
            try:
                df = self.get_etf_ohlcv(
                    ticker, start_date, end_date,
                    source=source, use_cache=use_cache
                )
                if not df.empty:
                    results[ticker] = df
            except Exception as e:
                warnings.warn(f"Failed to load {ticker}: {e}")

        return results

    def _select_best_source(self) -> str:
        """Select the best available data source"""
        if self.available_sources.get('pykrx'):
            return 'pykrx'
        elif self.available_sources.get('fdr'):
            return 'fdr'
        elif self.available_sources.get('kis'):
            return 'kis'
        else:
            raise RuntimeError(
                "No data source available. Please install pykrx or "
                "FinanceDataReader, or configure KIS API credentials."
            )

    def _fetch_pykrx(
        self, ticker: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data using pykrx"""
        try:
            from pykrx import stock

            # Convert date format
            start = start_date.replace("-", "")
            end = end_date.replace("-", "")

            df = stock.get_etf_ohlcv_by_date(
                fromdate=start,
                todate=end,
                ticker=ticker
            )

            if df.empty:
                return None

            # Standardize column names
            df = df.reset_index()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

            return df

        except Exception as e:
            warnings.warn(f"pykrx failed for {ticker}: {e}")
            return None

    def _fetch_fdr(
        self, ticker: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data using FinanceDataReader"""
        try:
            import FinanceDataReader as fdr

            df = fdr.DataReader(ticker, start_date, end_date)

            if df.empty:
                return None

            # Standardize column names
            df = df.reset_index()
            df.columns = df.columns.str.lower()

            # Rename if needed
            column_mapping = {
                'index': 'date',
                'adj close': 'close'
            }
            df = df.rename(columns=column_mapping)

            # Select standard columns
            cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in cols if c in df.columns]]

            return df

        except Exception as e:
            warnings.warn(f"FinanceDataReader failed for {ticker}: {e}")
            return None

    def _fetch_kis(
        self, ticker: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch data using KIS API"""
        try:
            from .kis_client import KISClient

            client = KISClient(mode="mock")

            # Convert date format
            start = start_date.replace("-", "")
            end = end_date.replace("-", "")

            df = client.get_daily_price(ticker, start, end)

            return df if not df.empty else None

        except Exception as e:
            warnings.warn(f"KIS API failed for {ticker}: {e}")
            return None

    def _load_from_cache(
        self, ticker: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        cache_file = os.path.join(self.cache_dir, f"{ticker}.parquet")

        if not os.path.exists(cache_file):
            return None

        try:
            df = pd.read_parquet(cache_file)
            df['date'] = pd.to_datetime(df['date'])

            # Filter by date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)

            df = df[(df['date'] >= start) & (df['date'] <= end)]

            return df if not df.empty else None

        except Exception as e:
            warnings.warn(f"Failed to load cache for {ticker}: {e}")
            return None

    def _save_to_cache(self, ticker: str, df: pd.DataFrame):
        """Save data to cache"""
        cache_file = os.path.join(self.cache_dir, f"{ticker}.parquet")

        try:
            # Load existing cache if available
            if os.path.exists(cache_file):
                existing = pd.read_parquet(cache_file)
                existing['date'] = pd.to_datetime(existing['date'])

                # Merge with new data
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset='date', keep='last')

            df = df.sort_values('date').reset_index(drop=True)
            df.to_parquet(cache_file, index=False)

        except Exception as e:
            warnings.warn(f"Failed to cache {ticker}: {e}")


if __name__ == "__main__":
    # Test the loader
    loader = KoreanETFLoader()

    print("Available data sources:")
    for source, available in loader.available_sources.items():
        status = "✓" if available else "✗"
        print(f"  {status} {source}")

    print("\nTesting data fetch...")
    df = loader.get_etf_ohlcv(
        ticker="069500",
        start_date="2024-10-01",
        end_date="2024-10-26",
        source="auto"
    )

    if not df.empty:
        print(f"✓ Success! Got {len(df)} days of data")
        print(df.head())
    else:
        print("✗ No data available")
