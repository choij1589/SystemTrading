"""
ETF Data Loader with Disk Caching

Handles downloading and caching ETF price data from KIS API.
Implements disk caching to avoid redundant API calls.
"""

import os
import yaml
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pickle
import json

from .kis_client import KISClient

logger = logging.getLogger(__name__)


class ETFDataLoader:
    """
    Manages ETF OHLCV data fetching and caching.

    Features:
    - Disk caching (parquet or pickle format)
    - Cache invalidation by date
    - Batch downloads for multiple ETFs
    - ETF universe management
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        cache_format: str = "parquet",
        mode: str = "mock",
        config_path: Optional[str] = None
    ):
        """
        Initialize ETF data loader.

        Args:
            cache_dir: Directory for cached data
            cache_format: 'parquet' or 'pickle'
            mode: 'mock' or 'real' for KIS API
            config_path: Path to secrets.yaml
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_format = cache_format.lower()
        if self.cache_format not in ["parquet", "pickle"]:
            raise ValueError(f"Invalid cache_format: {cache_format}")

        # Initialize KIS client
        self.client = KISClient(mode=mode, config_path=config_path)

        # Load ETF universe
        self.etf_universe = self._load_etf_universe()

        logger.info(f"ETFDataLoader initialized with cache: {self.cache_dir}")

    def _load_etf_universe(self) -> Dict:
        """Load ETF universe from config file"""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "../config/etf_universe.yaml"
        )

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_cache_path(self, ticker: str) -> Path:
        """Get cache file path for a ticker."""
        ext = "parquet" if self.cache_format == "parquet" else "pkl"
        filename = f"{ticker}_daily.{ext}"
        return self.cache_dir / filename

    def _get_metadata_path(self, ticker: str) -> Path:
        """Get metadata file path for a ticker."""
        filename = f"{ticker}_daily.meta.json"
        return self.cache_dir / filename

    def _save_to_cache(self, df: pd.DataFrame, ticker: str) -> None:
        """
        Save DataFrame to cache.

        Args:
            df: DataFrame to cache
            ticker: ETF ticker code
        """
        cache_path = self._get_cache_path(ticker)

        try:
            if self.cache_format == "parquet":
                df.to_parquet(cache_path)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(df, f)

            # Save metadata
            self._save_metadata(df, ticker)

            logger.info(f"Cached {len(df)} rows for {ticker}")

        except Exception as e:
            logger.warning(f"Failed to cache {ticker}: {e}")

    def _save_metadata(self, df: pd.DataFrame, ticker: str) -> None:
        """
        Save metadata for cached data.

        Args:
            df: DataFrame being cached
            ticker: ETF ticker code
        """
        metadata_path = self._get_metadata_path(ticker)

        try:
            metadata = {
                'ticker': ticker,
                'start_date': df['date'].iloc[0].isoformat() if not df.empty else None,
                'end_date': df['date'].iloc[-1].isoformat() if not df.empty else None,
                'num_rows': len(df),
                'cache_timestamp': datetime.now().isoformat(),
                'last_close_price': float(df['close'].iloc[-1]) if not df.empty else None,
                'columns': list(df.columns)
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save metadata for {ticker}: {e}")

    def _load_from_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from cache if available.

        Args:
            ticker: ETF ticker code

        Returns:
            Cached DataFrame or None
        """
        cache_path = self._get_cache_path(ticker)

        if not cache_path.exists():
            return None

        try:
            if self.cache_format == "parquet":
                df = pd.read_parquet(cache_path)
            else:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)

            logger.info(f"Loaded {len(df)} rows from cache for {ticker}")
            return df

        except Exception as e:
            logger.warning(f"Failed to load cache for {ticker}: {e}")
            return None

    def _is_cache_valid(self, ticker: str, max_age_hours: int = 24) -> bool:
        """
        Check if cached data is still valid.

        Args:
            ticker: ETF ticker code
            max_age_hours: Maximum cache age in hours

        Returns:
            True if cache is valid
        """
        metadata_path = self._get_metadata_path(ticker)

        if not metadata_path.exists():
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            cache_time = datetime.fromisoformat(metadata['cache_timestamp'])
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600

            return age_hours < max_age_hours

        except Exception as e:
            logger.warning(f"Failed to check cache validity for {ticker}: {e}")
            return False

    def load_single(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        force_update: bool = False
    ) -> pd.DataFrame:
        """
        Load data for a single ETF.

        Args:
            ticker: ETF ticker code (e.g., "069500")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date (None = today)
            use_cache: Whether to use cached data
            force_update: Force update even if cache is valid

        Returns:
            DataFrame with OHLCV data
        """
        # Try cache first
        if use_cache and not force_update:
            if self._is_cache_valid(ticker):
                cached_df = self._load_from_cache(ticker)
                if cached_df is not None:
                    # Filter by date range
                    mask = cached_df['date'] >= pd.to_datetime(start_date)
                    if end_date:
                        mask &= cached_df['date'] <= pd.to_datetime(end_date)
                    return cached_df[mask].reset_index(drop=True)

        # Fetch from API
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start_date_fmt = start_date.replace("-", "")
        end_date_fmt = end_date.replace("-", "")

        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")

        df = self.client.get_daily_price(
            ticker=ticker,
            start_date=start_date_fmt,
            end_date=end_date_fmt,
            adjusted=True
        )

        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        # Save to cache
        if use_cache:
            self._save_to_cache(df, ticker)

        return df

    def load_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        force_update: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple ETFs.

        Args:
            tickers: List of ETF ticker codes
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date (None = today)
            use_cache: Whether to use cached data
            force_update: Force update even if cache is valid

        Returns:
            Dictionary mapping ticker to DataFrame
        """
        result = {}

        for ticker in tickers:
            try:
                df = self.load_single(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache,
                    force_update=force_update
                )

                if not df.empty:
                    result[ticker] = df
                else:
                    logger.warning(f"Empty data for {ticker}")

            except Exception as e:
                logger.error(f"Failed to load {ticker}: {e}")

        return result

    def get_etf_info(self, ticker: str) -> Optional[Dict]:
        """
        Get ETF information from universe.

        Args:
            ticker: ETF ticker code

        Returns:
            Dictionary with ETF info or None
        """
        for category, etfs in self.etf_universe.items():
            if category == 'strategy_universe':
                continue
            if ticker in etfs:
                return {
                    'ticker': ticker,
                    'category': category,
                    **etfs[ticker]
                }
        return None

    def get_strategy_universe(self, strategy_name: str) -> List[str]:
        """
        Get list of ETFs for a specific strategy.

        Args:
            strategy_name: Name of strategy

        Returns:
            List of ticker codes
        """
        return self.etf_universe.get('strategy_universe', {}).get(strategy_name, [])


if __name__ == "__main__":
    # Simple test
    print("ETF Data Loader Test")
    print("=" * 50)

    # Example usage (requires valid API credentials):
    # loader = ETFDataLoader(mode="mock")
    #
    # # Load single ETF
    # df = loader.load_single(
    #     ticker="069500",  # KODEX 200
    #     start_date="2020-01-01",
    #     use_cache=True
    # )
    # print(df.head())
    # print(df.tail())
    #
    # # Get strategy universe
    # tickers = loader.get_strategy_universe("global_asset_allocation")
    # print(f"Strategy tickers: {tickers}")
