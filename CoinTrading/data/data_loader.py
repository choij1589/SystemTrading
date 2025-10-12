"""
Data Loader with Disk Caching

Handles downloading and caching OHLCV data from Binance.
Implements disk caching to avoid redundant API calls.
"""

from typing import List, Dict, Optional
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pickle
import json

from .binance_client import BinanceClient

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Manages OHLCV data fetching and caching.

    Features:
    - Disk caching (parquet or pickle format)
    - Cache invalidation by date
    - Batch downloads for multiple tickers
    """

    def __init__(
        self,
        cache_dir: str = ".cache/ohlcv",
        cache_format: str = "parquet",
        market_type: str = "futures",
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None
    ):
        """
        Initialize data loader.

        Args:
            cache_dir: Directory for cached data
            cache_format: 'parquet' or 'pickle'
            market_type: 'spot' or 'futures'
            api_key: Binance API key (optional)
            secret_key: Binance secret key (optional)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_format = cache_format.lower()
        if self.cache_format not in ["parquet", "pickle"]:
            raise ValueError(f"Invalid cache_format: {cache_format}")

        self.client = BinanceClient(api_key, secret_key, market_type)

        logger.info(f"DataLoader initialized with cache: {self.cache_dir}")

    def _get_cache_path(self, symbol: str, interval: str) -> Path:
        """Get cache file path for a symbol."""
        ext = "parquet" if self.cache_format == "parquet" else "pkl"
        filename = f"{symbol}_{interval}.{ext}"
        return self.cache_dir / filename

    def _get_metadata_path(self, symbol: str, interval: str) -> Path:
        """Get metadata file path for a symbol."""
        filename = f"{symbol}_{interval}.meta.json"
        return self.cache_dir / filename

    def _save_metadata(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> None:
        """
        Save metadata for cached data.

        Args:
            df: DataFrame being cached
            symbol: Trading pair symbol
            interval: Kline interval
        """
        metadata_path = self._get_metadata_path(symbol, interval)

        try:
            metadata = {
                'symbol': symbol,
                'interval': interval,
                'start_date': df.index[0].isoformat() if not df.empty else None,
                'end_date': df.index[-1].isoformat() if not df.empty else None,
                'num_rows': len(df),
                'cache_timestamp': datetime.now().isoformat(),
                'last_close_price': float(df['close'].iloc[-1]) if not df.empty and 'close' in df.columns else None,
                'columns': list(df.columns)
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Saved metadata for {symbol}")

        except Exception as e:
            logger.warning(f"Failed to save metadata for {symbol}: {e}")

    def _load_metadata(
        self,
        symbol: str,
        interval: str
    ) -> Optional[Dict]:
        """
        Load metadata for cached data.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval

        Returns:
            Metadata dict or None if not found
        """
        metadata_path = self._get_metadata_path(symbol, interval)

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata

        except Exception as e:
            logger.warning(f"Failed to load metadata for {symbol}: {e}")
            return None

    def _load_from_cache(
        self,
        symbol: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Load data from cache if it exists and is up-to-date.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval

        Returns:
            Cached DataFrame or None if cache miss/stale
        """
        cache_path = self._get_cache_path(symbol, interval)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {symbol}")
            return None

        try:
            # Load cached data
            if self.cache_format == "parquet":
                df = pd.read_parquet(cache_path)
            else:
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)

            # Check if cache is up-to-date (last date should be today or yesterday)
            last_date = df.index[-1].date()
            today = datetime.today().date()
            yesterday = (datetime.today() - timedelta(days=1)).date()

            if last_date in [today, yesterday]:
                logger.debug(f"Cache hit: {symbol} (last: {last_date})")
                return df
            else:
                logger.debug(f"Cache stale: {symbol} (last: {last_date})")
                return None

        except Exception as e:
            logger.warning(f"Failed to load cache for {symbol}: {e}")
            return None

    def _save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> None:
        """
        Save DataFrame to cache.

        Args:
            df: DataFrame to cache
            symbol: Trading pair symbol
            interval: Kline interval
        """
        cache_path = self._get_cache_path(symbol, interval)

        try:
            if self.cache_format == "parquet":
                df.to_parquet(cache_path)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(df, f)

            # Save metadata alongside cache
            self._save_metadata(df, symbol, interval)

            logger.debug(f"Cached {len(df)} rows for {symbol}")

        except Exception as e:
            logger.warning(f"Failed to cache {symbol}: {e}")

    def load_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[str] = None,
        limit: int = 500,
        use_cache: bool = True,
        paginate: bool = False
    ) -> pd.DataFrame:
        """
        Load OHLCV data with caching.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            start_date: Start date string (e.g., '1 Apr, 2020')
            limit: Number of candles to fetch
            use_cache: Whether to use cached data
            paginate: If True, fetch all data from start_date to now (ignores cache)

        Returns:
            DataFrame with OHLCV data
        """
        # Try cache first (if enabled and not paginating)
        if use_cache and not paginate:
            cached_df = self._load_from_cache(symbol, interval)
            if cached_df is not None:
                return cached_df

        # Download from API
        if paginate and start_date:
            logger.info(f"Downloading {symbol} with pagination from {start_date}")
        else:
            logger.info(f"Downloading {symbol} from Binance API")

        df = self.client.get_ohlcv(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            limit=limit,
            paginate=paginate
        )

        # Save to cache (only if not paginating, as paginated data can be stale)
        if use_cache and not paginate and not df.empty:
            self._save_to_cache(df, symbol, interval)

        return df

    def load_multiple(
        self,
        symbols: List[str],
        interval: str = "1d",
        start_date: Optional[str] = None,
        limit: int = 500,
        use_cache: bool = True,
        paginate: bool = False,
        skip_errors: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for multiple symbols.

        Args:
            symbols: List of trading pair symbols
            interval: Kline interval
            start_date: Start date string
            limit: Number of candles per symbol
            use_cache: Whether to use cached data
            paginate: If True, fetch all data from start_date to now
            skip_errors: Skip symbols that fail to download

        Returns:
            Dict mapping symbol to DataFrame
        """
        results = {}

        for i, symbol in enumerate(symbols):
            try:
                df = self.load_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    limit=limit,
                    use_cache=use_cache,
                    paginate=paginate
                )

                if not df.empty:
                    results[symbol] = df

                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Loaded {i + 1}/{len(symbols)} symbols")

            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                if not skip_errors:
                    raise

        logger.info(f"Successfully loaded {len(results)}/{len(symbols)} symbols")
        return results

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached data.

        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            # Clear specific symbol
            for interval in ["1d", "1h", "4h"]:
                cache_path = self._get_cache_path(symbol, interval)
                if cache_path.exists():
                    cache_path.unlink()
                    logger.info(f"Cleared cache for {symbol}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*"):
                cache_file.unlink()
            logger.info("Cleared all cache")

    def get_cache_stats(self) -> Dict:
        """
        Get statistics about cached data.

        Returns:
            Dict with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*"))

        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'num_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

    def get_all_metadata(self, interval: str = "1d") -> pd.DataFrame:
        """
        Get metadata for all cached symbols.

        Args:
            interval: Kline interval (default: '1d')

        Returns:
            DataFrame with metadata for all symbols
        """
        metadata_list = []

        # Find all metadata files
        pattern = f"*_{interval}.meta.json"
        for meta_file in self.cache_dir.glob(pattern):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load {meta_file.name}: {e}")

        if not metadata_list:
            return pd.DataFrame()

        return pd.DataFrame(metadata_list)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize loader
    loader = DataLoader(cache_dir=".cache/test", cache_format="parquet")

    # Load single symbol
    btc_data = loader.load_ohlcv('BTCUSDT', limit=200)
    print(f"BTC data: {btc_data.shape}")
    print(btc_data.tail())

    # Load multiple symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    data_dict = loader.load_multiple(symbols, limit=100)
    print(f"\nLoaded {len(data_dict)} symbols")

    # Cache stats
    stats = loader.get_cache_stats()
    print(f"\nCache stats: {stats}")

    # Test cache hit (should be instant)
    print("\nTesting cache hit...")
    btc_data2 = loader.load_ohlcv('BTCUSDT', limit=200)
    print(f"Loaded from cache: {btc_data2.shape}")
