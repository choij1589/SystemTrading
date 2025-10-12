"""
Binance API Client Wrapper

Clean wrapper around python-binance for fetching OHLCV data.
Focuses on read-only operations (no trading).
"""

from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from binance import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)


class BinanceClient:
    """
    Wrapper for Binance API operations.

    Provides read-only access to market data with error handling
    and rate limiting.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        market_type: str = "futures"
    ):
        """
        Initialize Binance client.

        Args:
            api_key: Binance API key (optional for public endpoints)
            secret_key: Binance secret key (optional for public endpoints)
            market_type: 'spot' or 'futures'
        """
        self.client = Client(api_key, secret_key)
        self.market_type = market_type.lower()

        if self.market_type not in ["spot", "futures"]:
            raise ValueError(f"Invalid market_type: {market_type}. Must be 'spot' or 'futures'")

        logger.info(f"Initialized BinanceClient for {market_type} market")

    def get_tickers(self, quote_asset: str = "USDT") -> List[str]:
        """
        Get all trading pairs ending with quote_asset.

        Args:
            quote_asset: Quote currency (default: USDT)

        Returns:
            List of ticker symbols (e.g., ['BTCUSDT', 'ETHUSDT', ...])
        """
        try:
            if self.market_type == "futures":
                prices = self.client.futures_mark_price()
                tickers = [
                    item['symbol']
                    for item in prices
                    if item['symbol'].endswith(quote_asset)
                ]
            else:
                prices = self.client.get_all_tickers()
                tickers = [
                    item['symbol']
                    for item in prices
                    if item['symbol'].endswith(quote_asset)
                ]

            logger.info(f"Found {len(tickers)} {quote_asset} pairs")
            return sorted(tickers)

        except BinanceAPIException as e:
            logger.error(f"Failed to fetch tickers: {e}")
            raise

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = Client.KLINE_INTERVAL_1DAY,
        start_date: Optional[str] = None,
        limit: int = 500,
        paginate: bool = False
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (default: 1 day)
            start_date: Start date string (e.g., '1 Apr, 2020') or None
            limit: Number of candles to fetch (max 1000 per request)
            paginate: If True and start_date provided, fetch all data from start_date to now

        Returns:
            DataFrame with columns: [open, high, low, close, volume]
            Index: DatetimeIndex
        """
        # If pagination requested with start_date, fetch all data in chunks
        if paginate and start_date:
            return self._get_ohlcv_paginated(symbol, interval, start_date)

        try:
            if self.market_type == "futures":
                klines = self.client.futures_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_date,
                    limit=limit
                )
            else:
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_date,
                    limit=limit
                )

            if not klines:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = self._parse_klines(klines)
            logger.debug(f"Fetched {len(df)} candles for {symbol}")
            return df

        except BinanceAPIException as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            raise

    def _parse_klines(self, klines: List) -> pd.DataFrame:
        """
        Parse klines data into DataFrame.

        Args:
            klines: Raw klines data from Binance API

        Returns:
            Parsed DataFrame with OHLCV data
        """
        df = pd.DataFrame(
            klines,
            columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ]
        )

        # Convert types and set index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')

        # Select and convert OHLCV columns
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[ohlcv_cols].astype(float)

        return df

    def _get_ohlcv_paginated(
        self,
        symbol: str,
        interval: str,
        start_date: str
    ) -> pd.DataFrame:
        """
        Fetch all OHLCV data from start_date to now using pagination.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            start_date: Start date string

        Returns:
            Complete DataFrame with all historical data
        """
        all_klines = []
        current_start = start_date
        batch_size = 1000  # Max allowed by Binance

        logger.info(f"Fetching paginated data for {symbol} from {start_date}")

        while True:
            try:
                if self.market_type == "futures":
                    klines = self.client.futures_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=current_start,
                        limit=batch_size
                    )
                else:
                    klines = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=current_start,
                        limit=batch_size
                    )

                if not klines:
                    break

                all_klines.extend(klines)

                # Check if we got less than batch_size (reached end)
                if len(klines) < batch_size:
                    break

                # Update start date to last timestamp + 1ms
                last_timestamp = klines[-1][0]
                current_start = str(last_timestamp + 1)

                logger.debug(f"Fetched {len(all_klines)} candles so far...")

            except BinanceAPIException as e:
                logger.error(f"Failed during pagination for {symbol}: {e}")
                break

        if not all_klines:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Remove duplicates (can happen at batch boundaries)
        df = self._parse_klines(all_klines)
        df = df[~df.index.duplicated(keep='first')]

        logger.info(f"Fetched {len(df)} total candles for {symbol}")
        return df

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        Get trading rules and filters for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            Symbol info dict or None if not found
        """
        try:
            if self.market_type == "futures":
                info = self.client.futures_exchange_info()
            else:
                info = self.client.get_exchange_info()

            for symbol_info in info['symbols']:
                if symbol_info['symbol'] == symbol:
                    return symbol_info

            logger.warning(f"Symbol info not found for {symbol}")
            return None

        except BinanceAPIException as e:
            logger.error(f"Failed to fetch symbol info for {symbol}: {e}")
            raise

    def get_lot_size_filter(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Extract LOT_SIZE filter for precision/quantity validation.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dict with 'minQty', 'maxQty', 'stepSize' or None
        """
        info = self.get_symbol_info(symbol)
        if not info:
            return None

        for f in info.get('filters', []):
            if f['filterType'] == 'LOT_SIZE':
                return {
                    'minQty': float(f['minQty']),
                    'maxQty': float(f['maxQty']),
                    'stepSize': float(f['stepSize'])
                }

        return None

    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price (float)
        """
        try:
            if self.market_type == "futures":
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
            else:
                ticker = self.client.get_symbol_ticker(symbol=symbol)

            return float(ticker['price'])

        except BinanceAPIException as e:
            logger.error(f"Failed to fetch price for {symbol}: {e}")
            raise

    def validate_data_completeness(
        self,
        df: pd.DataFrame,
        expected_length: Optional[int] = None,
        check_today: bool = True
    ) -> Tuple[bool, str]:
        """
        Validate fetched data for completeness.

        Args:
            df: DataFrame to validate
            expected_length: Expected number of rows (None to skip)
            check_today: Whether last date should be today

        Returns:
            (is_valid, message) tuple
        """
        if df.empty:
            return False, "DataFrame is empty"

        if expected_length and len(df) != expected_length:
            return False, f"Expected {expected_length} rows, got {len(df)}"

        if check_today:
            last_date = df.index[-1].date()
            today = datetime.today().date()
            if last_date != today:
                return False, f"Last date {last_date} != today {today}"

        # Check for missing values
        if df.isnull().any().any():
            return False, "Contains NaN values"

        return True, "Valid"


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize client (no API key needed for public data)
    client = BinanceClient(market_type="futures")

    # Get all USDT pairs
    tickers = client.get_tickers()
    print(f"Found {len(tickers)} USDT pairs")
    print(f"First 5: {tickers[:5]}")

    # Fetch BTC data
    btc_data = client.get_ohlcv('BTCUSDT', limit=100)
    print(f"\nBTC data shape: {btc_data.shape}")
    print(btc_data.head())

    # Get current price
    price = client.get_current_price('BTCUSDT')
    print(f"\nBTC current price: ${price:,.2f}")
