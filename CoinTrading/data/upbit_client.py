"""
Upbit Client

Wrapper for pyupbit library providing data fetching and caching.
"""

from typing import Optional, List, Dict
import pandas as pd
import pyupbit
import logging
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class UpbitClient:
    """
    Upbit API client for market data and trading.

    Provides interface similar to BinanceClient for consistency.
    """

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        cache_dir: str = ".cache/upbit"
    ):
        """
        Initialize Upbit client.

        Args:
            access_key: Upbit API access key (optional for public data)
            secret_key: Upbit API secret key (optional for public data)
            cache_dir: Directory for caching data
        """
        self.access_key = access_key
        self.secret_key = secret_key
        self.cache_dir = cache_dir

        # Create authenticated client if keys provided
        if access_key and secret_key:
            self.upbit = pyupbit.Upbit(access_key, secret_key)
            logger.info("Initialized UpbitClient with authentication")
        else:
            self.upbit = None
            logger.info("Initialized UpbitClient in public mode (no authentication)")

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

    def get_ticker_list(self, currency: str = "KRW") -> List[str]:
        """
        Get list of available tickers.

        Args:
            currency: Base currency (default: KRW)

        Returns:
            List of ticker symbols (e.g., ['KRW-BTC', 'KRW-ETH'])
        """
        try:
            tickers = pyupbit.get_tickers(fiat=currency)
            logger.info(f"Found {len(tickers)} {currency} markets")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching ticker list: {e}")
            return []

    def get_ohlcv(
        self,
        ticker: str,
        interval: str = "day",
        count: int = 200,
        to: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for a ticker.

        Args:
            ticker: Ticker symbol (e.g., 'KRW-BTC')
            interval: Time interval ('day', 'minute1', 'minute5', etc.)
            count: Number of candles to fetch (max 200 per request)
            to: End date in 'YYYYMMDD' or 'YYYYMMDD HH:MM:SS' format

        Returns:
            DataFrame with OHLCV data, or None if error
        """
        try:
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=count, to=to)

            if df is None or df.empty:
                logger.warning(f"{ticker}: No data returned")
                return None

            # Standardize column names to match existing system
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'value': 'value'  # Trading value in KRW
            })

            # Ensure index is timezone-naive for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            logger.debug(f"{ticker}: Fetched {len(df)} candles")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {ticker}: {e}")
            return None

    def get_ohlcv_paginated(
        self,
        ticker: str,
        interval: str = "day",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data with pagination for long time periods.

        Upbit limits to 200 candles per request, so we need pagination.

        Args:
            ticker: Ticker symbol (e.g., 'KRW-BTC')
            interval: Time interval (default: 'day')
            start_date: Start date (optional)
            end_date: End date (optional, default: now)

        Returns:
            DataFrame with OHLCV data, or None if error
        """
        if end_date is None:
            end_date = datetime.now()

        all_data = []
        current_to = end_date

        # Calculate approximately how many requests needed
        if start_date:
            days_diff = (end_date - start_date).days
            requests_needed = (days_diff // 200) + 1
            logger.info(f"{ticker}: Fetching ~{days_diff} days ({requests_needed} requests)")

        while True:
            # Fetch batch
            to_str = current_to.strftime('%Y%m%d %H:%M:%S')
            df = self.get_ohlcv(ticker, interval=interval, count=200, to=to_str)

            if df is None or df.empty:
                break

            all_data.append(df)

            # Check if we've reached start_date
            oldest_date = df.index.min()
            if start_date and oldest_date <= start_date:
                break

            # Move to next batch
            current_to = oldest_date - timedelta(seconds=1)

            # Safety check: don't go back more than 5 years
            if (end_date - oldest_date).days > 1825:
                logger.warning(f"{ticker}: Reached 5-year limit")
                break

        if not all_data:
            logger.error(f"{ticker}: No data fetched")
            return None

        # Combine all batches
        combined_df = pd.concat(all_data)
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        # Filter to date range if specified
        if start_date:
            combined_df = combined_df[combined_df.index >= start_date]
        if end_date:
            combined_df = combined_df[combined_df.index <= end_date]

        logger.info(f"{ticker}: Total {len(combined_df)} candles from {combined_df.index.min().date()} to {combined_df.index.max().date()}")
        return combined_df

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current price for a ticker.

        Args:
            ticker: Ticker symbol (e.g., 'KRW-BTC')

        Returns:
            Current price, or None if error
        """
        try:
            price = pyupbit.get_current_price(ticker)
            if price is None:
                logger.warning(f"{ticker}: No current price available")
            return price
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {e}")
            return None

    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dict of {ticker: price}
        """
        try:
            prices = pyupbit.get_current_price(tickers)

            # Handle both dict and single value returns
            if isinstance(prices, dict):
                # Filter out None values
                return {k: v for k, v in prices.items() if v is not None}
            elif len(tickers) == 1 and prices is not None:
                return {tickers[0]: prices}
            else:
                logger.warning("No prices returned")
                return {}

        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
            return {}

    def get_orderbook(self, ticker: str) -> Optional[Dict]:
        """
        Get orderbook for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Orderbook dict with 'orderbook_units' containing bids/asks
        """
        try:
            orderbook = pyupbit.get_orderbook(ticker)
            return orderbook
        except Exception as e:
            logger.error(f"Error fetching orderbook for {ticker}: {e}")
            return None

    def get_balances(self) -> Optional[pd.DataFrame]:
        """
        Get account balances (requires authentication).

        Returns:
            DataFrame with columns: currency, balance, locked, avg_buy_price, avg_buy_price_modified
        """
        if self.upbit is None:
            logger.error("Authentication required for get_balances()")
            return None

        try:
            balances = self.upbit.get_balances()
            if balances:
                df = pd.DataFrame(balances)
                # Convert numeric columns
                numeric_cols = ['balance', 'locked', 'avg_buy_price']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            else:
                logger.warning("No balances returned")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching balances: {e}")
            return None

    def get_balance(self, ticker: str = "KRW") -> float:
        """
        Get balance for specific currency.

        Args:
            ticker: Currency ticker (default: 'KRW')

        Returns:
            Balance amount
        """
        if self.upbit is None:
            logger.error("Authentication required for get_balance()")
            return 0.0

        try:
            balance = self.upbit.get_balance(ticker)
            return float(balance) if balance else 0.0
        except Exception as e:
            logger.error(f"Error fetching balance for {ticker}: {e}")
            return 0.0

    def buy_market_order(self, ticker: str, price: float) -> Optional[Dict]:
        """
        Place market buy order.

        Args:
            ticker: Ticker symbol (e.g., 'KRW-BTC')
            price: Total KRW amount to spend

        Returns:
            Order result dict
        """
        if self.upbit is None:
            logger.error("Authentication required for buy_market_order()")
            return None

        try:
            result = self.upbit.buy_market_order(ticker, price)
            logger.info(f"BUY {ticker}: {price} KRW")
            return result
        except Exception as e:
            logger.error(f"Error placing buy order for {ticker}: {e}")
            return None

    def sell_market_order(self, ticker: str, volume: float) -> Optional[Dict]:
        """
        Place market sell order.

        Args:
            ticker: Ticker symbol (e.g., 'KRW-BTC')
            volume: Amount of coin to sell

        Returns:
            Order result dict
        """
        if self.upbit is None:
            logger.error("Authentication required for sell_market_order()")
            return None

        try:
            result = self.upbit.sell_market_order(ticker, volume)
            logger.info(f"SELL {ticker}: {volume} units")
            return result
        except Exception as e:
            logger.error(f"Error placing sell order for {ticker}: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("UpbitClient Example (Public Data)")
    print("=" * 80)

    # Create client without authentication
    client = UpbitClient()

    # Get available tickers
    tickers = client.get_ticker_list("KRW")
    print(f"\nFound {len(tickers)} KRW markets")
    print(f"Sample tickers: {tickers[:5]}")

    # Get current prices
    test_tickers = ['KRW-BTC', 'KRW-ETH', 'KRW-USDT']
    prices = client.get_current_prices(test_tickers)
    print(f"\nCurrent prices:")
    for ticker, price in prices.items():
        print(f"  {ticker}: {price:,.0f} KRW")

    # Get historical data
    print(f"\nFetching OHLCV data for KRW-BTC...")
    df = client.get_ohlcv('KRW-BTC', interval='day', count=30)
    if df is not None:
        print(f"  Fetched {len(df)} days")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Latest close: {df['close'].iloc[-1]:,.0f} KRW")

    print("\n✓ UpbitClient ready!")
