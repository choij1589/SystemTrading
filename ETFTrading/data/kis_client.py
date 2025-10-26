"""
Korea Investment & Securities API Client

This module provides a Python wrapper for the KIS (Korea Investment & Securities)
REST API for stock and ETF trading.
"""

import os
import time
import json
import yaml
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd


class KISClient:
    """
    Korea Investment & Securities API Client

    Supports both real and mock trading environments.
    """

    # API URLs
    REAL_BASE_URL = "https://openapi.koreainvestment.com:9443"
    MOCK_BASE_URL = "https://openapivts.koreainvestment.com:29443"

    def __init__(self, mode: str = "mock", config_path: Optional[str] = None):
        """
        Initialize KIS API client

        Args:
            mode: "mock" for paper trading, "real" for live trading
            config_path: Path to secrets.yaml file
        """
        self.mode = mode
        self.base_url = self.MOCK_BASE_URL if mode == "mock" else self.REAL_BASE_URL

        # Load credentials
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "../config/secrets.yaml"
            )

        self.credentials = self._load_credentials(config_path)
        self.access_token = None
        self.token_expires_at = None

    def _load_credentials(self, config_path: str) -> Dict:
        """Load API credentials from secrets.yaml"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Secrets file not found: {config_path}\n"
                "Please copy secrets.yaml.example to secrets.yaml and fill in your credentials"
            )

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        mode_config = config['kis'][self.mode]
        return {
            'app_key': mode_config['app_key'],
            'app_secret': mode_config['app_secret'],
            'account_number': mode_config['account_number'],
            'account_code': mode_config['account_code']
        }

    def _get_access_token(self) -> str:
        """
        Get OAuth 2.0 access token

        Returns:
            Access token string
        """
        # Return cached token if still valid
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at:
                return self.access_token

        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": self.credentials['app_key'],
            "appsecret": self.credentials['app_secret']
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()
        self.access_token = result['access_token']

        # Token typically expires in 24 hours, set expiry to 23 hours for safety
        self.token_expires_at = datetime.now() + timedelta(hours=23)

        return self.access_token

    def _make_request(
        self,
        tr_id: str,
        url_path: str,
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Dict:
        """
        Make authenticated API request

        Args:
            tr_id: Transaction ID (API endpoint identifier)
            url_path: API endpoint path
            params: Query parameters
            method: HTTP method

        Returns:
            Response JSON
        """
        token = self._get_access_token()

        url = f"{self.base_url}{url_path}"
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.credentials['app_key'],
            "appsecret": self.credentials['app_secret'],
            "tr_id": tr_id
        }

        if method == "GET":
            response = requests.get(url, headers=headers, params=params)
        else:
            response = requests.post(url, headers=headers, json=params)

        response.raise_for_status()
        return response.json()

    def get_daily_price(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Get daily OHLCV data for a stock/ETF

        Args:
            ticker: Stock/ETF code (e.g., "069500")
            start_date: Start date in "YYYYMMDD" format
            end_date: End date in "YYYYMMDD" format
            adjusted: Whether to adjust for splits/dividends

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        # KIS API uses different TR_ID for mock vs real
        tr_id = "FHKST03010100" if self.mode == "real" else "FHKST03010100"

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # J for stock market
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": start_date,
            "FID_INPUT_DATE_2": end_date,
            "FID_PERIOD_DIV_CODE": "D",  # D for daily
            "FID_ORG_ADJ_PRC": "0" if adjusted else "1"  # 0=adjusted, 1=original
        }

        try:
            result = self._make_request(
                tr_id=tr_id,
                url_path="/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                params=params
            )

            if result['rt_cd'] != '0':
                raise Exception(f"API Error: {result.get('msg1', 'Unknown error')}")

            # Parse response
            data = result['output2']

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # Rename columns
            df = df.rename(columns={
                'stck_bsop_date': 'date',
                'stck_oprc': 'open',
                'stck_hgpr': 'high',
                'stck_lwpr': 'low',
                'stck_clpr': 'close',
                'acml_vol': 'volume'
            })

            # Select and convert columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['date'] = pd.to_datetime(df['date'])

            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current price for a stock/ETF

        Args:
            ticker: Stock/ETF code

        Returns:
            Current price or None if error
        """
        tr_id = "FHKST01010100" if self.mode == "real" else "FHKST01010100"

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker
        }

        try:
            result = self._make_request(
                tr_id=tr_id,
                url_path="/uapi/domestic-stock/v1/quotations/inquire-price",
                params=params
            )

            if result['rt_cd'] != '0':
                return None

            return float(result['output']['stck_prpr'])

        except Exception as e:
            print(f"Error fetching current price for {ticker}: {e}")
            return None

    def place_order(
        self,
        ticker: str,
        order_type: str,  # "buy" or "sell"
        quantity: int,
        price: Optional[float] = None  # None for market order
    ) -> Dict:
        """
        Place buy or sell order

        Args:
            ticker: Stock/ETF code
            order_type: "buy" or "sell"
            quantity: Number of shares
            price: Limit price (None for market order)

        Returns:
            Order result dictionary
        """
        # Determine TR_ID based on order type and mode
        if order_type == "buy":
            tr_id = "TTTC0802U" if self.mode == "real" else "VTTC0802U"
        else:
            tr_id = "TTTC0801U" if self.mode == "real" else "VTTC0801U"

        # Order type code
        ord_dv = "00" if price else "01"  # 00=limit, 01=market

        data = {
            "CANO": self.credentials['account_number'],
            "ACNT_PRDT_CD": self.credentials['account_code'],
            "PDNO": ticker,
            "ORD_DVSN": ord_dv,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(int(price)) if price else "0"
        }

        try:
            result = self._make_request(
                tr_id=tr_id,
                url_path="/uapi/domestic-stock/v1/trading/order-cash",
                params=data,
                method="POST"
            )

            return result

        except Exception as e:
            print(f"Error placing {order_type} order for {ticker}: {e}")
            return {"rt_cd": "-1", "msg1": str(e)}

    def get_balance(self) -> pd.DataFrame:
        """
        Get current account balance and holdings

        Returns:
            DataFrame with holdings information
        """
        tr_id = "TTTC8434R" if self.mode == "real" else "VTTC8434R"

        params = {
            "CANO": self.credentials['account_number'],
            "ACNT_PRDT_CD": self.credentials['account_code'],
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }

        try:
            result = self._make_request(
                tr_id=tr_id,
                url_path="/uapi/domestic-stock/v1/trading/inquire-balance",
                params=params
            )

            if result['rt_cd'] != '0':
                return pd.DataFrame()

            holdings = result['output1']
            df = pd.DataFrame(holdings)

            return df

        except Exception as e:
            print(f"Error fetching balance: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Simple test
    print("KIS API Client")
    print("=" * 50)
    print("Note: This requires valid API credentials in secrets.yaml")
    print("To test, uncomment the code below and add your credentials")

    # Example usage (uncomment to test):
    # client = KISClient(mode="mock")
    #
    # # Get daily price data
    # df = client.get_daily_price(
    #     ticker="069500",  # KODEX 200
    #     start_date="20240101",
    #     end_date="20241026"
    # )
    # print(df.head())
    # print(df.tail())
