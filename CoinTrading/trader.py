#########################################
##### Trading bot class             #####
##### Author: Jin Choi              #####
##### Contact: chlwls1589@snu.ac.kr #####
#########################################
import numpy as np
import pandas as pd
from math import log
from datetime import datetime

import ccxt
import binance
from secrets import binance_api_key, binance_secret_key

import telegram
from secrets import telegram_token, telegram_chat_id


class Trader(ccxt.binance):
    def __init__(self):
        super().__init__(config={
                "apiKey": binance_api_key,
                "secret": binance_secret_key,
                "enableRateLimit": True,
                "options": {"defaultType": "future"}
        })
        self.client = binance.Client()          # To use python-binance functionality
        self.telegram = telegram.Bot(token=telegram_token)

    def log(self, text):
        message = f"[Binance {datetime.now()}] {text}"
        self.telegram.sendMessage(chat_id=telegram_chat_id, text=message)
        print(message)

    def get_tickers(self):
        tickers = []
        for market in self.load_markets():
            if market[-4:] == "USDT":
                tickers.append(market)
        return tickers

    def get_current_price(self, ticker):
        return self.fetch_ticker(ticker)['last']

    def get_current_asset(self, ticker):
        positions = self.fetch_balance()['info']['positions']
        for pos in positions:
            if pos['symbol'] == ticker.replace("/", ""):
                return pos['positionAmt']
            else:
                continue

    def get_total_balance(self):
        total_balance = self.fetch_balance(params={"type":"future"})["USDT"]["total"]
        positions = self.fetch_balance()['info']['positions']
        for pos in positions:
            if float(pos['positionAmt']) != 0.:
                total_balance += float(pos['unrealizedProfit'])
        return total_balance

    def get_sample(self, ticker, limit=25):
        resp = self.fetch_ohlcv(
            symbol=ticker,
            timeframe="1d",
            since=None,
            limit=limit
        )
        sample = pd.DataFrame(resp, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        sample['datetime'] = pd.to_datetime(sample['datetime'], unit='ms')
        sample.set_index('datetime', inplace=True)
        return sample

    def set_leverage(self, ticker, lev):
        ticker = ticker.replace("/", "")
        resp = self.fapiPrivate_post_leverage({
            'symbol': ticker,
            'leverage': lev
        })
        print(resp)

    def buy_market_order(self, ticker, amount): # USDT
        price = self.get_current_price(ticker)
        quantity = amount/price

        info = self.client.get_symbol_info(ticker.replace("/", ""))
        # set precision
        for f in info['filters']:
            if f['filterType'] == "LOT_SIZE":
                precision = float(f['stepSize'])
                break
        precision = int(round(-log(precision, 10), 0))
        quantity = float(round(quantity, precision))

        #check whether exceed minQty
        minQty = float(info['filters'][2]['minQty'])
        if quantity <= minQty:
            return None

        order = self.create_market_buy_order(ticker, quantity)
        return order
