from math import log
import numpy as np
import pandas as pd
from time import sleep
from datetime import datetime

import telegram
import ccxt
from binance import Client
from secrets import telegram_token, telegram_chat_id
from secrets import binance_api_key, binance_secret_key

class Trader(ccxt.binance):
    def __init__(self):
        super().__init__(config={
            "apiKey": binance_api_key,
            "secret": binance_secret_key,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}
        })
        self.client = Client()
        self.telegram = telegram.Bot(token=telegram_token)

    def send_message(self, text):
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

    def get_sample(self, ticker):
        resp = self.fetch_ohlcv(
            symbol=ticker,
            timeframe="1d",
            since=None,
            limit=220
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

if __name__ == "__main__":
    trader = Trader()
    trader.send_message("Start trading...")

    # cancel orders first
    positions = trader.fetch_balance()['info']['positions']
    for pos in positions:
        if float(pos['positionAmt']) != 0.:
            ticker= f"{pos['symbol'][:-4]}/{pos['symbol'][-4:]}"
            amount = pos['positionAmt']
            trader.create_market_sell_order(ticker, amount, params={"type": "future"})
    trader.send_message("Cancled all orders")
    sleep(20)

    tickers = trader.get_tickers()
    samples = dict()
    for ticker in tickers:
        sample = trader.get_sample(ticker)
        if len(sample) != 220: continue
        # check update
        assert sample.index[-1].date() == datetime.today().date()

        # preprocess
        sample['noise'] = 1.-abs(sample['close'] - sample['open'])/(sample['high']-sample['low'])
        sample['reward_up'] = np.log(1. + sample['close'].pct_change())
        sample['reward_down'] = np.log(1. - sample['close'].pct_change())
        sample.dropna(inplace=True)

        sample['value_up'] = 0.
        sample['value_down'] = 0.
        for idx in sample.index:
            if idx == sample.index[0]:
                sample.loc[idx, 'value_up'] = sample.loc[idx, 'reward_up']
                sample.loc[idx, 'value_down'] = sample.loc[idx, 'reward_down']
                continue
            if idx == sample.index[-1]:
                continue
            gamma = 1.-0.5*sample.loc[idx, 'noise']
            sample.loc[idx, 'value_up'] = sample.loc[idx, 'reward_up'] + gamma*(sample.shift(1).loc[idx, 'value_up'])
            sample.loc[idx, 'value_down'] = sample.loc[idx, 'reward_down'] + gamma*(sample.shift(1).loc[idx, 'value_down'])
        sample = sample.iloc[-12:].copy()
        sample['momentum'] = (sample['close'] - sample.shift(10)['close'])/sample.shift(10)['close']
        sample.dropna(inplace=True)

        samples[ticker] = sample.copy(); del sample

    volumes = dict()
    for ticker in samples.keys():
        if len(samples[ticker]) == 0:
            continue
        volumes[ticker] = samples[ticker].iloc[-2]['close']*samples[ticker].iloc[-2]['volume']
    top21v = dict(sorted(volumes.items(), key=(lambda x: x[1]), reverse=True)[:21])
    values = dict()
    for ticker in top21v.keys():
        values[f"{ticker}-UP"] = samples[ticker].iloc[-2]['value_up']
        values[f"{ticker}-DOWN"] = samples[ticker].iloc[-2]['value_down']
    res = dict(sorted(values.items(), key=(lambda x: x[1]), reverse=True)[:6])
    trader.send_message(f"target coins: {list(res.keys())}")

    # Start trading
    target_amount = trader.get_total_balance()/len(res)*0.9
    trader.send_message(f"target_amount: {target_amount}")
    for coin in res.keys():
        if "UP" in coin:
            ticker = coin[:-3]
            if samples[ticker].iloc[-2]['momentum'] < 0.3:
                continue
            elif samples[ticker].iloc[-2]['momentum'] < 0.8:
                trader.set_leverage(ticker, 1)
                order = trader.buy_market_order(ticker, target_amount)
            else:
                trader.set_leverage(ticker, 4)
                order = trader.buy_market_order(ticker, target_amount*2.)
        else:   # Down
            ticker = coin[:-5]
            if samples[ticker].iloc[-2]['momentum'] > -0.3:
                continue
            elif samples[ticker].iloc[-2]['momentum'] > -0.6:
                trader.set_leverage(ticker, 2)
                order = trader.buy_market_order(ticker, target_amount*2)
            else:
                trader.set_leverage(ticker, 12)
                order = trader.buy_market_order(ticker, target_amount*5)
        trader.send_message(order)
    trader.send_message("End trading")
