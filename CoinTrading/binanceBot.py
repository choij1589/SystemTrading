import math
import numpy as np
import pandas as pd
from time import sleep
from datetime import datetime

import telegram
from secrets import telegram_token, telegram_chat_id
from binance import Client
from secrets import binance_api_key, binance_secret_key

class Trader(Client):
    def __init__(self):
        super().__init__(binance_api_key, binance_secret_key)
        self.telegram = telegram.Bot(token=telegram_token)

    def send_message(self, text):
        message = f"[Binance {datetime.now()}] {text}"
        self.telegram.sendMessage(chat_id=telegram_chat_id, text=message)
        print(message)

    def get_tickers(self):
        tickers = []
        for ticker in self.get_all_tickers():
            if ticker['symbol'][-4:] == "USDT":
                tickers.append(ticker['symbol'])
        return tickers

    def get_current_price(self, coin):
        orderbook = self.get_order_book(symbol=coin)
        bid = float(orderbook['bids'][0][0])    # first bid
        ask = float(orderbook['asks'][0][0])    # first ask
        return (bid+ask)/2.

    def get_current_asset(self, coin):
        if coin == "USDT":
            asset = self.get_asset_balance(asset=coin)['free']
        else:
            asset = self.get_asset_balance(asset=coin[:-4])['free']
        return float(asset)

    def get_total_balance(self):
        total_balance = self.get_current_asset("USDT")
        accounts = self.get_account()
        for info in accounts['balances']:
            if float(info['free']) != 0. and info['asset'] != "USDT":
                total_balance += float(info['free'])*self.get_current_price(f"{info['asset']}USDT")
        return total_balance

    def get_possessing_list(self):
        accounts = self.get_account()
        p_list = []
        for info in accounts['balances']:
            if float(info['free']) != 0. and info['asset'] != "USDT":
                if float(info['free'])*self.get_current_price(f"{info['asset']}USDT") > 10.:
                    p_list.append(f"{info['asset']}USDT")
        return p_list

    def buy_market_order(self, coin, amount):
        orderbook = self.get_order_book(symbol=coin)
        price = float(orderbook['asks'][0][0])
        quantity = amount/price

        info = self.get_symbol_info(coin)
        
        # set precision
        for f in info['filters']:
            if f['filterType'] == "LOT_SIZE":
                precision = float(f['stepSize'])
                break
        precision = int(round(-math.log(precision, 10), 0))
        quantity = float(round(quantity, precision))

        # check whether exceed minQty
        minQty = float(info['filters'][2]['minQty'])
        if quantity <= minQty:
            return None

        order = trader.create_order(
            symbol=coin,
            side="BUY",
            type="MARKET",
            quantity=quantity
        )
        return order

    def sell_market_order(self, coin, amount):
        # check whether exceed minQty
        info = self.get_symbol_info(coin)
        minQty = float(info['filters'][2]['minQty'])

        # set precision
        for f in info['filters']:
            if f['filterType'] == "LOT_SIZE":
                precision = float(f['stepSize'])
                break
        precision = int(round(-math.log(precision, 10), 0))
        quantity = math.floor(amount*pow(10, precision))/pow(10, precision)
        
        # check whether exceed minQty
        minQty = float(info['filters'][2]['minQty'])
        if quantity <= minQty:
            return None

        order = trader.create_order(
            symbol=coin,
            side="SELL",
            type="MARKET",
            quantity=quantity
        )
        return order

    def get_sample(self, coin):
        klines = np.array(super().get_historical_klines(coin, Client.KLINE_INTERVAL_1DAY, "220 day ago UTC"))
        sample = pd.DataFrame(klines.reshape(-1, 12), dtype=float, columns=['datetime',
                                                                            'open',
                                                                            'high',
                                                                            'low',
                                                                            'close',
                                                                            'volume',
                                                                            'close time',
                                                                            'quote asset volume, number of trades',
                                                                            'number of trades',
                                                                            'taker buy base asset volume',
                                                                            'taker buy quote asset volume',
                                                                            'ignore'])
        sample['datetime'] = pd.to_datetime(sample['datetime'], unit='ms')
        sample.set_index('datetime', inplace=True)
        sample = sample[['open', 'high', 'low', 'close', 'volume']].copy()
        return sample

if __name__ == "__main__":
    trader = Trader()
    trader.send_message("Start trading...")

    coins = trader.get_tickers()
    samples = dict()
    for coin in coins:
        # get samples
        sample = trader.get_sample(coin)
        if len(sample) != 220:
            del sample
            continue
        # check update
        assert sample.index[-1].date() == datetime.today().date()

        # preprocess
        sample['noise'] = 1.-abs(sample['close'] - sample['open'])/(sample['high'] - sample['low'])
        sample['reward'] = 1.+sample['close'].pct_change()
        sample['log_reward'] = np.log(sample['reward'])
        sample['value'] = 0.
        sample.dropna(inplace=True)
        for idx in sample.index:
            if idx == sample.index[0]:
                sample.loc[idx, 'value'] = sample.loc[idx, 'log_reward']
                continue

            log_reward = sample.loc[idx, 'log_reward']
            gamma = 1. - 0.5*sample.loc[idx, 'noise']
            sample.loc[idx, 'value'] = log_reward + gamma*(sample.shift(1).loc[idx, 'value'])
        sample = sample.iloc[-12:].copy()
        sample['momentum'] = (sample['close'] - sample.shift(10)['close'])/sample.shift(10)['close']
        sample.dropna(inplace=True)

        samples[coin] = sample.copy(); del sample
    
    # choose 21 top trading coins & 5 top value coins
    volumes = dict()
    for coin in samples.keys():
        volumes[coin] = samples[coin].iloc[-2]['close'] * samples[coin].iloc[-2]['volume']
    top21v = dict(sorted(volumes.items(), key=(lambda x: x[1]), reverse=True)[:21])
    values = dict()
    for coin in top21v.keys():
        values[coin] = samples[coin].iloc[-2]['value']
    results = dict(sorted(values.items(), key=(lambda x: x[1]), reverse=True)[:5])
    
    # filter by momentum
    buying_list = []
    for coin in results.keys():
        momentum = samples[coin].iloc[-2]['momentum']
        if momentum < 0.3:
            continue
        buying_list.append(coin)
    trader.send_message(f"buying_list {buying_list}")
    
    # Start trading
    possessing_list = trader.get_possessing_list()
    # sell the coins that currently possess and not in the buying list
    rebalance_list = []
    for ticker in possessing_list:
        i = 0
        while True:
            try:
                unit = trader.get_current_asset(ticker)
                price = trader.get_current_price(ticker)
                if (unit*price) > 5. and (ticker not in buying_list):
                    order = trader.sell_market_order(ticker, unit)
                    trader.send_message(order)
                    break
                elif (unit*price) > 5. and (ticker in buying_list):
                    rebalance_list.append(ticker)
                    break
                else:
                    break
            except Exception as e:
                trader.send_message(e)
                i += 1
                if i == 5: exit()
                sleep(0.2)
                continue
    print("stage1")

    # sort the rebalance_list in balance order
    balance_dict = {}
    for ticker in rebalance_list:
        balance_dict[ticker] = trader.get_current_asset(ticker)*trader.get_current_price(ticker)
    balance_dict = dict(sorted(balance_dict.items(), key=lambda item: item[1]))
    rebalance_list = list(balance_dict.keys())

    # Set target amount and rebalance
    target_amount = trader.get_total_balance()*0.9/5.
    trader.send_message(f"target amount: {target_amount}")

    for ticker in rebalance_list:
        i = 0
        while True:
            try:
                unit = trader.get_current_asset(ticker)
                price = trader.get_current_price(ticker)
                diff = unit*price - target_amount
                if diff > 5.:   # sell
                    unit = diff/price
                    order = trader.sell_market_order(ticker, unit)
                    trader.send_message(order)
                    buying_list.remove(ticker)
                    break
                elif diff < -5.:    # buy
                    order = trader.buy_market_order(ticker, abs(diff))
                    trader.send_message(order)
                    buying_list.remove(ticker)
                    break
                else:
                    buying_list.remove(ticker)
                    break
            except Exception as e:
                trader.send_message(e)
                i += 1
                if i == 5: exit()
                sleep(0.2)
                continue
    print("stage2")

    # now buy the all
    for ticker in buying_list:
        i = 0
        while True:
            try:
                order = trader.buy_market_order(ticker, target_amount)
                trader.send_message(order)
                break
            except Exception as e:
                trader.send_message(e)
                i += 1
                if i == 5: exit()
                sleep(0.2)
                continue

    trader.send_message("End trading")
