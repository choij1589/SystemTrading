import pybithumb
import telegram
from time import sleep
from datetime import datetime

from bithumb_api import connect_key, secret_key
from telegram_api import token

# base class
class TraderBase:
    def __init__(self, connect_key, secret_key, token):
        self.bithumb = pybithumb.Bithumb(connect_key, secret_key)
        self.telegram = telegram.Bot(token=token)

    def get_coins(self):
        return self.coins

    def send_message(self, text):
        chat_id = "1759533516"
        text = f"[Bithumb {datetime.now()}] {text}"
        self.telegram.sendMessage(chat_id=chat_id, text=text)
        print(text)

    # return current value of an input coin in KRW
    def get_current_value(self, coin):
        current_value = self.bithumb.get_balance(coin)[0] * self.bithumb.get_current_price(coin)
        return current_value

    # return total balance
    def get_total_balance(self):
        total_balance = self.bithumb.get_balance("BTC")[2]    # possessing KRW
        for ticker in self.coins:
            total_balance += self.get_current_value(ticker)
            sleep(0.01)

        return total_balance

    def buy(self, coin, amount):
        orderbook = self.bithumb.get_orderbook(coin)
        asks = orderbook['asks']
        sell_price = asks[0]['price']
        unit = amount / float(sell_price)
        order = self.bithumb.buy_market_order(coin, unit)

        if type(order) == dict:
            self.send_message(f"error occured while buying {coin}!")
            self.send_message(order)
            return None

        return order

    def sell(self, coin, amount):
        unit = amount / self.bithumb.get_current_price(coin)
        order = self.bithumb.sell_market_order(coin, unit)

        if type(order) == dict:
            self.send_message(f"error occurred while selling {coin}!")
            self.send_message(order)
            return None

        return order

# Actual class for trading
class VolBreakout(TraderBase):
    def __init__(self, connect_key, secret_key, token):
        super().__init__(connect_key, secret_key, token)
        self.coins = []

    def update_possessing_coins(self):
        for ticker in self.bithumb.get_tickers():
            current_value = super().get_current_value(ticker)
            if current_value > 5000. and ticker not in self.coins:
                self.coins.append(ticker)
            elif current_value < 5000. and ticker in self.coins:
                self.coins.remove(ticker)
            else:
                continue

    def select_t12hot_coins(self):
        all_info = self.bithumb.get_current_price("ALL")
        acc_trade_info = {}
        for coin, info in all_info.items():
            acc_trade_info[coin] = all_info[coin]["acc_trade_value_24H"]
        # sort
        acc_trade_info = dict(sorted(acc_trade_info.items(), key=lambda item: float(item[1]), reverse=True))
        return list(acc_trade_info.keys())[:12]
    
    def eval_target_prices(self, target_coins):
        # target price = open + noise15*distance
        target_prices = {}
        for coin in target_coins:
            ohlc = self.bithumb.get_candlestick(coin)[-20:]
            ohlc['noise'] = 1. - abs((ohlc['close']-ohlc['open'])/(ohlc['high']-ohlc['low']))
            ohlc['noise15'] = ohlc['noise'].ewm(span=15).mean()
            last_idx = ohlc.index[-1]
            yesterday_idx = ohlc.index[-2]
            distance = ohlc.loc[yesterday_idx, 'high'] - ohlc.loc[yesterday_idx, 'low']
            target_price = ohlc.loc[last_idx, 'open'] + ohlc.loc[yesterday_idx, 'noise15']*distance
            target_prices[coin] = target_price

        return target_prices

    def sell_all(self):
        for coin in self.coins:
            value = super().get_current_value(coin)
            order = super().sell(coin, value)
            if order != None:
                super().send_message(f"selling {coin} has been done")
            else:
                raise(AttributeError)
            sleep(0.1)
        self.coins = []

    def trade(self):
        super().send_message("Start Volatility Breakout...")
        self.update_possessing_coins()
        # trade using volatility breakout
        # for *:00 ~ *:02, rest
        # for *:02 ~ *:05, update target coins
        # for *:05 ~ *:00, trade
        # for 23:40 ~ 00:00, sell all coins and rest
        buying_amount = super().get_total_balance()*0.9/5
        target_coins = self.select_t12hot_coins()
        target_prices = self.eval_target_prices(target_coins)
        
        for coin, price in target_prices.items():
            super().send_message(f"{coin}: {price}")
        for coin in self.coins:
            if coin in target_coins:
                target_coins.remove(coin)

        while True:
            hour = datetime.now().hour
            minute = datetime.now().minute

            try:
                if hour == 23 and minute > 40:  # sell all
                    self.sell_all()
                    sleep(60*20)
                elif 0 <= minute < 2:
                    sleep(60*2)
                elif 2 <= minute < 5:           # update target coins
                    if hour == 0:
                        buying_amount = super().get_total_balance()*0.9/5

                    if hour == 0 or (len(self.coins) != 5):
                        target_coins = self.select_t12hot_coins()
                        target_prices = self.eval_target_prices(target_coins)
                        super().send_message("updating target price complete")
                        for coin, price in target_prices.items():
                            super().send_message(f"{coin}: {price}")
                        for coin in self.coins:
                            if coin in target_coins:
                                target_coins.remove(coin)
                    else:
                        pass
                    sleep(60*3)
                else:
                    if len(self.coins) == 5:
                        pass
                    else:
                        for coin in target_coins:
                            current_price = self.bithumb.get_current_price(coin)
                            target_price = target_prices[coin]
                            if current_price > target_prices[coin]:
                                super().send_message(f"{coin} matched to the target price {target_price}!")
                                super().send_message(f"(target price, current price) = ({target_price}, {current_price})")
                                super().buy(coin, buying_amount)
                                self.coins.append(coin)
                                target_coins.remove(coin)
            except Exception as e:
                super().send_message(e)
                sleep(10)

if __name__ == "__main__":
    trader = VolBreakout(connect_key, secret_key, token)
    trader.trade()

