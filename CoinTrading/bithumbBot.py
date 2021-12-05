import pybithumb
import telegram
from time import sleep
from datetime import datetime

from bithumb_api import connect_key, secret_key
from telegram_api import token

class Trader:
    def __init__(self, connect_key, secret_key, token):
        self.bithumb = pybithumb.Bithumb(connect_key, secret_key)
        self.telegram = telegram.Bot(token=token)
        self.coins = []

    def send_message(self, text):
        chat_id = "1759533516"
        self.telegram.sendMessage(chat_id=chat_id, text=text)
        print(text)

    def get_current_value(self, coin):
        current_value = self.bithumb.get_balance(coin)[0] * self.bithumb.get_current_price(coin)
        return current_value

    def update_possessing_coins(self):
        for ticker in self.bithumb.get_tickers():
            current_value = self.get_current_value(ticker)
            sleep(0.1)
            if current_value > 10000. and ticker not in self.coins:
                self.coins.append(ticker)
            elif current_value < 10000. and ticker in self.coins:
                 self.coins.remove(ticker)
            else:
                continue
        print(self.coins)

    def get_total_value(self):
        total_value = self.bithumb.get_balance("BTC")[2]
        for ticker in self.coins:
            total_value += self.get_current_value(ticker)
            sleep(0.01)

        return total_value

    def select_t20hot_coins(self):
        """Choose top 15 coins with largest price*volume"""
        all_info = self.bithumb.get_current_price("ALL")
        sample = {}
        for coin, info in all_info.items():
            sample[coin] = all_info[coin]["acc_trade_value_24H"]
        # sort w.r.t values
        sample = dict(sorted(sample.items(), key=lambda item: float(item[1]), reverse=True))
        return list(sample.keys())[:15]

    def choose_coins(self):
        """Choose 5 top PB voins within 20 hottest coins"""
        coins = self.select_t20hot_coins()
        sample = {}
        for coin in coins:
            # caculate PB
            ohlc = self.bithumb.get_candlestick(coin)[-25:]
            last_idx = ohlc.index[-1]
            ohlc['ma20'] = ohlc['close'].rolling(20).mean()
            ohlc['upper'] = ohlc['ma20'] + 2.*ohlc['close'].rolling(20).std()
            ohlc['lower'] = ohlc['ma20'] - 2.*ohlc['close'].rolling(20).std()
            ohlc['PB'] = (ohlc['close'] - ohlc['lower'])/(ohlc['upper'] - ohlc['lower'])
            if ohlc.loc[last_idx, 'PB'] < 0.7:
                del ohlc
                continue
            else:
                sample[coin] = ohlc.loc[last_idx, 'PB']
                del ohlc

        # sort
        sample = dict(sorted(sample.items(), key=lambda item: item[1], reverse=True))
        return list(sample.keys())[:5]

    def buy(self, coin, amount):
        orderbook = self.bithumb.get_orderbook(coin)
        asks = orderbook['asks']
        sell_price = asks[0]['price']
        unit = amount / float(sell_price)
        order = self.bithumb.buy_market_order(coin, unit)

        if type(order) == dict:
            self.send_message(f"[Bithumb {datetime.now()}] error occured while buying {coin}!")
            self.send_message(f"[Bithumb {datetime.now()}] {order}")
            return None

        return order

    def sell(self, coin, amount):
        unit = amount / self.bithumb.get_current_price(coin)
        order = self.bithumb.sell_market_order(coin, unit)
        
        if type(order) == dict:
            self.send_message(f"[Bithumb {datetime.now()}] error occurred while selling {coin}!")
            self.send_message(f"[Bithumb {datetime.now()}] {order}")
            return None

        return order

    def trade(self):
        # choose top 5 coins and calculate target amount
        target_coins = self.choose_coins()
        target_amount = self.get_total_value()*0.9/5.
        self.send_message(f"[Bithumb {datetime.now()}] Start rebalanceing...")
        self.send_message(f"[Bithumb {datetime.now()}] target coins: {target_coins}")
        self.send_message(f"[Bithumb {datetime.now()}] current balance: {self.get_total_value()}")

        order = None
        for coin in self.coins:
            # sell coins that not in the target
            if coin not in target_coins:
                value = self.get_current_value(coin)
                order = None
                for _ in range(5):
                    order = self.sell(coin, value)
                    if order != None:
                        self.send_message(f"[Bithumb {datetime.now()}] trading {coin} has been done")
                        self.coins.remove(coin)
                        sleep(0.4)
                        break
                    else:
                        sleep(0.4)
                        continue
                if order == None:
                    self.send_message(f"[Bithumb {datetime.now()}] selling {coin} has not been done")
        
        # estimate difference and buy coins
        for coin in self.coins:
            if coin in target_coins:
                value = self.get_current_value(coin)
                order = None
                if target_amount - value > 5000.:
                    for _ in range(5):
                        order = self.buy(coin, target_amount-value)
                        if order != None:
                            self.send_message(f"[Bithumb {datetime.now()}] trading {coin} has been done")
                            sleep(0.4)
                            break
                        else:
                            sleep(0.4)
                            continue
                    if order == None:
                        self.send_message(f"[Bithumb {datetime.now()}] buying {coin} has not been done")
                elif target_amount - value < -5000.:
                    for _ in range(5):
                        order = self.sell(coin, value-target_amount)
                        if order != None:
                            self.send_message(f"[Bithumb {datetime.now()}] trading {coin} has been done")
                            sleep(0.4)
                            break
                        else:
                            sleep(0.4)
                            continue
                    if order == None:
                        self.send_message(f"[Bithumb {datetime.now()}] selling {coin} has not been done")
                else:
                    continue
        sleep(10)
            
        # buy additional coins
        for coin in target_coins:
            order = None
            if coin in self.coins:
                continue
            else:
                for _ in range(5):
                    order = self.buy(coin, target_amount)
                    if order != None:
                        self.send_message(f"[Bithumb {datetime.now()}] trading {coin} has been done")
                        self.coins.append(coin)
                        sleep(0.4)
                        break
                    else:
                        sleep(0.4)
                        continue
            if order == None:
                self.send_message(f"[Bithumb {datetime.now()}] trading {coin} has not been done")

if __name__ == "__main__":
    trader = Trader(connect_key, secret_key, token)
    trader.send_message(f"[Bithumb {datetime.now()}] Start trading...")

    trader.update_possessing_coins()
    # trader.trade()

    while True:
        hour = datetime.now().hour
        minute = datetime.now().minute
        second = datetime.now().second
        
        # trade at 24:01 ~ 24:05
        if hour == 0 and 1 < minute < 5:
            try:
                trader.trade()
                sleep(300)
            except:
                continue
        # update possessing coins every hour
        elif hour != 0 and minute < 5:
            try:
                trader.update_possessing_coins()
                sleep(300)
            except:
                continue
        # check current time
        else:
            if second < 10:
                print(f"[Bithumb {datetime.now()}] I'm alive!")
            sleep(10)
