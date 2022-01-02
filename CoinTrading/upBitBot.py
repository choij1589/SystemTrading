import numpy as np
import pandas as pd
import pyupbit
import telegram
from time import sleep
from datetime import datetime

from secrets import upbit_access_key, upbit_secret_key
from secrets import telegram_token, telegram_chat_id

class Trader(pyupbit.Upbit):
    def __init__(self):
        super().__init__(upbit_access_key, upbit_secret_key)
        self.telegram = telegram.Bot(token=telegram_token)

    def send_message(self, text):
        self.telegram.sendMessage(chat_id=telegram_chat_id, text=f"[Upbit {datetime.now()}] {text}")
        print(f"[Upbit {datetime.now()}] {text}")

    def select_t20_coins(self):
        out = dict()
        tickers = pyupbit.get_tickers(fiat="KRW")
        for ticker in tickers:
            try:
                out[ticker] = pyupbit.get_ohlcv(ticker, count=3).values[-2][-1]
                sleep(0.1)
            except Exception as e:
                print(e)
                continue
        out = dict(sorted(out.items(), key=lambda item: item[1], reverse=True)[:20])
        
        return out

    def get_total_balance(self):
        tickers = pyupbit.get_tickers(fiat="KRW")
        balance = self.get_balance("KRW")
        for ticker in tickers:
            balance += self.get_balance(ticker)*pyupbit.get_current_price(ticker)
        
        return balance

    def estimate(self, tickers):
        out = dict()
        for ticker in tickers:
            sample = pyupbit.get_ohlcv(ticker, count=300)
            if len(sample.index) < 200.:
                continue
            sample['log_reward'] = np.log(1. + sample['close'].pct_change())
            sample['noise'] = 1.-abs(sample['close']-sample['open'])/(sample['high']-sample['low'])
            sample['momentum'] = (sample['close'] - sample.shift(10)['close'])/sample.shift(10)['close']
            sample.dropna(inplace=True)
            estimate = 0.
            for idx in sample.index:
                if idx == sample.index[0]:
                    estimate = sample.loc[idx, 'log_reward']
                    continue
                log_reward = sample.loc[idx, 'log_reward']
                gamma = 1. - 0.5*sample.loc[idx, 'noise']
                estimate = log_reward + gamma*estimate
            momentum = sample['momentum'].iloc[-1]
            #sample = sample[-15:]
            #sample['TP'] = (sample['high']+sample['low']+sample['close'])/3.
            #sample['PMF'] = 0.; sample['NMF'] = 0.;
            #for idx in sample.index:
            #    if idx == sample.index[0]:
            #        continue
            #    
            #    if sample.shift(1).loc[idx, 'TP'] < sample.loc[idx, 'TP']:
            #        sample.loc[idx, 'PMF'] = sample.loc[idx, 'TP']*sample.loc[idx, 'volume']
            #    else:
            #        sample.loc[idx, 'NFM'] = sample.loc[idx, 'TP']*sample.loc[idx, 'volume']
            #sample['MFI'] = sample['PMF'].rolling(10).sum()/(sample['PMF'].rolling(10).sum()+sample['NMF'].rolling(10).sum())
            #MFI = sample['MFI'].iloc[-1]
            out[ticker] = [estimate, momentum]
        out = dict(sorted(out.items(), key=lambda item: item[1][0], reverse=True)[:5])
        return out

if __name__ == "__main__":
    # trading rule
    # choose coins first from top 20 trading coins
    trader = Trader()
    trader.send_message("Start trading...")

    # make a buying list first
    tickers = trader.select_t20_coins()
    buying_list_up1 = []
    buying_list_up2 = []
    buying_list_down1 = []
    buying_list_down2 = []
    for key, value in trader.estimate(tickers).items():
        momentum = value[1]
        if momentum > 0.75: 
            buying_list_up1.append(key)
        elif momentum > 0.3: 
            buying_list_up2.append(key)
        elif momentum > -0.25:
            continue
        elif momentum > -0.55:
            buying_list_down1.append(key)
        else:
            buying_list_down2.append(key)
    buying_list = buying_list_up1+buying_list_up2+buying_list_down1+buying_list_down2
    trader.send_message(f"mom > 0.75: {buying_list_up1}")
    trader.send_message(f"0.3 < mom < 0.75: {buying_list_up2}")
    trader.send_message(f"-0.55 < mom < -0.25: {buying_list_down1}")
    trader.send_message(f"mom < -0.55: {buying_list_down2}")

    # sell the coins that currently possess and not in the buying list
    possessing_list = []
    tickers = pyupbit.get_tickers(fiat="KRW")
    for ticker in tickers:
        i = 0
        while True:
            try:
                unit = trader.get_balance(ticker)
                price = pyupbit.get_current_price(ticker)
                if (unit*price > 5000.) and (ticker not in buying_list):    
                    # not in the buying list, sell the coin
                    order = trader.sell_market_order(ticker, unit)
                    trader.send_message(order)
                    break
                elif (unit*price > 5000.) and (ticker in buying_list):       
                    # in the buying list, estimate later
                    possessing_list.append(ticker)
                    break
                else:
                    break
            except Exception as e:
                trader.send_message(e)
                i += 1
                if i == 5: exit()
                else: continue

    # sort the possessing_list in balance order
    balance_dict = {}
    for ticker in possessing_list:
        balance_dict[ticker] = trader.get_balance(ticker)*pyupbit.get_current_price(ticker)
    balance_dict = dict(sorted(balance_dict.items(), key=lambda item: item[1]))
    possessing_list = list(balance_dict.keys())

    # estimate the difference and sell/buy the coins
    target_amount = trader.get_total_balance()*0.9/5
    trader.send_message(f"target amount: {target_amount}")

    # sell first
    for ticker in possessing_list:
        i = 0
        # update target amount
        if ticker in buying_list_up1: 
            this_target_amount = target_amount
        elif ticker in buying_list_up2: 
            this_target_amount = target_amount*0.2
        elif ticker in buying_list_down1:
            this_target_amount = target_amount*0.4
        elif ticker in buying_list_down2:
            this_target_amount = target_amount
        else:
            pass
        while True:
            try:
                unit= trader.get_balance(ticker)
                price = pyupbit.get_current_price(ticker)
                difference = unit*price - this_target_amount
                if difference > 5000.:
                    selling_unit = round(difference/price, 3)
                    order = trader.sell_market_order(ticker, selling_unit)
                    trader.send_message(order)
                    buying_list.remove(ticker)
                elif difference < -5000.:
                    order = trader.buy_market_order(ticker, int(abs(difference)))
                    buying_list.remove(ticker)
                else:
                    buying_list.remove(ticker)
                break
            except Exception as e:
                trader.send_message(e)
                i += 1
                if i == 5: exit()
                else: continue
    
    print(buying_list)
    # now buy the all
    for ticker in buying_list:
        i = 0
        # update target amount
        if ticker in buying_list_up1:
            this_target_amount = target_amount
        elif ticker in buying_list_up2:
            this_target_amount = target_amount*0.2
        elif ticker in buying_list_down1:
            this_target_amount = target_amount*0.4
        elif ticker in buying_list_down2:
            this_target_amount = target_amount
        else:
            pass
        while True:
            try:
                order = trader.buy_market_order(ticker, int(this_target_amount))
                trader.send_message(order)
                break
            except Exception as e:
                trader.send_message(e)
                i += 1
                if i == 5: exit()
                else: continue

    trader.send_message("End trading")
