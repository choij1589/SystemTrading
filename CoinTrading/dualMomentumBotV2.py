#########################################################################################
#### Quant strategy based on dual momentum                                           ####
#### Author: Jin Choi                                                                ####
#### Contact: chlwls1589@snu.ac.kr                                                   ####
#########################################################################################
#### Strategy                                                                        ####
#### 1. Rebalancing everyday                                                         ####
#### 2. If Close < EMA50 for BTC or ETH, no trading                                  ####
#### 3. Select top 21 trading coins. If 10 days noise > 0.7, drop from the coin list ####
#### 4. Sort in momentum order. Select top 5 coins                                   ####
#### 5. Set leverage for each coin.                                                  ####
####    - If MOM7 > 0: +1                                                            ####
####    - If MOM20 > 0: +1                                                           ####
####    - If Close < EMA7: -1                                                        ####
####    - If Close < EMA20: -1                                                       ####
#########################################################################################
#### Back Testing result can be found at                                             ####
#### $link                                                                           ####
#########################################################################################

# TODO: make plots for traking?
from datetime import datetime
from time import sleep
from trader import Trader


def cancel_all_positions(bot):
    positions = bot.fetch_balance()['info']['positions']
    for pos in positions:
        if float(pos['positionAmt']) != 0.:
            ticker= f"{pos['symbol'][:-4]}/{pos['symbol'][-4:]}"
            amount = pos['positionAmt']
            bot.create_market_sell_order(ticker, amount, params={"type": "future"})
    bot.log("Cancelled all orders")


def is_market_timing(bot):
    btc = bot.get_sample("BTC/USDT", limit=55)
    eth = bot.get_sample("ETH/USDT", limit=55)
    btc['ema50'] = btc['close'].ewm(50).mean()
    eth['ema50'] = eth['close'].ewm(50).mean()
    
    #print(btc.iloc[-2]['close'], btc.iloc[-2]['ema50'])
    #print(eth.iloc[-2]['close'], eth.iloc[-2]['ema50'])

    if btc.iloc[-2]['close'] > btc.iloc[-2]['ema50'] and eth.iloc[-2]['close'] > eth.iloc[-2]['ema50']:
        return True
    else:
        return False


def select_top21v_coins(bot):
    volumes = dict()
    for ticker in bot.get_tickers():
        sample = bot.get_sample(ticker, limit=200)
        try:
            assert len(sample) == 200
            assert sample.index[-1].date() == datetime.today().date()
        except Exception as e:
            #print(len(sample))
            continue
        # filter out noisy coins
        sample = sample.iloc[-20:]
        sample['noise'] = 1. - abs(sample['open']-sample['close'])/(sample['high']-sample['low'])
        sample['noise'] = sample['noise'].rolling(15).mean()
        noise = sample.iloc[-2]['noise']
        if sample.iloc[-2]['noise'] > 0.7:
            print(ticker, noise)
            continue

        high, low, close = sample.iloc[-2]['high'], sample.iloc[-2]['low'], sample.iloc[-2]['close']
        volume = sample.iloc[-2]['volume']
        TP = (high+low+close)/3.
        volumes[ticker] = TP*volume
        del sample

    top21v = dict(sorted(volumes.items(), key=(lambda x: x[1]), reverse=True)[:21])
    return list(top21v.keys())

def select_top5m_coins(bot, top21v):
    values = dict()
    for ticker in top21v:
        sample = bot.get_sample(ticker, limit=25)
        try:
            assert len(sample) == 25
            assert sample.index[-1].date() == datetime.today().date()
        except Exception as e:
            continue
        sample['mom7'] = (sample['close'] - sample.shift(7)['close'])/sample.shift(7)['close']
        values[ticker] = sample.iloc[-2]['mom7']
    values = dict(sorted(values.items(), key=(lambda x: x[1]), reverse=True)[:5])
    return list(values.keys())

def get_leverage(bot, coin):
    sample = bot.get_sample(coin, limit=25)
    assert len(sample) == 25
    sample['mom7'] = sample['close']/sample.shift(7)['close']
    sample['mom20'] = sample['close']/sample.shift(20)['close']
    sample['ema7'] = sample['close'].ewm(7).mean()
    sample['ema20'] = sample['close'].ewm(20).mean()
    sample.dropna(inplace=True)

    lev = 0
    if sample.iloc[-2]['mom7'] > 0.: lev += 1
    if sample.iloc[-2]['mom20'] > 0.: lev += 1
    if sample.iloc[-2]['ema7'] > sample.iloc[-2]['close']: lev -= 1
    if sample.iloc[-2]['ema20'] > sample.iloc[-2]['close']: lev -= 1
    return lev

if __name__ == "__main__":
    bot = Trader()
    bot.log("Start trading...")
    cancel_all_positions(bot)
    sleep(60)

    if not is_market_timing(bot):
        bot.log("Not market timing, exit program...")
        exit()

    top21v = select_top21v_coins(bot)
    top5m = select_top5m_coins(bot, top21v)
    
    # start trading
    target_amount = bot.get_total_balance()/5.*0.95
    bot.log(f"target amoung: {target_amount:.3f}")
    for coin in top5m:
        lev = get_leverage(bot, coin)
        bot.log(f"leverage for {coin}: {lev}")
        if lev == 0: 
            continue

        bot.set_leverage(coin, lev)
        order = bot.buy_market_order(coin, target_amount*lev)

        bot.log(order)

    bot.log("End trading")
        
