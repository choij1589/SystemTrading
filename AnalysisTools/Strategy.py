import numpy as np
import math
import matplotlib.dates as mdates


class Strategy:
    def __init__(self, sample, weights):
        self.book = sample[['close']].copy()
        self.book[['number']] = self.book.index.map(mdates.date2num)
        self.book['trade'] = 0
        self.weights = weights
        self.buy = 0.

    def trade(self, date, signals, another, out):
        if self.book.shift(1).loc[date, 'trade'] == 0:
            if all(signals):
                self.book.loc[date, 'trade'] = 1
                self.buy = self.book.loc[date, 'close']
            else:
                self.book.loc[date, 'trade'] = self.book.shift(
                    1).loc[date, 'trade']
        elif self.book.shift(1).loc[date, 'trade'] < len(self.weights):
            if out:
                self.book.loc[date, 'trade'] = 0
                self.buy = 0.
            elif another:
                self.book.loc[date, 'trade'] = self.book.shift(
                    1).loc[date, 'trade'] + 1
            else:
                self.book.loc[date, 'trade'] = self.book.shift(
                    1).loc[date, 'trade']
        else:
            if out:
                self.book.loc[date, 'trade'] = 0
                self.buy = 0.
            else:
                self.book.loc[date, 'trade'] = self.book.shift(
                    1).loc[date, 'trade']

    def HighAndLow(self, ohlc, ma):
        # 1. Today the ETF is above the 200-day moving average.
        # 2. Today the ETF closes below it's 5-day moving average.
        # 3. Two days ago the high and low price of the day is below the previous day's high and low.
        # 4. Yesterday the hight and low price of the day is below the previous day's.
        # 5. Today's high and low price is below yesterday's.
        # 6. Buy on the close today.
        # 7. Aggressive Version - Buy a second unit if prices close lower than your initial entry price
        #    anytime you're in the position.
        # 8. Exit on the close when the ETF closes above its 5-day simple moving average.
        for date in self.book.index:
            # No trading occurs for the first 3 days
            if date in self.book.index[0:3]:
                continue

            signals = []
            signals.append(ohlc.loc[date, 'close'] > ma.loc[date, 'MA200'])
            signals.append(ohlc.loc[date, 'close'] < ma.loc[date, 'MA5'])
            signals.append(ohlc.shift(
                2).loc[date, 'high'] < ohlc.shift(3).loc[date, 'high'])
            signals.append(ohlc.shift(
                2).loc[date, 'low'] < ohlc.shift(3).loc[date, 'low'])
            signals.append(ohlc.shift(
                1).loc[date, 'high'] < ohlc.shift(2).loc[date, 'high'])
            signals.append(ohlc.shift(
                1).loc[date, 'low'] < ohlc.shift(2).loc[date, 'low'])
            signals.append(ohlc.loc[date, 'high'] <
                           ohlc.shift(1).loc[date, 'high'])
            signals.append(ohlc.loc[date, 'low'] <
                           ohlc.shift(1).loc[date, 'low'])
            another = ohlc.loc[date, 'close'] < self.buy
            out = ohlc.loc[date, 'close'] > ma.loc[date, 'MA5']

            self.trade(date, signals, another, out)

    def RSI25(self, ohlc, ma, rsi):
        # 1. An ETF is above its 200-day moving average.
        # 2. The 4-period RSI closes under 25. Buy on the close
        # 3. Aggressive version - Buy a second unit if at anytime while you're in the position ths 4-period RSI closes under 20
        # 4. Exit when the 4-period RSI closes above 55.
        for date in self.book.index:
            # skip the first day for convenience
            if date == self.book.index[0]:
                continue

            signals = []
            signals.append(ohlc.loc[date, 'close'] > ma.loc[date, 'MA200'])
            signals.append(rsi.loc[date, 'RSI'] < .25)
            another = rsi.loc[date, 'RSI'] < .2
            out = rsi.loc[date, 'RSI'] > .55

            self.trade(date, signals, another, out)

    def R3(self, ohlc, ma, rsi):
        # 1. An ETF is above its 200-day moving average.
        # 2. The 2-period RSI drops three days in a row and the first day's drop is from below 60.
        # 3. The 2-period RSI closes under 10 today. Buy on the close
        # 3. Aggressive version - Buy a second unit if prices close lower than your initial entry price
        #    anytime you're in the position.
        # 4. Exit when the 2-period RSI closes above 70.
        for date in self.book.index:
            # skip the first day for convenience
            if date == self.book.index[0]:
                continue

            signals = []
            signals.append(ohlc.loc[date, 'close'] > ma.loc[date, 'MA200'])
            signals.append(rsi.loc[date, 'RSI'] <
                           rsi.shift(1).loc[date, 'RSI'])
            signals.append(
                rsi.shift(1).loc[date, 'RSI'] < rsi.shift(2).loc[date, 'RSI'])
            signals.append(
                rsi.shift(2).loc[date, 'RSI'] < rsi.shift(3).loc[date, 'RSI'])
            signals.append(rsi.shift(2).loc[date, 'RSI'] < .6)
            signals.append(rsi.loc[date, 'RSI'] < .1)
            another = ohlc.loc[date, 'close'] < self.buy
            out = rsi.loc[date, 'RSI'] > .7

            self.trade(date, signals, another, out)

    def PB(self, ohlc, ma, bollinger):
        # 1. An ETF is above its 200-day moving average.
        # 2. The %b must close under 0.2 for 3 days in a row. If this occurs buy the ETF on the close.
        # 3. Aggressive version - Any additional day while you're in the position, if the %b of the ETF closes again below 0.2 buy a second unit on the cloes.
        # 4. Exit when the %b closes above 0.8.
        for date in self.book.index:
            # skip the first two days
            if date in self.book.index[0:2]:
                continue

            signals = []
            signals.append(bollinger.shift(2).loc[date, 'PB'] < 0.2)
            signals.append(bollinger.shift(1).loc[date, 'PB'] < 0.2)
            signals.append(bollinger.loc[date, 'PB'] < 0.2)
            another = bollinger.loc[date, 'PB'] < 0.2
            out = bollinger.loc[date, 'PB'] > 0.8

            self.trade(date, signals, another, out)

    def MDD(self, ohlc, ma):
        # 1. An ETF is trading above its 200-day moving average.
        # 2. The ETF closes below its 5-period moving average on the entry day.
        # 3. The ETF must drop 4 out of the past 5 days. This means closing prices were lower than
        #    the day before for 4 out of the past 5 days. If this happens we buy the ETF on the close today.
        # 4. Aggressive Version - Buy a second unit if prices close lower than your initial entry price
        #    anytime you're in the position
        # 5. Our exit on the close when the ETF closes above its 5-period simple moving average
        for date in self.book.index:
            # skip the first 5 days
            if date in self.book.index[0:5]:
                continue

            signals = []
            signals.append(ohlc.loc[date, 'close'] > ma.loc[date, 'MA200'])
            signals.append(ohlc.loc[date, 'close'] < ma.loc[date, 'MA5'])
            mdd = []
            mdd.append(ohlc.loc[date, 'close'] <
                       ohlc.shift(1).loc[date, 'close'])
            mdd.append(ohlc.shift(1).loc[date, 'close']
                       < ohlc.shift(2).loc[date, 'close'])
            mdd.append(ohlc.shift(2).loc[date, 'close']
                       < ohlc.shift(3).loc[date, 'close'])
            mdd.append(ohlc.shift(3).loc[date, 'close']
                       < ohlc.shift(4).loc[date, 'close'])
            mdd.append(ohlc.shift(4).loc[date, 'close']
                       < ohlc.shift(5).loc[date, 'close'])
            signals.append(sum(mdd) >= 4)
            another = ohlc.loc[date, 'close'] < self.buy
            out = ohlc.loc[date, 'close'] > ma.loc[date, 'MA5']

            self.trade(date, signals, another, out)

    def RSI_10_6(self, ohlc, ma, rsi):
        # 1. The ETF is trading above its 200-day moving average
        # 2. We buy when the 2-period RSI of the ETF goes under 10.
        # 3. We buy a second unit if the ETF closes with an RSI reading under 6.
        # 4. We exit our position on the close when the ETF closes above its 5-period moving average.
        for date in self.book.index:
            # skip the first day for convenience
            if date == self.book.index[0]:
                continue

            signals = []
            signals.append(ohlc.loc[date, 'close'] > ma.loc[date, 'MA200'])
            signals.append(rsi.loc[date, 'RSI'] < .1)
            another = rsi.loc[date, 'RSI'] < .06
            out = ohlc.loc[date, 'close'] > ma.loc[date, 'MA5']

            self.trade(date, signals, another, out)

    def TPS(self, ohlc, ma, rsi):
        # 1. We wait for an up-trending ETF(above the 200-day) to become oversold
        #    by waiting for the 2-period RSI to close under 25 for two consecutive days.
        #    We then dip our toes into the water by only buying 10%.
        # 2. We then wait for the ETF to close lower and become more oversold another day
        #    and then commit only 20% more.
        # 3. We then wait for the ETF to become even more oversold as prices drop further
        #    and then we commit 30% more.
        # 4. We then wait for the ETF to become even more oversold another day and commit
        #    40% more. This gets us to a full position.
        # 5. We exit when our position (no matter if we have 10% or up to 100%) moves higher
        #    getting the 2-period RSI above 70.
        for date in self.book.index:
            # skip the first day for convenience
            if date == self.book.index[0]:
                continue

            signals = []
            signals.append(ohlc.loc[date, 'close'] > ma.loc[date, 'MA200'])
            signals.append(rsi.loc[date, 'RSI'] < .25)
            another = self.buy > ohlc.loc[date, 'close']
            out = rsi.loc[date, 'RSI'] > .7

            self.trade(date, signals, another, out)

    def TrendFollowing(self, ohlc, ma, bollinger, mfi):
        # 1. The ETF is trading above its 130-days expoenetial moving average.
        # 2. We buy when Percent B is above 0.8 and MFI 10 days is above 0.8
        # 3. Aggressive version - We buy another unit if prices still above EMA130 and close lower than your
        #    initial entry price
        # 4. Exit out position when Percent B is below 0.2 and MFI 10 days is below 0.2
        for date in self.book.index:
            # skip the first day for convenience
            if date == self.book.index[0]:
                continue

            signals = []
            signals.append(ohlc.loc[date, 'close'] > ma.loc[date, 'EMA130'])
            signals.append(bollinger.loc[date, 'PB'] > 0.8)
            signals.append(mfi.loc[date, 'MFI'] > 0.8)
            another = self.buy > ohlc.loc[date, 'close']
            out = []
            out.append(bollinger.loc[date, 'PB'] < 0.2)
            #out.append(mfi.loc[date, 'MFI'] < 0.2)
            out = all(out)

            self.trade(date, signals, another, out)

    def Reversal(self, ohlc, ma, bollinger, ii):
        for date in self.book.index:
            # skip the first 3 days
            if date in self.book.index[0:3]:
                continue

            signals = []
            signals.append(ohlc.loc[date, 'close'] < ma.loc[date, 'EMA60'])
            signals.append(bollinger.loc[date, 'PB'] > 0.8)
            signals.append(ii.loc[date, 'IIP14'] > 0.)

            another = []
            #another.append(ohlc.loc[date, 'close'] > ma.loc[date, 'EMA130'])
            another.append(self.buy > ohlc.loc[date, 'close'])
            another = all(another)
            out = []
            out.append(ohlc.loc[date, 'close'] > ma.loc[date, 'EMA60'])
            out.append(bollinger.loc[date, 'PB'] < 0.2)
            out.append(ii.loc[date, 'IIP14'] < 0.)
            out = all(out)

            self.trade(date, signals, another, out)

    def TripleScreen(self, ohlc, ma, oscillator):
        # 1. The ETF is trading above its 130-days exponential moving average.
        # 2. We buy when slow_d upward breakthrough 0.2
        # 3. Aggressive version - We buy another unit if prices still above EMA130 and close lower than your
        #    initial entry price
        # 4. Exit our position when the ETF is below its 130-days exponential moving average
        #    and slow_d downward breakthrough 0.8
        for date in self.book.index:
            # skip the first day for convenience
            if date == self.book.index[0]:
                continue

            signals = []
            signals.append(ohlc.loc[date, 'close'] < ma.loc[date, 'EMA130'])
            signals.append(oscillator.loc[date, 'slow_d'] < 0.2)
            signals.append(oscillator.shift(1).loc[date, 'slow_d'] > 0.2)
            another = []
            #another.append(ohlc.loc[date, 'close'] > ma.loc[date, 'EMA130'])
            another.append(self.buy > ohlc.loc[date, 'close'])
            another.append(oscillator.loc[date, 'slow_d'] < 0.2)
            another.append(oscillator.shift(1).loc[date, 'slow_d'] > 0.2)
            another = all(another)
            out = []
            out.append(ohlc.loc[date, 'close'] > ma.loc[date, 'EMA130'])
            out.append(oscillator.loc[date, 'slow_d'] > 0.8)
            out.append(oscillator.shift(1).loc[date, 'slow_d'] < 0.8)
            out = all(out)

            self.trade(date, signals, another, out)

    def MACD(self, ohlc, ma, macd):
        for date in self.book.index:
            if date in self.book.index[0:2]:
                continue

            signals = []
            signals.append(ohlc.loc[date, 'close'] < ma.loc[date, 'EMA130'])
            signals.append(macd.loc[date, 'MACDHist'] < 0)
            signals.append(macd.loc[date, 'MACDHist'] >
                           macd.shift(1).loc[date, 'MACDHist'])
            signals.append(macd.shift(1).loc[date, 'MACDHist'] < macd.shift(
                2).loc[date, 'MACDHist'])
            another = []
            another.append(self.buy > ohlc.loc[date, 'close'])
            another.append(macd.loc[date, 'MACDHist'] < 0)
            another.append(macd.loc[date, 'MACDHist'] >
                           macd.shift(1).loc[date, 'MACDHist'])
            another.append(macd.shift(1).loc[date, 'MACDHist'] < macd.shift(
                2).loc[date, 'MACDHist'])
            another = all(another)
            out = []
            out.append(ohlc.loc[date, 'close'] > ma.loc[date, 'EMA130'])
            #out.append(self.buy > ohlc.loc[date, 'close'])
            out.append(macd.loc[date, 'MACDHist'] <
                       macd.shift(1).loc[date, 'MACDHist'])
            out.append(macd.shift(1).loc[date, 'MACDHist'] > macd.shift(
                2).loc[date, 'MACDHist'])
            out = all(out)

            self.trade(date, signals, another, out)

    def evaluate(self):
        # Daily Returns
        rtn = 1.0
        self.book['rtn'] = 1.
        buy = []
        sell = 0.
        for date in self.book.index:
            if self.book.loc[date, 'trade'] == False:
                if self.book.shift(1).loc[date, 'trade'] == False or date == self.book.index[0]:
                    # not trading
                    buy = []
                else:
                    n_buy = int(self.book.shift(1).loc[date, 'trade'])
                    weight = sum(self.weights[0:n_buy])
                    sell = self.book.loc[date, 'close']
                    gain = 0.
                    for i in range(n_buy):
                        gain += (sell/buy[i]-1.)*self.weights[i]
                    print("[Long]", date, "sell stocks:", sell,
                          "gain:", round(gain*100, 2), "%")
                    self.book.loc[date, 'rtn'] = self.book.loc[date, 'close'] / \
                        self.book.shift(
                            1).loc[date, 'close']*weight + (1.-weight)
            else:
                n_buy = int(self.book.loc[date, 'trade'])
                if n_buy == self.book.shift(1).loc[date, 'trade']+1:
                    # just bought another unit
                    weight = sum(self.weights[0:n_buy-1])
                    buy.append(self.book.loc[date, 'close'])
                    print("[Long]", date, "buy", n_buy,
                          "occurred:", buy[n_buy-1])
                    self.book.loc[date, 'rtn'] = self.book.loc[date, 'close'] / \
                        self.book.shift(
                            1).loc[date, 'close']*weight + (1.-weight)
                else:
                    # update daily return
                    weight = sum(self.weights[0:n_buy])
                    self.book.loc[date, 'rtn'] = self.book.loc[date, 'close'] / \
                        self.book.shift(
                            1).loc[date, 'close']*weight + (1.-weight)

        print()
        ref = self.book.copy()
        ref['rtn'] = ref['close'].pct_change()
        ref['rtn'] = ref['rtn']+1.
        ref['acc_rtn'] = ref['rtn'].cumprod()
        historical_max = ref['close'].cummax()
        daily_drawdown = ref['close']/historical_max - 1.
        historical_dd = daily_drawdown.cummin()

        CAGR = ref['acc_rtn'].iloc[-1]**(252./len(ref.index))-1.
        MDD = historical_dd.min()
        VOL = np.std(ref['rtn'])*np.sqrt(252.)
        Sharpe = (np.mean(ref['rtn']-1.)/np.std(ref['rtn'])) * np.sqrt(252.)
        print("==== Buy And Hold ====")
        print("CAGR:", round(CAGR*100, 2), "%")
        print("MDD:", round(MDD*100, 2), "%")
        print("VOL:", round(VOL*100, 2), "%")
        print("Sharpe:", round(Sharpe*100, 2), "%")

        # Accumulated Returns
        self.book['acc_rtn'] = self.book['rtn'].cumprod()
        historical_max = self.book['acc_rtn'].cummax()
        daily_drawdown = self.book['acc_rtn']/historical_max - 1.
        historical_dd = daily_drawdown.cummin()

        CAGR = self.book['acc_rtn'].iloc[-1]**(252./len(self.book.index))-1.
        MDD = historical_dd.min()
        VOL = np.std(self.book['rtn'])*np.sqrt(252.)
        Sharpe = (np.mean(self.book['rtn']-1.) /
                  np.std(self.book['rtn'])) * np.sqrt(252.)
        print("==== Evaluate ====")
        print("CAGR:", round(CAGR*100, 2), "%")
        print("MDD:", round(MDD*100, 2), "%")
        print("VOL:", round(VOL*100, 2), "%")
        print("Sharpe:", round(Sharpe*100, 2), "%")
