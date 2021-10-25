import pandas as pd
import matplotlib.dates as mdates


class IndexMaker:
    def __init__(self, sample):
        self.sample = sample.copy()

    def movingAverage(self):
        # return MA200, MA5, EMA130
        ma = self.sample[['close']].copy()
        ma['number'] = ma.index.map(mdates.date2num)
        ma['MA200'] = ma.close.rolling(window=200).mean()
        ma['MA5'] = ma.close.rolling(window=5).mean()
        ma['EMA10'] = ma.close.ewm(span=10).mean()
        ma['EMA20'] = ma.close.ewm(span=20).mean()
        ma['EMA60'] = ma.close.ewm(span=60).mean()
        ma['EMA130'] = ma.close.ewm(span=130).mean()
        ma = ma[['number', 'MA200', 'MA5', 'EMA10', 'EMA20', 'EMA60', 'EMA130']]
        return ma

    def ohlc(self, volume=False):
        ohlc = self.sample.copy()
        ohlc['number'] = ohlc.index.map(mdates.date2num)
        if volume:
            ohlc = ohlc[['number', 'open', 'high', 'low', 'close', 'volume']]
        else:
            ohlc = ohlc[['number', 'open', 'high', 'low', 'close']]
        return ohlc

    def bollinger(self):
        bb = self.sample[['close']].copy()
        bb['number'] = bb.index.map(mdates.date2num)
        bb['center'] = bb.close.rolling(window=20).mean()
        bb['upper'] = bb.center + 2.*bb.close.rolling(window=20).std()
        bb['lower'] = bb.center - 2.*bb.close.rolling(window=20).std()
        bb['PB'] = (bb.close - bb.lower) / (bb.upper - bb.lower)
        return bb

    def RSI(self, window=2):
        rsi = self.sample[['close']].copy()
        rsi['number'] = rsi.index.map(mdates.date2num)
        rsi['U'] = 0.
        rsi['D'] = 0.
        for date in rsi.index:
            if date == rsi.index[0]:
                continue
            if rsi.loc[date, 'close'] > rsi.shift(1).loc[date, 'close']:
                rsi.loc[date, 'U'] = rsi.loc[date, 'close'] - \
                    rsi.shift(1).loc[date, 'close']
            else:
                rsi.loc[date, 'D'] = rsi.shift(
                    1).loc[date, 'close'] - rsi.loc[date, 'close']
        rsi['AU'] = rsi['U'].rolling(window=window).mean()
        rsi['AD'] = rsi['D'].rolling(window=window).mean()
        rsi['RSI'] = rsi['AU']/(rsi['AU']+rsi['AD'])
        rsi = rsi[['number', 'AU', 'AD', 'RSI']]
        return rsi

    def MFI(self, window=10):
        mfi = self.ohlc(volume=True)
        mfi['TP'] = (mfi['high']+mfi['low']+mfi['close'])/3
        mfi['PMF'] = 0.
        mfi['NMF'] = 0.
        for date in mfi.index:
            if mfi.loc[date, 'TP'] > mfi.shift(1).loc[date, 'TP']:
                mfi.loc[date, 'PMF'] = mfi.loc[date, 'TP'] * \
                    mfi.loc[date, 'volume']
                mfi.loc[date, 'NMF'] = 0.
            else:
                mfi.loc[date, 'NMF'] = mfi.loc[date, 'TP'] * \
                    mfi.loc[date, 'volume']
                mfi.loc[date, 'PMF'] = 0.
        mfi['MFI'] = mfi.PMF.rolling(window=window).sum(
        )/(mfi.PMF.rolling(window=window).sum() + mfi.NMF.rolling(window=window).sum())
        mfi = mfi[['number', 'PMF', 'NMF', 'MFI']]
        return mfi

    def II(self, window=14):
        II = self.ohlc(volume=True)
        II['II'] = (2*II.close - II.high - II.low)/(II.high - II.low)*II.volume
        II['IIP14'] = II.II.rolling(window=window).sum(
        )/II.volume.rolling(window=window).sum()
        II = II[['number', 'II', 'IIP14']]
        return II

    def macd(self):
        macd = self.movingAverage()
        macd['MACD'] = macd['EMA60'] - macd['EMA130']
        macd['signal'] = macd['MACD'].ewm(span=45).mean()
        macd['MACDHist'] = macd['MACD'] - macd['signal']
        macd = macd[['number', 'MACD', 'signal', 'MACDHist']]
        return macd

    def oscillator(self):
        oscillator = self.ohlc()
        ndays_high = oscillator.high.rolling(window=14).max()
        ndays_low = oscillator.low.rolling(window=14).min()
        oscillator['fast_k'] = (
            oscillator.close - ndays_low)/(ndays_high - ndays_low)
        oscillator['slow_d'] = oscillator.fast_k.rolling(window=3).mean()
        oscillator = oscillator[['number', 'fast_k', 'slow_d']]
        return oscillator
