import os, shutil
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick2_ochl, volume_overlay
from datetime import datetime, timedelta
from DBManager import MarketDB

base_dir = os.get_env("BASE_DIR")

# TODO: make two classes, one for stocks and one for coins
# TODO: make all images at one time is depreciated!
# TODO: make a function from dataframe to candlestick image

class CandleStickManager():
    def __init__(self, market, stock, start_date, end_date):
        # make cushion data
        mk = MarketDB(market)
        self.stock = stock
        self.symbol = -1
        for symbol, company in mk.get_comp_info().items():
            if self.stock == company:
                self.symbol = symbol

        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')

        cushion_date = self.start_date - timedelta(days=50)
        cushion_date = cushion_date.strftime("%Y-%m-%d")
        self.cushion = mk.get_daily_price(self.stock, cushion_date, end_date)

        print(f"==== INFO: Cushion data is initailized")
        print(f"==== from {start_date} to {end_date}")
        print(f"==== Maket: {market}")
        print(f"==== Stock: {self.symbol} {self.stock}\n")

    def ohlc_to_candlestick(self, days, dataset_type, use_volume=False, incl=False, overwrite=False):
        # make directories
        stock = self.stock.replace(' ', '_')

        # inclusive images are needed to avoid overfitting
        # since there are too small number of images for each stock
        if incl:
            path = f"{base_dir}/Datasets/CandleSticks/Incl"
        else:
            path = f"{base_dir}/Datasets/CandleSticks/{stock}"
        if use_volume:
            print("=== INFO: Converting ohlcv to candlesticks...")
            path += f"/ohlcv/days_{days}/{dataset_type}"
        else:
            print("==== INFO: Converting ohlc to candlesticks...")
            path += "/ohlc/days_{days}/{dataset_type}"

        
        if os.path.exists(path):
            if overwrite:
                shutil.rmtree(path)
                os.makedirs(path)
                os.makedirs(f"{path}/up")
                os.makedirs(f"{path}/down")
            else:
                pass
        else:
            os.makedirs(path)
            os.makedirs(f"{path}/up")
            os.makedirs(f"{path}/down")
        
        # now make images and store to dataset
        sample = self.cushion.copy()
        sample['number'] = sample['date'].map(mdates.date2num)

        plt.style.use('dark_background')
        
        # make labels
        # Labels are based on the last n days candlesticks, which makes more easy to deal with index
        # However, it could be confused in learning step
        # if raise tomorrow -> up, else -> down
        idx = 0
        for date in sample.index:
            if date < self.start_date.date():
                idx += 1
                continue

            if date != sample.index[idx]:
                print("==== Warning: {} and {} is not equal!".format(date, sample.index[idx]))

            label = ""
            if sample.loc[date, 'close'] > sample.shift(1).loc[date, 'close']:
                label = "up"
            else:
                label = "down"

            ohlc = sample.iloc[idx-days:idx, :]
            idx += 1
            dimension = 512
            my_dpi = 96
            fig = plt.figure(figsize=(dimension/my_dpi, dimension/my_dpi), dpi=my_dpi)
            ax1 = fig.add_subplot(1, 1, 1)
            candlestick2_ochl(ax1, ohlc['open'], ohlc['close'], ohlc['high'], ohlc['low'],
                                width=1, colorup='#77d879', colordown='#db3f3f')
            ax1.grid(False)
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.xaxis.set_visible(False)
            ax1.yaxis.set_visible(False)
            ax1.axis('off')

            if use_volume:
                ax2 = ax1.twinx()
                bc = volume_overlay(ax2, ohlc['open'], ohlc['close'], ohlc['volume'],
                                    width=1, colorup='#77d879', colordown='#db3f3f', alpha=0.5)
                ax2.add_collection(bc)
                ax2.grid(False)
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
                ax2.xaxis.set_visible(False)
                ax2.yaxis.set_visible(False)
                ax2.axis('off')

            # save figure
            if incl:
                pngfile = os.path.join(path, label, self.symbol + "_" + date.strftime("%Y-%m-%d") + ".png")
            else:
                pngfile = os.path.join(path, label, date.strftime("%Y-%m-%d")+".png")
            fig.savefig(pngfile, pad_inches=0, transparent=False)
            plt.close()
        print("==== INFO: Conversion for {} finished".format(self.stock))

