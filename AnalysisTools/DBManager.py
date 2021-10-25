import pandas as pd
import pymysql
from datetime import date, datetime, timedelta
import re
import matplotlib.dates as mdates
from secrets import mysql_password, mysql_bridge_ip

class MarketDB:
    def __init__(self, market):
        self.conn = pymysql.connect(host=mysql_bridge_ip, user='root', password=mysql_password,
                                    db=market, charset='utf8')
        self.symbols = {}
        self.get_comp_info()

    def __del__(self):
        self.conn.close()

    def get_comp_info(self):
        sql = "SELECT * FROM company_info"
        market = pd.read_sql(sql, self.conn)
        out = dict()
        for idx in range(len(market)):
            self.symbols[market['symbol'].values[idx]
                         ] = market['company'].values[idx]
        return self.symbols

    def get_daily_price(self, symbol, start_date=None, end_date=None):
        if (start_date is None):
            one_year_ago = datetime.today() - timedelta(days=365)
            start_date = one_year_ago.strftime('%Y-%m-%d')
            print("start_date is initialized to '{}'".format(start_date))
        else:
            start_lst = re.split('\D+', start_date)
            if (start_lst[0] == ''):
                start_year = int(start_lst[0])
                start_month = int(start_lst[1])
                start_day = int(start_lst[2])
                if start_year < 1900 or start_year > 2200:
                    print(f"ValueError: start_year({start_year:d}) is wrong.")
                    return
                if start_month < 1 or start_month > 12:
                    print(
                        f"ValueError: start_month({start_month:d}) is wrong.")
                    return
                if start_day < 1 or start_day > 31:
                    print(f"ValueError: start_day({start_day:d}) is wrong.")
                    return
                start_date = f"{start_year:04d}-{start_month:02d}-{start_day:02d}"

        if (end_date is None):
            end_date = datetime.today().strftime('%Y-%m-%d')
            print("end_date is initialized to '{}'".format(end_date))
        else:
            end_lst = re.split('\D+', end_date)
            if end_lst[0] == '':
                end_lst = end_lst[1:]
                end_year = int(end_lst[0])
                end_month = int(end_lst[1])
                end_day = int(end_lst[2])
                if end_year < 1800 or end_year > 2200:
                    print(f"ValueError: end_year({end_year:d}) is wrong.")
                    return
                if end_month < 1 or end_month > 12:
                    print(f"ValueError: end_month({end_month:d}) is wrong.")
                    return
                if end_day < 1 or end_day > 31:
                    print(f"ValueError: end_day({end_day:d}) is wrong.")
                    return
                end_date = f"{end_year:04d}-{end_month:02d}-{end_day:02d}"

        symbols_keys = list(self.symbols.keys())
        symbols_values = list(self.symbols.values())
        if symbol in symbols_keys:
            pass
        elif symbol in symbols_values:
            idx = symbols_values.index(symbol)
            symbol = symbols_keys[idx]
        else:
            print("ValueError: Code({}) doesn't exist.".format(symbol))

        sql = f"SELECT * FROM daily_price WHERE symbol = '{symbol}' and date >= '{start_date}' and date <= '{end_date}'"
        df = pd.read_sql(sql, self.conn)
        df.index = df['date']
        return df
