import argparse
import requests
import pymysql
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver

from secrets import mysql_password, mysql_bridge_ip

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--market", default=None, required=True,
                    type=str, help="market name")
parser.add_argument("--update", default=False,
                    action='store_true', help="only run for first time")
args = parser.parse_args()


class DBUpdater:
    def __init__(self, market):
        self.market = market
        self.conn = pymysql.connect(host=mysql_bridge_ip, user='root', password=mysql_password,
                                    db=self.market, charset='utf8')
        with self.conn.cursor() as curs:
            sql = """
            CREATE TABLE IF NOT EXISTS company_info (
                `symbol` VARCHAR(20),
                `company` VARCHAR(80),
                `last_update` DATE,
                PRIMARY KEY (`symbol`))
            """
            curs.execute(sql)

            sql = """ 
            CREATE TABLE IF NOT EXISTS daily_price (
                `symbol` VARCHAR(20),
                `date` DATE,
                `open` BIGINT(20),
                `high` BIGINT(20),
                `low` BIGINT(20),
                `close` BIGINT(20),
                `diff` BIGINT(20),
                `volume` BIGINT(20),
                PRIMARY KEY (`symbol`, `date`))
            """
            curs.execute(sql)
        self.conn.commit()

        self.symbols = dict()
        self.update_comp_info()

    def __del__(self):
        self.conn.close()

    def update_comp_info(self):
        sql = "SELECT * FROM company_info"
        df = pd.read_sql(sql, self.conn)
        for idx in range(len(df)):
            self.symbols[df['symbol'].values[idx]] = df['company'].values[idx]
        with self.conn.cursor() as curs:
            sql = "SELECT max(last_update) FROM company_info"
            curs.execute(sql)
            rs = curs.fetchone()
            today = datetime.today().strftime('%Y-%m-%d')

            if rs[0] == None or rs[0].strftime('%Y-%m-%d') < today:
                stocks = self.read_symbol()
                for idx in range(len(stocks)):
                    symbol = stocks.symbol.values[idx]
                    company = stocks.company.values[idx]
                    sql = f"""REPLACE INTO company_info (symbol, company, last_update) VALUES ("{symbol}", "{company}", "{today}")"""
                    curs.execute(sql)
                    self.symbols[symbol] = company
                    tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                    print(
                        f"[{tmnow}] {idx:04d}", sql)
                self.conn.commit()
                print('')

    def replace_into_db(self, df, num, symbol, company):
        with self.conn.cursor() as curs:
            for r in df.itertuples():
                sql = f"REPLACE INTO daily_price VALUES ('{symbol}', "\
                      f"'{r.date}', {r.open}, {r.high}, {r.low}, {r.close}, "\
                      f"{r.diff}, {r.volume})"
                curs.execute(sql)
            self.conn.commit()
            print('[{}] #{:04d} {} ({}) : {} rows > REPLACE INTO daily_price [OK]'.format(
                datetime.now().strftime('%Y-%m-%d %H:%M'), num+1, company, symbol, len(df)))

    def update_price(self):
        for idx, symbol in enumerate(self.symbols):
            df = self.get_dataframe(symbol, self.symbols[symbol])
            if df is None:
                continue
            self.replace_into_db(df, idx, symbol, self.symbols[symbol])
    

class KRXUpdater(DBUpdater):
    def __init__(self):
        super().__init__("KRX")

    def read_symbol(self):
        url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
        krx = pd.read_html(url, header=0)[0]
        krx = krx[['종목코드', '회사명']]
        krx = krx.rename(columns={'종목코드': 'symbol', '회사명': 'company'})
        krx.symbol = krx.symbol.map('{:06d}'.format)
        return krx

    def get_dataframe(self, symbol, company):
        df = pd.DataFrame()
        # KRX의 경우 naver에서 읽어서 dataframe을 구성
        try:
            url = f"https://finance.naver.com/item/sise_day.nhn?code={symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36'}
            with requests.get(url, headers=headers) as doc:
                html = BeautifulSoup(doc.text, "lxml")
                pgrr = html.find("td", class_="pgRR")
                if pgrr is None:
                    return None
                s = str(pgrr.a["href"]).split('=')
                lastpage = s[-1]

            if args.update:
                pages = min(int(lastpage), 1)
            else:
                pages = int(lastpage)

            for page in range(1, pages + 1):
                pg_url = '{}&page={}'.format(url, page)
                pg_url = requests.get(pg_url, headers=headers).text
                df = df.append(pd.read_html(pg_url, header=0)[0])
                tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                print('[{}] {} ({}) : {:04d}/{:04d} pages are downloading...'.
                      format(tmnow, company, symbol, page, pages), end="\r")
            df = df.rename(columns={'날짜': 'date', '종가': 'close', '전일비': 'diff',
                                    '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'})
            df['date'] = df['date'].replace('.', '-')
            df = df.dropna()
            df[['close', 'diff', 'open', 'high', 'low', 'volume']] = df[[
                'close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)
            df = df[['date', 'open', 'high',
                     'low', 'close', 'diff', 'volume']]
        except Exception as e:
            print('Exception occured :', str(e))
            return None

        return df
    
class ETFUpdater(DBUpdater):
    def __init__(self):
        # add webdriver
        opt = webdriver.ChromeOptions()
        opt.add_argument('headless')
        opt.add_argument('--no-sandbox')
        opt.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(
            r'/root/workspace/SystemTrading/MyPackages/chromedriver', options=opt)
        self.driver.implicitly_wait(3)
        super().__init__("ETF")

    def read_symbol(self):
        url = 'https://finance.naver.com/sise/etf.nhn'
        self.driver.get(url)

        # scraping table
        bs = BeautifulSoup(self.driver.page_source, 'lxml')
        self.driver.quit()
        etf_td = bs.find_all('td', class_='ctg')
        etfs = {}
        for td in etf_td:
            s = str(td.a['href']).split('=')
            symbol = s[-1]
            etfs[td.a.text] = symbol
        symbols = []
        companies = []
        for key, value in etfs.items():
            symbol = value
            company = key
            symbols.append(symbol)
            companies.append(company)
        etf = pd.DataFrame({'symbol': symbols, 'company': companies})
        return etf

    def get_dataframe(self, symbol, company):
        df = pd.DataFrame()
        # ETF의 경우 naver에서 읽어서 dataframe을 구성
        try:
            url = f"https://finance.naver.com/item/sise_day.nhn?code={symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36'}
            with requests.get(url, headers=headers) as doc:
                html = BeautifulSoup(doc.text, "lxml")
                pgrr = html.find("td", class_="pgRR")
                if pgrr is None:
                    return None
                s = str(pgrr.a["href"]).split('=')
                lastpage = s[-1]
            if args.update:
                pages = min(int(lastpage), 1)
            else:
                pages = int(lastpage)

            for page in range(1, pages + 1):
                pg_url = '{}&page={}'.format(url, page)
                pg_url = requests.get(pg_url, headers=headers).text
                df = df.append(pd.read_html(pg_url, header=0)[0])
                tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                print('[{}] {} ({}) : {:04d}/{:04d} pages are downloading...'.
                      format(tmnow, company, symbol, page, pages), end="\r")
            df = df.rename(columns={'날짜': 'date', '종가': 'close', '전일비': 'diff',
                                    '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'})
            df['date'] = df['date'].replace('.', '-')
            df = df.dropna()
            df[['close', 'diff', 'open', 'high', 'low', 'volume']] = df[[
                'close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)
            df = df[['date', 'open', 'high',
                     'low', 'close', 'diff', 'volume']]
        except Exception as e:
            print('Exception occured :', str(e))
            return None

        return df
    
if __name__ == '__main__':
    if args.market == "KRX":
        krx = KRXUpdater()
        krx.update_price()

    if args.market == "ETF":
        etf = ETFUpdater()
        etf.update_price()
