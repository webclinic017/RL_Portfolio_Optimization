# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.algorithms import mode
from src.trade_api.config import Polygon_API_KEY
import requests
import csv
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup
import time
from src.models.io_handling import pickle_out_data
import boto3
import os
import pandas_market_calendars as mcal

nyse = mcal.get_calendar('NYSE')


os.environ["AWS_DEFAULT_REGION"] = 'us-east-1'
os.environ["AWS_ACCESS_KEY_ID"] = 'AKIAZURDHJ7XTWF2Q55F'
os.environ["AWS_SECRET_ACCESS_KEY"] = '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'


class S3Bucket(object):
    def __init__(self) -> None:
        super().__init__()
        self.s3 = boto3.resource(
            service_name='s3',
            region_name='us-east-1',
            aws_access_key_id='AKIAZURDHJ7XTWF2Q55F',
            aws_secret_access_key='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'
        )
        self.s3_name = 'rganti-qqq-data'

    def push_to_s3(self, data_path, file_name):
        print("Pushing {} to {}".format(file_name, self.s3_name))
        self.s3.Bucket(self.s3_name).upload_file(Filename=data_path + file_name, Key=file_name)

    def load_from_s3(self, file_name):
        obj = self.s3.Bucket(self.s3_name).Object(file_name).get()
        df = pd.read_csv(obj['Body'], index_col=0)

        return df


class GetData(object):

    def __init__(self) -> None:
        super().__init__()
        self.etf = 'QQQ'
        self.num_holdings = 81
        self.holdings = self.get_qqq_holdings()
        self.missing_values = self.initialize_dictionary()
        self.data_path = '{}/RL_Portfolio_Optimization/data/processed/'.format(os.environ['HOME'])
        self.s3_bucket = S3Bucket()
        self.POLYGON_AGGS_URL = 'https://api.polygon.io/v2/aggs/ticker/{}/range/1/minute/{}/{}?adjusted=true&sort=asc&limit=5000&apiKey={}'
        self.session = requests.Session()
        self.month_range = pd.date_range(start='2019-09-01', end='2021-09-01', freq='MS').strftime("%Y-%m-%d")
    
    def initialize_dictionary(self):
        missing_values = {}
        missing_values[self.etf] = []

        for symbol in self.holdings:
            missing_values[symbol] = []

        return missing_values

    def get_bars(self, symbol, start, end, time_range):

        r = self.session.get(self.POLYGON_AGGS_URL.format(symbol, start, end, Polygon_API_KEY))

        df_symbol_reindex = pd.DataFrame({})
        
        if r:
            data = r.json()

            # Create a pandas dataframe from the information
            if data['queryCount'] > 1:
                df = pd.DataFrame(data['results'])
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('date', inplace=True)
                df['symbol'] = symbol

                df.drop(columns=['vw', 't', 'n'], inplace=True)
                df.rename(columns={'v': 'volume', 'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low'}, inplace=True)
                df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern').strftime("%Y-%m-%d %H:%M:%S")

                # print(df['close'].iloc[0])
                # print("{} exists".format(symbol))

                df_polygon = df['{} 09:30:00'.format(start):'{} 16:00:00'.format(end)]
                d_symbol = {'Time': pd.to_datetime(df_polygon.index), symbol: pd.to_numeric(df_polygon['close'])}
                df_symbol = pd.DataFrame(data=d_symbol).set_index('Time')
                df_symbol_reindex = df_symbol.reindex(time_range)

                missing_list = df_symbol_reindex[df_symbol_reindex[symbol].isnull()].index.tolist()
                if len(missing_list) > 0:
                    self.missing_values[symbol].append(missing_list)
                    # print(self.missing_values)
            
            else:
                msg = ('No data for symbol ' + str(symbol) + ' between {} and {}'.format(start, end))
                print(msg)
        
        return df_symbol_reindex
    
    def append_to_df_holdings(self, symbol, day, df_holdings, time_range):
        df_symbol = self.get_bars(symbol, start=day, end=day, time_range=time_range)

        if df_symbol.empty:
            df_holdings = df_symbol

        else:
            df_holdings[symbol] = df_symbol[symbol]
 
        return df_holdings

    def get_data_year_month(self):

        for m in range(len(self.month_range)-1):
            month_begin = self.month_range[m]
            month_end = self.month_range[m+1]

            day_range = nyse.valid_days(start_date=month_begin, end_date=month_end)[:-1].strftime("%Y-%m-%d")

            # Define df_holdings_month
            df_holdings_month = pd.DataFrame({})

            for day in day_range:
                print("day = {}".format(day))

                time_range = pd.date_range(day + " 09:30:00", day + " 16:00:00", freq="1min")
                date = [str(time).split()[0] for time in time_range]                
                
                df_holdings = pd.DataFrame({'Time': time_range, 'Date': date}).set_index('Time')
                df_holdings = self.append_to_df_holdings(self.etf, day, df_holdings, time_range)

                if df_holdings.empty:
                    print("No results available for trading day " + day)

                else:
                    for symbol in self.holdings:
                        df_holdings = self.append_to_df_holdings(symbol, day, df_holdings, time_range)

                        if df_holdings.empty:
                            print("No results available for symbol " + symbol)
                            break
                    
                    if not df_holdings.empty:
                        df_holdings = df_holdings.interpolate()
                        df_holdings_month = df_holdings_month.append(df_holdings)
            
            df_holdings_month['Date Time'] = df_holdings_month.index
            df_holdings_month.index = df_holdings_month['Date'].factorize()[0]

            pickle_out_data(self.missing_values, self.data_path + "timestamps_missed_{}".format(month_begin))
            file_name = 'data_{}.csv'.format(month_begin)
            df_holdings_month.to_csv(self.data_path + file_name)
            self.s3_bucket.push_to_s3(self.data_path, file_name)

            self.missing_values = self.initialize_dictionary()

    # def get_holdings(self):
    #     url = "https://etfdb.com/etf/{0}/#holdings".format(self.etf)
    #     uClient = uReq(url)
    #     page_html = uClient.read()
    #     uClient.close()
    #     page_soup = soup(page_html, "html.parser")

    #     # Use page_soup.findAll("table", {"class": "table"}) to show additional information e.g. holdings, etc.
    #     containers = page_soup.findAll("td", {"data-th": "Symbol"})
    #     ETF_holdings = [c.a.get_text() for c in containers]

    #     return ETF_holdings
    
    # def get_ndx_holdings(self):
    #     url = "https://www.slickcharts.com/nasdaq100"
    #     req = Request(url, headers={'User-Agent': 'Chrome'})
    #     webpage = urlopen(req).read()

    #     page_soup = soup(webpage, "html.parser")
    #     table = page_soup.find("div", {"class": "table-responsive"})
    #     stocks = table.find_all("a")
    #     stock_symbols = [stocks[i].get_text() for i in range(len(stocks)) if i % 2 == 1]
    #     top_holdings = stock_symbols[:self.num_holdings]
    #     top_holdings.sort()

    #     return top_holdings
    
    def get_qqq_holdings(self):
        df = pd.read_csv("../../data/raw/qqq_holdings.csv")
        symbols_list = df['Holding Ticker'].to_list()
        symbols_list = [symbol.strip() for symbol in symbols_list]
        top_holdings = symbols_list[:self.num_holdings]
        top_holdings.sort()

        return top_holdings

    # def get_data_year_month(self, symbol, year, month):
    #     CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={}&interval=1min&slice=year{}month{}&apikey={}'.format(symbol, year, month, AV_API_KEY)
        
    #     with requests.Session() as s:
    #         download = s.get(CSV_URL)
    #         decoded_content = download.content.decode('utf-8')
    #         cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    #         my_list = list(cr)

    #         # for row in my_list:
    #         #     print(row)
        
    #     df = pd.DataFrame(my_list[1:], columns=my_list[0])
        
    #     # Post-processing to get adjusted close price data from 9:30 - 16:00 for all days
    #     df = df[['time', 'close']]
        
    #     return df
    
    # def post_process_data(self, df, symbol):
    #     d_symbol = {'Time': pd.to_datetime(df['time']), symbol: pd.to_numeric(df['close'])}
    #     df_symbol = pd.DataFrame(data=d_symbol).set_index('Time')

    #     return df_symbol
    
    # def append_to_portfolio(self, df, df_portfolio, symbol, full_date_time_range):
    #     d_symbol = {'Time': pd.to_datetime(df['time']), symbol: pd.to_numeric(df['close'])}
    #     df_symbol = pd.DataFrame(data=d_symbol).set_index('Time')
    #     df_new = df_symbol.reindex(full_date_time_range)

    #     self.missing_values[symbol] = df_new[df_new[symbol].isnull()].index.tolist()

    #     df_portfolio[symbol] = df_new[symbol]

    #     return df_portfolio

    # def get_all_data_year_month(self, year, month):
        
    #     print("Symbol = " + str(self.etf))
    #     df = self.get_data_year_month(self.etf, year, month)
    #     df['date'] = [d.split(" ")[0] for d in df['time']]
    #     df['new time'] = [d.split(" ")[1] for d in df['time']]

    #     date_range = df.drop_duplicates('date')['date']
    #     full_date_time_range = [pd.date_range(d + " 09:31:00", d + " 15:59:00", freq="1min") for d in date_range[::-1]]
    #     full_date_time_range_flat = [time for sublist in full_date_time_range for time in sublist]

    #     d_portfolio = {'Time': full_date_time_range_flat}
    #     df_portfolio = pd.DataFrame(data=d_portfolio).set_index('Time')

    #     df_portfolio = self.append_to_portfolio(df, df_portfolio, self.etf, full_date_time_range_flat)

    #     for symbol in self.holdings[:10]:
    #         print("Symbol = " + str(symbol))
    #         df = self.get_data_year_month(symbol, year, month)
    #         self.append_to_portfolio(df, df_portfolio, symbol, full_date_time_range_flat)
    #         time.sleep(15)
            
    #     pickle_out_data(self.missing_values, self.data_path + "timestamps_missed_{}_{}".format(year, month))
    #     df_portfolio = df_portfolio.interpolate()

    #     file_name = 'data_{}_{}.csv'.format(year, month)
    #     df_portfolio.to_csv(self.data_path + file_name)
    #     self.s3_bucket.push_to_s3(self.data_path, file_name)

    # def get_extended_history(self):
    #     for year in range(1, 3):
    #         print("year = {}".format(year))

    #         for month in range(1, 13):
    #             print("month = {}".format(month))

    #             self.get_all_data_year_month(year, month)
    

if __name__ == "__main__":

    data = GetData()

    # print(data.holdings)
    # print(data.data_path)

    # print(len(data.holdings))

    data.get_data_year_month()

    # data.get_all_data_year_month(1, 1)

    # data.push_to_s3('../../data/processed/data_{}_{}.h5'.format(year, month))

