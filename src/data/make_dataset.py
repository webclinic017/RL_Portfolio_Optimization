# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.algorithms import mode
from src.trade_api.config import AV_API_KEY
import requests
import csv
import requests
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import time
from src.models.io_handling import pickle_out_data
import boto3
import os

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
        self.s3_name = 'rganti-stock-price-data'

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
        self.etf = 'XLE'
        self.holdings = self.get_holdings()
        self.missing_values = {}
        self.data_path = '{}/RL_Portfolio_Optimization/data/processed/'.format(os.environ['HOME'])
        self.s3_bucket = S3Bucket()
    
    def get_holdings(self):
        url = "https://etfdb.com/etf/{0}/#holdings".format(self.etf)
        uClient = uReq(url)
        page_html = uClient.read()
        uClient.close()
        page_soup = soup(page_html, "html.parser")

        # Use page_soup.findAll("table", {"class": "table"}) to show additional information e.g. holdings, etc.
        containers = page_soup.findAll("td", {"data-th": "Symbol"})
        ETF_holdings = [c.a.get_text() for c in containers]

        return ETF_holdings

    def get_data_year_month(self, symbol, year, month):
        CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={}&interval=1min&slice=year{}month{}&apikey={}'.format(symbol, year, month, AV_API_KEY)
        
        with requests.Session() as s:
            download = s.get(CSV_URL)
            decoded_content = download.content.decode('utf-8')
            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            my_list = list(cr)

            # for row in my_list:
            #     print(row)
        
        df = pd.DataFrame(my_list[1:], columns=my_list[0])
        
        # Post-processing to get adjusted close price data from 9:30 - 16:00 for all days
        df = df[['time', 'close']]
        
        return df
    
    def post_process_data(self, df, symbol):
        d_symbol = {'Time': pd.to_datetime(df['time']), symbol: pd.to_numeric(df['close'])}
        df_symbol = pd.DataFrame(data=d_symbol).set_index('Time')

        return df_symbol
    
    def append_to_portfolio(self, df, df_portfolio, symbol, full_date_time_range):
        d_symbol = {'Time': pd.to_datetime(df['time']), symbol: pd.to_numeric(df['close'])}
        df_symbol = pd.DataFrame(data=d_symbol).set_index('Time')
        df_new = df_symbol.reindex(full_date_time_range)

        self.missing_values[symbol] = df_new[df_new[symbol].isnull()].index.tolist()

        df_portfolio[symbol] = df_new[symbol]

        return df_portfolio

    def get_all_data_year_month(self, year, month):
        
        print("Symbol = " + str(self.etf))
        df = self.get_data_year_month(self.etf, year, month)
        df['date'] = [d.split(" ")[0] for d in df['time']]
        df['new time'] = [d.split(" ")[1] for d in df['time']]

        date_range = df.drop_duplicates('date')['date']
        full_date_time_range = [pd.date_range(d + " 09:31:00", d + " 15:59:00", freq="1min") for d in date_range[::-1]]
        full_date_time_range_flat = [time for sublist in full_date_time_range for time in sublist]

        d_portfolio = {'Time': full_date_time_range_flat}
        df_portfolio = pd.DataFrame(data=d_portfolio).set_index('Time')

        df_portfolio = self.append_to_portfolio(df, df_portfolio, self.etf, full_date_time_range_flat)

        for symbol in self.holdings:
            print("Symbol = " + str(symbol))
            df = self.get_data_year_month(symbol, year, month)
            self.append_to_portfolio(df, df_portfolio, symbol, full_date_time_range_flat)
            time.sleep(20)
            
        pickle_out_data(self.missing_values, self.data_path + "timestamps_missed_{}_{}".format(year, month))
        df_portfolio = df_portfolio.interpolate()

        file_name = 'data_{}_{}.csv'.format(year, month)
        df_portfolio.to_csv(self.data_path + file_name)
        self.s3_bucket.push_to_s3(self.data_path, file_name)

    def get_extended_history(self):
        for year in range(1, 3):
            print("year = {}".format(year))

            for month in range(1, 13):
                print("month = {}".format(month))

                self.get_all_data_year_month(year, month)
    

if __name__ == "__main__":

    data = GetData()

    data.get_extended_history()

    # data.get_all_data_year_month(1, 1)

    # data.push_to_s3('../../data/processed/data_{}_{}.h5'.format(year, month))

