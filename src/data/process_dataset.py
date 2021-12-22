# %%
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


#%%
s3 = boto3.resource(
            service_name='s3',
            region_name='us-east-1',
            aws_access_key_id='AKIAZURDHJ7XTWF2Q55F',
            aws_secret_access_key='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'
        )

# %%
for bucket in s3.buckets.all():
    print(bucket.name)

# %%


# %%
# Symbols
ETF = 'XLE'
ETF_Holdings = []

class GetData(object):

    def __init__(self) -> None:
        super().__init__()
        self.etf = 'XLE'
        self.holdings = self.get_holdings()
        self.missing_values = {}
    
    def get_holdings(self):
        url = "https://etfdb.com/etf/{0}/#holdings".format(self.etf)
        uClient = uReq(url)
        page_html = uClient.read()
        uClient.close()
        page_soup = soup(page_html, "html.parser")

        # page_soup.findAll("table", {"class": "table"}) # shows additional information e.g. holdings, etc.
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

            for row in my_list:
                print(row)
        
        df = pd.DataFrame(my_list[1:], columns=my_list[0])
        
        # Post-processing to get adjusted close price data from 9:30 - 16:00 for all days
        df = df[['time', 'close']]

        # df['date'] = [d.split(" ")[0] for d in df['time']]
        # df['new time'] = [d.split(" ")[1] for d in df['time']]
        
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
            
        pickle_out_data(self.missing_values, "../../data/processed/missing_values")
        df_portfolio.to_hdf('../../data/processed/data_{}_{}_pre_interpolate.h5'.format(year, month), key='df', mode='w')
        df_portfolio = df_portfolio.interpolate()
        df_portfolio.to_hdf('../../data/processed/data_{}_{}.h5'.format(year, month), key='df', mode='w')

        # d_etf = {'Time': pd.to_datetime(df['time']), self.etf: pd.to_numeric(df['close'])}
        # df_etf = pd.DataFrame(data=d_etf).set_index('Time', inplace=True)
        # df_etf = df_etf.reindex(full_date_time_range_flat)
        # df_final[self.etf] = df_etf[self.etf]
        
        # holdings_dictionary = {}
        # date_range_length = {}
        
        # for symbol in self.holdings:
        #     df = self.get_data_year_month(symbol, year, month)
        #     holdings_dictionary[symbol] = df
            
        #     date_range = df.drop_duplicates('date')['date']
        #     date_range_length[]

        # df2 = df[(df['new time'] > '09:30:00') & (df['new time'] < '16:00:00')]
        # date_range = df2.drop_duplicates('date')['date']
        # full_date_time_range = [pd.date_range(d + " 09:31:00", d + " 15:59:00", freq="1min") for d in date_range[::-1]]
        # full_date_time_range_flat = [time for sublist in full_date_time_range for time in sublist]

        # # Define new dataframe containing only time and close price
        # d = {'Time': pd.to_datetime(df2['time']), 'Close Price': pd.to_numeric(df2['close'])}
        # df3 = pd.DataFrame(data=d)
        # df3.set_index('Time', inplace=True)
        # df_full = df3.reindex(full_date_time_range_flat)
        # df_full = df_full.interpolate()

        return df_portfolio
    

    def get_past_years(self, symbol):        
        for year in range(1, 3):
            print("year = {}".format(year))

            for month in range(1, 13):
                print("month = {}".format(month))

                if not (year == 1 and month == 1):
                    df2 = self.get_data_year_month(symbol, year, month)
                    df = df.append(df2, ignore_index=True)
        
        df.to_hdf('data_{}.h5'.format(symbol), key='df', mode='w')

# %%
data = GetData()

# %%
data.holdings

# %%
df = data.get_all_data_year_month(1, 1)

# %%
data.missing_values

# %%
df_missing = pd.read_hdf('../../data/processed/data_1_1_pre_interpolate.h5')

# %%
df_interpolate = pd.read_hdf("../../data/processed/data_1_1.h5")

# %%
df_missing['PXD']['2021-06-25 09:34:00':'2021-06-25 09:50:00']

# %%
df_interpolate['PXD']['2021-06-25 09:34:00':'2021-06-25 09:50:00']

# %%
df_missing.isnull().any()

# %%
data.missing_values

# %%
df.to_csv('data_{}_{}.csv'.format(1, 1))

# %%
df_sample = data.get_data_year_month('HES', 1, 1)

# %%
df_sample = data.post_process_data(df_sample, 'HES')

# %%
df_sample['2021-06-23 11:00:00':'2021-06-23 11:10:00']

# %%
data.missing_values

# %%
df_new = pd.read_hdf('./data_1_1.h5')

# %%
df_new['2021-06-23 11:00:00':'2021-06-23 11:10:00']
# %%
'data_{}_{}.h5'.format(1, 1)
# %%
data.missing_values

# %%
df['2021-05-05 11:25:00':'2021-05-05 11:45:00']

# %%
df_original = data.get_data_year_month('XLE', 1, 3)

# %%
df_original.columns

# %%
df_original[(df_original['time'] > '2021-05-05 11:25:00') & (df_original['time'] < '2021-05-05 11:45:00')]

# df_original['2021-05-05 11:45:00':'2021-05-05 11:25:00']

# %%
data.missing_values

# %%
my_url = "https://etfdb.com/etf/XLE/#holdings"
uClient = uReq(my_url)

# %%
# opening up connection, grabbing the page
page_html = uClient.read()
uClient.close()

# %%
page_soup = soup(page_html, "html.parser")

# %%
containers = page_soup.findAll("td", {"data-th": "Symbol"})

# %%
ETF_holdings = [c.a.get_text() for c in containers]

# %%
len(ETF_holdings)

# %%


# %%
# df = pd.DataFrame()
# df['date'] = pd.date_range("2021-04-20 09:31:00", "2021-04-20 15:59:00", freq="3min")
# df['data1'] = np.random.randint(1, 10, df.shape[0])
# df['data2'] = np.random.randint(1, 10, df.shape[0])
# df['data3'] = np.arange(len(df))


# %%
df

# %%
df['date'] = [d.split(" ")[0] for d in df['time']]
df['new time'] = [d.split(" ")[1] for d in df['time']]

# %%
df2 = df[(df['new time'] > '09:30:00') & (df['new time'] < '16:00:00')]

# %%
df2

# %%
# Scratch code below!!

date_range = df.drop_duplicates('date')['date']

# %%
full_date_time_range = [pd.date_range(d + " 09:31:00", d + " 15:59:00", freq="1min") for d in date_range[::-1]]

# %%
full_date_time_range_flat = [time for sublist in full_date_time_range for time in sublist]

# %%
df_new = df2.reindex(full_date_time_range_flat)

# %%
df_new

# %%
nan_list = df_new[df_new['Close Price'].isnull()].index.tolist()

# %%
df_new.loc[nan_list[0]:nan_list[-1]]

# %%
df_new

# %%
df_new['2021-05-05 11:25:00':'2021-05-05 11:40:00']

# %%
df['2021-05-05 11:40:00':'2021-05-05 11:25:00']

# %%
df_new.interpolate()['2021-05-05 11:25:00':'2021-05-05 11:40:00']

# %%


# %%
df_new.tail()

# %%
df.head()

# %%
df.tail()


# %%
len(df2.drop_duplicates('date')) * 389

# %%
df_resample = df.resample('T', on='time').sum()

# %%
df_resample
# %%
len(df.drop_duplicates('date')) * 389

# %%
date_range = df.drop_duplicates('date')['date']

# %%
date_range

# %%
date_range.iloc[0]

# %%
pd.date_range(date_range.iloc[0] + " 09:31:00", date_range.iloc[0] + " 15:59:00", freq="1min")

# %%
date_range.iloc[0]

# %%
df_date = df[df['date'] == date_range.iloc[0]]

# %%
df_date

# %%
df_date['time'] = pd.to_datetime(df_date['time'])

# %%
df_date.resample('1T', on='time')

# %%
df_date
# %%
# missing_timestamps = []
# for date in date_range:
#     df_date = df[df['date'] == date]

#     for time in 

# %%
len(df.drop_duplicates('date')) * 389

# %%
df2
# %%
df[:100]

# %%
df.reset_index()

# %%
df
# %%
df2 = data.get_data_year_month('XOM', 1, 1)

# %%
df3 = data.get_data_year_month('CVX', 1, 1)

# %%
df2['date'] = [d.split(" ")[0] for d in df2['time']]
df2['new time'] = [d.split(" ")[1] for d in df2['time']]

# %%
df2

# %%
df2[(df2['new time'] > '09:30:00') & (df2['new time'] < '16:00:00')]

# %%
df2[(df2['new time'] > '09:30:00') & (df2['new time'] < '16:00:00')]

# %%
df2_sample = df2[(df2['new time'] > '09:30:00') & (df2['new time'] < '16:00:00')]

# %%
df2_sample['date'].drop_duplicates()

# %%
len(df2_sample)

# %%
pd.to_numeric(df2_sample['close'])

# %%
pd.to_datetime(df2_sample['time'])

# %%
d = {'time': pd.to_datetime(df2_sample['time']), 'close': pd.to_numeric(df2_sample['close'])}

# %%
df2_sample_new = pd.DataFrame(data=d)

# %%
df2_sample_new

# %%
389 * 21
# %%
plt.plot(pd.to_datetime(df2_sample['time']), pd.to_numeric(df2_sample['close']))

# %%
df2_sample['close'] = [float(c) for c in df2['close']]
# df2_sample.plot(x = 'time', y = 'close')

# %%
df2

# %%
df3[(df3['time'] >= '2021-07-14 09:31:00') & (df3['time'] <= '2021-07-14 16:00:00')]


# %%
df2[['time', 'close']]

# %%
(60*(16 - 9.5) - 1) * len(df2_sample['date'].drop_duplicates())

# %%
df2['time']

# %%
df3.loc[0:20]

# %%
df.loc[0:20]

# %%
df2.loc[0:20]

# %%
date = '2021-07-15'

# %%
mask = (df['date'] > start_date) & (df['date'] <= end_date)

# %%
df.loc['{0} 16:00:00'.format(date):'{0} 09:30:00'.format(date)]




# %%
data = GetData()
data.get_past_years('XLE')

# %%
df = pd.read_hdf('data_XLE.h5', 'df')

# %%
df[40:80]
# %%

with requests.Session() as s:
    download = s.get(CSV_URL)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)

    for row in my_list:
        print(row)

# %%
df = pd.DataFrame(my_list[1:], columns=my_list[0])

# %%
df

# %%
my_list[-1]

# %%

# %%
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={0}&interval=1min&apikey={1}'.format(ETF, AV_API_KEY)
r = requests.get(url)
data = r.json()

# %%
data
# %%
## Data from alpha vantage
ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
data = ts.get_intraday_extended('XLE', interval='1min')

# %%
data[0]

# %%
## Data from alpaca
api = tradeapi.REST(key_id=API_KEY, secret_key=SECRET_KEY)
date = '2021-06-18'
bars = api.get_bars('TSLA', TimeFrame.Minute, start=date, end=date).df

# %%
alpaca = bars['close'].loc['{0} 08:00:00+00:00'.format(date):'{0} 20:00:00+00:00'.format(date)]

# %%
## Data from alpha vantage
ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')
data = ts.get_intraday('TSLA', interval='1min', outputsize='full')

# %%
av = data[0].loc['{0} 20:00:00'.format(date):'{0} 08:00:00'.format(date)]['4. close']

# %%
av.plot(label="av")
alpaca.plot(label="alpaca")
plt.legend()

# %%
len(alpaca)

# %%
len(av)
# %%
data[0]

# %%
close = data[0]['4. close']
# %%
close.plot()
# %%
