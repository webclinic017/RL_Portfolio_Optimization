# %%
import argparse
import pandas as pd
from models.s3_bucket import S3Bucket
import numpy as np
from pyalgotrade.feed import csvfeed
import yfinance
from src.data.make_dataset import GetData
import pathlib
import os
import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')

# %%


# %%
s3_bucket = S3Bucket()
# data_path = '/Users/ramanganti/Desktop/ML_Trading/RL_Portfolio_Optimization/data/raw/'
# file_name = 'qqq_holdings.csv'
# s3_bucket.push_to_s3(data_path + file_name, file_name)

# %%
data = GetData()

# %%
valid_range = pd.date_range(start='2021-08-01', end='2021-09-01', freq='MS').strftime("%Y-%m-%d")
month_begin = valid_range[0]
month_end = valid_range[1]
day_range = nyse.valid_days(start_date=month_begin, end_date=month_end)[:-1].strftime("%Y-%m-%d")

for day in day_range:
    time_range = pd.date_range(day + " 09:30:00", day + " 16:00:00", freq="1min")
    df_symbol = data.get_bars(data.etf, start=day, end=day, time_range=time_range)


# %%




# %%
month_range
# %%
month_begin = month_range[0]

# %%
month_end = month_range[-1]

# %%
month_end

# %%
day_range = nyse.valid_days(start_date=month_begin, end_date=month_end)[:-1].strftime("%Y-%m-%d")

# %%
day = day_range[0]

# %%
day

# %%
time_range = pd.date_range(day + " 09:30:00", day + " 16:00:00", freq="1min")

# %%
df_symbol_reindex, df_polygon = data.get_bars(symbol="QQQ", start=day, end=day, time_range=time_range)

# %%
df_polygon.to_csv("qqq_data.csv", index_label='Date Time')

# %%
df_symbol_reindex

# %%
data = s3_bucket.load_from_s3("Data/data_{}.csv".format(month_begin))

# %%
data.loc[0]

# %%

## data = yfinance.download("SPY", start="2000-01-01", end="2019-10-04")
## data.to_csv("spy.csv")
# %%


# # %%
valid_range = pd.date_range(start='2021-08-01', end='2021-08-01', freq='MS').strftime("%Y-%m-%d")
s3_bucket = S3Bucket()
T = 10
T_predict = 1
m = 0

month_begin = valid_range[m]
print("Month: " + str(month_begin))
data = s3_bucket.load_from_s3("Data/data_{}.csv".format(month_begin))

day_indices = np.unique(data.index)
day_length = data.loc[0].shape[0]
input_size = data.loc[0].shape[1] - 3

ref_idx = np.array(range(day_length - T))

# %%
i = 0
data_day = data.loc[i]
data_day = data_day.bfill()

# %%
data_day
# %%
month_begin
# %%
df = data_day[['QQQ', 'Date Time']]
df['Date Time'] = pd.to_datetime(df['Date Time'])
df = df.set_index('Date Time')
df = df.rename(columns={'QQQ': "Close"})

# %%
df

# %%
df.to_csv("qqq_data.csv")

# # %%
# df = pd.read_csv("qqq_data.csv")
# # %%
# feed = csvfeed.Feed("Date Time", "%Y-%m-%d %H:%M:%S")

# # %%
# feed.addValuesFromCSV("qqq_data.csv")
# # %%
# for dateTime, value in feed:
#     print(dateTime, value)
# # %%

# %%
