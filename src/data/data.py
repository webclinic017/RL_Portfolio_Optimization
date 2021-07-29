from alpha_vantage.timeseries import TimeSeries

ALPHA_API_KEY = 'DKPRX8VRBRX0S9NJ'

ts = TimeSeries(key = ALPHA_API_KEY, output_format='pandas')
