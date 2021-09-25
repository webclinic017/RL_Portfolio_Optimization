import os

from alpaca_trade_api import Stream
from alpaca_trade_api.rest import REST, TimeFrame

from alpaca_config import *

# Uses alpaca python client from https://github.com/alpacahq/alpaca-trade-api-python/
# Set environment variables that alpaca client uses
os.environ["APCA_API_KEY_ID"] = API_KEY
os.environ["APCA_API_SECRET_KEY"] = SECRET_KEY
os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
os.environ["APCA_API_DATA_URL"] = "https://data.alpaca.markets"
os.environ["APCA_RETRY_MAX"] = "3"
os.environ["APCA_RETRY_WAIT"] = "3"
os.environ["APCA_RETRY_CODES"] = "429504"

print("======== Alpaca Python Client - START ==========")
api = REST()

print("get_account")
account = api.get_account()
print(account)

print("list_assets")
assets = api.list_assets()
print(len(assets))
print(assets[0])

print("get_asset")
asset = api.get_asset(symbol='AAPL')
print(asset)

print("list_orders")
orders = api.list_orders()
print(orders)

print("submit_order")
response = api.submit_order(symbol="AAPL",
                            qty=1,
                            side="buy",
                            type="market",
                            time_in_force="gtc")
print(response)

print("cancel_all_orders")
api.cancel_all_orders()

print("list_positions")
positions = api.list_positions()
print(positions)

# Call will fail if no position exists
print("close_position")
response = api.close_position(symbol='AAPL', qty=None)
print(response)

print("close_all_positions")
response = api.close_all_positions()
print(response)

print("get_trades")
trades_df = api.get_trades("AAPL", "2021-06-08", "2021-06-08", limit=10).df
print(trades_df.head())

print("get_quotes")
quotes_df = api.get_quotes("AAPL", "2021-06-08", "2021-06-08", limit=10).df
print(quotes_df.head())

print("get_bars")
bars_df = api.get_bars("AAPL", TimeFrame.Hour, "2021-06-08", "2021-06-08", adjustment='raw').df
print(bars_df.head())

print("======== Alpaca Python Client - END ==========")

print("Alpaca live stream - START")

# Initiate Class Instance
stream = Stream(data_feed='iex')  # <- replace to SIP if you have PRO subscription


async def trade_callback(t):
    print('trade', t)


async def quote_callback(q):
    print('quote', q)


# subscribing to event
stream.subscribe_trades(trade_callback, 'AAPL')
stream.subscribe_quotes(quote_callback, 'IBM')

stream.run()

print("Alpaca live stream - END")
