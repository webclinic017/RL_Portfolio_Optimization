#%%
import requests, json

from requests.api import head

#%%
API_KEY = "PK0BM53L5KNLH09APWP9"
SECRET_KEY = "K3tdRQB9XmhDIgUNNQ9TD0beJ79jeupcU3pr394C"
BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_URL = "{}/v2/account".format(BASE_URL)
ORDERS_URL = "{}/v2/orders".format(BASE_URL)
POSITIONS_URL = "{}/v2/positions".format(BASE_URL)
HEADERS = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': SECRET_KEY}

#%%
def get_account():
    r = requests.get(ACCOUNT_URL, headers=HEADERS)

    return json.loads(r.content)

def get_positions():
    r = requests.get(POSITIONS_URL, headers=HEADERS)

    return json.loads(r.content)

def close_all_positions():
    r = requests.delete(POSITIONS_URL, headers=HEADERS)

    return json.loads(r.content)

def create_order(symbol, qty, side, type, time_in_force):
    data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": type,
        "time_in_force": time_in_force
    }

    r = requests.post(ORDERS_URL, json=data, headers=HEADERS)

    return json.loads(r.content)

def get_orders():
    r = requests.get(ORDERS_URL, headers=HEADERS)

    return json.loads(r.content)

#%%
response = close_all_positions()

print(response)

#%%
orders = get_orders()

print(orders)

#%%
positions = get_positions()

#%%
positions

#%%
a = positions[0]

#%%

#%%
# response = create_order("AAPL", 100, "buy", "market", "gtc")
# response = create_order("MSFT", 1000, "buy", "market", "gtc")
# print(response)

#%%
response = get_account()

print(response)

#%%

#%%


#%%