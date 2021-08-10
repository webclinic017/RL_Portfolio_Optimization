import matplotlib.pyplot as plt
import numpy as np
import gym
import pandas as pd
from gym.utils import seeding
from gym import spaces
from src.data.make_dataset import GetData

stock_data = GetData()

HMAX_NORMALIZE = 100
INITIAL_BALANCE = 1000000
STOCK_DIM = len(stock_data.holdings) + 1
TRANSACTION_FEE_PERCENT = 0.000
REWARD_SCALING = 1e-4

def load_dataframe(year, month):
    df = stock_data.s3_bucket.load_from_s3("data_{}_{}.csv".format(year, month))
    df2 = df.stack().reset_index().rename(columns={'level_1': 'Tic', 0: 'Price'})
    df2.index = df2['Time'].factorize()[0]

    return df2


class StockEnv(gym.Env):

    def __init__(self, df, previous_state=[]):

        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))

        # Shape = [Current Balance] + [prices 1-n_stocks] + [owned shares 1-n_stocks]
        self.state_size = 2*STOCK_DIM + 1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.state_size,))

        self.df = df
        self.time = 0
        self.data = self.df.loc[self.time, :]

        print("Start = {}".format(self.data['Time'].iloc[0]))
        self.terminal = False

        # Initialize rewards
        self.reward = 0
        self.cost = 0
        self.df_portfolio = pd.DataFrame({'portfolio': [INITIAL_BALANCE]})

        self.trades = 0
        self.rewards_memory = []

        # Previous state
        self.previous_state = previous_state

        if len(self.previous_state) > 0:
            self.state = [self.previous_state[0]] + self.data['Price'].tolist() + \
                    list(self.previous_state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])
        else:
            self.state = [INITIAL_BALANCE] + self.data['Price'].tolist() + [0]*STOCK_DIM

        self.df_state = pd.DataFrame({'state': [self.state]})
        self.df_actions = pd.DataFrame({'actions': [np.zeros(self.action_space.shape[0])]})


    def _sell_stock(self, index, action):
        # Perform sell action based on the sign of the action
        # Index is position of stock that is being traded + 1 for Cash balance + STOCK_DIM (length of price vector)

        hold_index = index + STOCK_DIM + 1
        if self.state[hold_index] > 0:
            # Update balance
            self.state[0] += self.state[index+1]*min(abs(action), self.state[hold_index]) * \
             (1 - TRANSACTION_FEE_PERCENT)

            self.state[hold_index] -= min(abs(action), self.state[hold_index])
            self.cost += self.state[index+1]*min(abs(action), self.state[hold_index]) * \
             TRANSACTION_FEE_PERCENT
            self.trades += 1

        else:
            pass

    def _buy_stock(self, index, action):
        # Perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1]
        hold_index = index + STOCK_DIM + 1

        # Update balance
        self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                          (1 + TRANSACTION_FEE_PERCENT)

        self.state[hold_index] += min(available_amount, action)

        self.cost += self.state[index+1]*min(available_amount, action) * \
                          TRANSACTION_FEE_PERCENT
        self.trades += 1

    def compute_asset_value(self):
        total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
            self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

        return total_asset

    def compute_sortino_rewards(self, total_assets):
        self.df_portfolio = self.df_portfolio.append(pd.DataFrame({'portfolio': [total_assets]}), ignore_index=True)
        df_percent = self.df_portfolio.pct_change(1)
        std_dev_neg = np.nan_to_num(df_percent[df_percent['portfolio'] < 0].std()[0])

        if std_dev_neg > 0:
            sortino_ratio = df_percent.iloc[-1] / std_dev_neg
        else:
            sortino_ratio = df_percent.iloc[-1]

        return sortino_ratio

    def compute_log_optimal_rewards(self):
        pass

    def step(self, actions):
        
        self.terminal = self.time >= len(self.df.index.unique()) - 1

        if self.terminal:
            print("End = {}".format(self.data['Time'].iloc[-1]))
            print("State = " + str(self.state))

            # self.df_portfolio.to_csv("results/returns_{}_{}.csv".format(self.year, self.month))
            # self.df_state.to_csv("results/state_{}_{}.csv".format(self.year, self.month))
            # self.df_actions.to_csv("results/actions_{}_{}.csv".format(self.year, self.month))

            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * HMAX_NORMALIZE
            # begin_total_asset = self.compute_asset_value()

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]
            begin_total_asset = self.compute_asset_value()

            for index in sell_index:
                # Loop through all sell_indices and make sell action for each index
                # print('Take sell action: {}'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # Loop through all buy_indices and make buy action for each index
                # print('Take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            # Increment next time-step
            self.time += 1

            # Load next time-step data
            self.data = self.df.loc[self.time, :]
            self.state = [self.state[0]] + self.data['Price'].tolist() + \
                         list(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])
            
            # Append to state and actions dataframes
            self.df_state = self.df_state.append(pd.DataFrame({'state': [self.state]}), ignore_index=True)
            self.df_actions = self.df_actions.append(pd.DataFrame({'actions': [actions]}), ignore_index=True)

            end_total_asset = self.compute_asset_value()
            self.reward = end_total_asset - begin_total_asset

            self.reward = self.reward*REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        print("Environment is reset.")

        self.time = 0
        self.data = self.df.loc[self.time, :]

        print("Start = {}".format(self.data['Time'].iloc[0]))
        self.terminal = False

        # Initialize rewards
        self.reward = 0
        self.cost = 0

        self.trades = 0
        self.rewards_memory = []

        if len(self.previous_state) > 0:
            self.state = [self.previous_state[0]] + self.data['Price'].tolist() + \
                    list(self.previous_state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])
        else:
            self.state = [INITIAL_BALANCE] + self.data['Price'].tolist() + [0]*STOCK_DIM
        
        print("Initial State = {}".format(self.state))

        return self.state



class StockEnvContinual(gym.Env):

    def __init__(self, dates):

        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))

        # Shape = [Current Balance] + [prices 1-n_stocks] + [owned shares 1-n_stocks]
        self.state_size = 2*STOCK_DIM + 1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.state_size,))

        # Training Dates
        self.dates = dates
        self.date_count = 0
        self.year, self.month = self.dates[self.date_count]

        print("Year = {}, Month = {}".format(self.year, self.month))
        self.df = load_dataframe(self.year, self.month)
        self.time = 0

        self.data = self.df.loc[self.time, :]
        print("Start = {}".format(self.data['Time'].iloc[0]))

        self.df_terminal = False
        self.terminal = False

        # Initialize rewards
        self.reward = 0
        self.cost = 0
        self.df_portfolio = pd.DataFrame({'portfolio': [INITIAL_BALANCE]})

        self.trades = 0
        self.rewards_memory = []

        # Initial state space
        self.state = [INITIAL_BALANCE] + self.data['Price'].tolist() + [0]*STOCK_DIM
        self.df_state = pd.DataFrame({'state': [self.state]})
        self.df_actions = pd.DataFrame({'actions': [np.zeros(self.action_space.shape[0])]})


    def _sell_stock(self, index, action):
        # Perform sell action based on the sign of the action
        # Index is position of stock that is being traded + 1 for Cash balance + STOCK_DIM (length of price vector)

        hold_index = index + STOCK_DIM + 1
        if self.state[hold_index] > 0:
            # Update balance
            self.state[0] += self.state[index+1]*min(abs(action), self.state[hold_index]) * \
             (1 - TRANSACTION_FEE_PERCENT)

            self.state[hold_index] -= min(abs(action), self.state[hold_index])
            self.cost += self.state[index+1]*min(abs(action), self.state[hold_index]) * \
             TRANSACTION_FEE_PERCENT
            self.trades += 1

        else:
            pass

    def _buy_stock(self, index, action):
        # Perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1]
        hold_index = index + STOCK_DIM + 1

        # Update balance
        self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                          (1 + TRANSACTION_FEE_PERCENT)

        self.state[hold_index] += min(available_amount, action)

        self.cost += self.state[index+1]*min(available_amount, action) * \
                          TRANSACTION_FEE_PERCENT
        self.trades += 1

    def compute_asset_value(self):
        total_asset = self.state[0] + sum(np.array(self.state[1:(STOCK_DIM + 1)]) * np.array(
            self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]))

        return total_asset

    def compute_sortino_rewards(self, total_assets):
        self.df_portfolio = self.df_portfolio.append(pd.DataFrame({'portfolio': [total_assets]}), ignore_index=True)
        df_percent = self.df_portfolio.pct_change(1)
        std_dev_neg = np.nan_to_num(df_percent[df_percent['portfolio'] < 0].std()[0])

        if std_dev_neg > 0:
            sortino_ratio = df_percent.iloc[-1] / std_dev_neg
        else:
            sortino_ratio = df_percent.iloc[-1]

        return sortino_ratio

    def compute_log_optimal_rewards(self):
        pass

    def step(self, actions):
        
        self.df_terminal = self.time >= len(self.df.index.unique()) - 1

        if self.df_terminal:
            print("End = {}".format(self.data['Time'].iloc[-1]))
            print("State = " + str(self.state))
            self.df_portfolio.to_csv("results/returns_{}_{}.csv".format(self.year, self.month))
            self.df_state.to_csv("results/state_{}_{}.csv".format(self.year, self.month))
            self.df_actions.to_csv("results/actions_{}_{}.csv".format(self.year, self.month))

        if self.df_terminal and self.date_count >= len(self.dates) - 1:
            self.terminal = True

            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * HMAX_NORMALIZE
            # begin_total_asset = self.compute_asset_value()

            begin_total_asset = self.compute_asset_value()
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # Loop through all sell_indices and make sell action for each index
                # print('Take sell action: {}'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # Loop through all buy_indices and make buy action for each index
                # print('Take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            if self.df_terminal:
                self.date_count += 1
                self.year, self.month = self.dates[self.date_count]

                self.df = load_dataframe(self.year, self.month)
                self.time = 0
                self.df_terminal = False

                print("Year = {}, Month = {}".format(self.year, self.month))
                print("Start Next = {}".format(self.df['Time'].iloc[0]))

            else:
                self.time += 1

            # Load next time-step data
            self.data = self.df.loc[self.time, :]
            self.state = [self.state[0]] + self.data['Price'].tolist() + \
                         list(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])
            
            # Append to state and actions dataframes
            self.df_state = self.df_state.append(pd.DataFrame({'state': [self.state]}), ignore_index=True)
            self.df_actions = self.df_actions.append(pd.DataFrame({'actions': [actions]}), ignore_index=True)

            end_total_asset = self.compute_asset_value()
            self.df_portfolio = self.df_portfolio.append(pd.DataFrame({'portfolio': [end_total_asset]}), ignore_index=True)

            self.reward = end_total_asset - begin_total_asset
            self.reward = self.reward*REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        print("Environment is reset.")

        # Training Dates
        self.date_count = 0
        self.year, self.month = self.dates[self.date_count]

        print("Year = {}, Month = {}".format(self.year, self.month))
        self.df = load_dataframe(self.year, self.month)
        self.time = 0

        self.data = self.df.loc[self.time, :]
        print("Start = {}".format(self.data['Time'].iloc[0]))

        self.df_terminal = False
        self.terminal = False

        # Initialize rewards
        self.reward = 0
        self.cost = 0

        self.trades = 0
        self.rewards_memory = []

        # Initial state space
        self.state = [INITIAL_BALANCE] + self.data['Price'].tolist() + [0]*STOCK_DIM

        print("Initial State = {}".format(self.state))

        return self.state










