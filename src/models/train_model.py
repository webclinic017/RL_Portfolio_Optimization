import argparse

import pandas as pd
import numpy as np
import time
from src.models.env_train import StockEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO


def train_PPO(env_train, model_name, output_dir, timesteps=50000):
    """PPO model"""
    start = time.time()
    # model = PPO('MlpPolicy', env_train, ent_coef=0.005, nminibatches=8)
    model = PPO("MlpPolicy", env_train, verbose=1)

    print('PPO start training.')
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save("{}/{}".format(output_dir, model_name))
    print('Training time (PPO): ', (end - start) / 60, ' minutes')

    return model


class Model(object):

    def __init__(self) -> None:
        super().__init__()
        self.training_dates = [(2, i) for i in range(1, 13)]

    def main(self, timesteps=20000):
        env_train = DummyVecEnv([lambda: StockEnv(self.training_dates)])
        model = train_PPO(env_train, "PPO", "trained_models", timesteps=timesteps)

        # trading_dates = [(1, i) for i in range(1, 13)]
        #
        # for date in trading_dates:
        #     env_train = DummyVecEnv([lambda: StockEnv(self.training_dates)])
        #     env_val = DummyVecEnv([lambda: StockEnv(date)])


if __name__ == "__main__":
    # extract hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=20000, help='Number of timesteps to run (default: 20000)')
    args = parser.parse_args()

    train_model = Model()

    print("Running model training.")
    train_model.main(timesteps=args.timesteps)
