import os

import numpy as np
import pandas as pd

from src.models.io_handling import pickle_out_data
from src.models.s3_bucket import S3Bucket


class ProcessData(object):

    def __init__(self, T=10, T_predict=1) -> None:
        super().__init__()

        self.s3_bucket = S3Bucket()
        self.month_range = pd.date_range(start='2019-09-01', end='2021-08-01', freq='MS').strftime("%Y-%m-%d")
        self.data_path = '{}/RL_Portfolio_Optimization/data/processed/'.format(os.environ['HOME'])
        self.T = T
        self.T_predict = T_predict
    
    def process_monthly_data(self, month_range, s3_path):
        # Load month data into dataframe and loop through each day creating the x, y_prev, and y_gt arrays
        # Concatenate arrays to create a month's array, save x, y_prev, and y_gt to a pickle file for each month

        for m in range(len(month_range)):
            month_begin = month_range[m]
            print("Month: " + str(month_begin))
            
            data = self.s3_bucket.load_from_s3("Data/data_{}.csv".format(month_begin))
            day_indices = np.unique(data.index)
            day_length = data.loc[0].shape[0]
            input_size = data.loc[0].shape[1] - 3

            ref_idx = np.array(range(day_length - self.T))

            x_month = np.zeros((len(ref_idx) * len(day_indices), self.T - 1, input_size))
            y_prev_month = np.zeros((len(ref_idx) * len(day_indices), self.T - 1))
            y_gt_month = np.zeros((len(ref_idx) * len(day_indices), self.T_predict))

            for i in day_indices:
                data_day = data.loc[i]
                X = data_day.loc[:, [x for x in data_day.columns.tolist() if x != 'QQQ' and x != 'Date' and x != 'Date Time']].to_numpy()
                y = np.array(data_day['QQQ'])

                x = np.zeros((len(ref_idx), self.T - 1, input_size))
                y_prev = np.zeros((len(ref_idx), self.T - 1))
                y_gt = np.zeros((len(ref_idx), self.T_predict))

                # format x into 3D tensor
                for bs in range(len(ref_idx)):
                    x[bs, :, :] = X[ref_idx[bs]:(ref_idx[bs] + self.T - 1), :]
                    y_prev[bs, :] = y[ref_idx[bs]:(ref_idx[bs] + self.T - 1)]
                    y_gt[bs, :] = y[ref_idx[bs] + self.T - 1: ref_idx[bs] + self.T - 1 + self.T_predict]

                x_month[len(ref_idx) * i:len(ref_idx) * (i+1)] = x
                y_prev_month[len(ref_idx) * i:len(ref_idx) * (i+1)] = y_prev
                y_gt_month[len(ref_idx) * i:len(ref_idx) * (i+1)] = y_gt

            pickle_out_data(x_month, self.data_path + "x_{}".format(month_begin))
            pickle_out_data(y_prev_month, self.data_path + "y_prev_{}".format(month_begin))
            pickle_out_data(y_gt_month, self.data_path + "y_gt_{}".format(month_begin))

            self.s3_bucket.push_to_s3(self.data_path + "x_{}.pickle".format(month_begin), s3_path + "x_{}.pickle".format(month_begin))
            self.s3_bucket.push_to_s3(self.data_path + "y_prev_{}.pickle".format(month_begin), s3_path + "y_prev_{}.pickle".format(month_begin))
            self.s3_bucket.push_to_s3(self.data_path + "y_gt_{}.pickle".format(month_begin), s3_path + "y_gt_{}.pickle".format(month_begin))

    def main(self):

        train_range = pd.date_range(start='2019-09-01', end='2021-07-01', freq='MS').strftime("%Y-%m-%d")
        self.process_monthly_data(month_range=train_range, s3_path='Train_Data/')

        valid_range = pd.date_range(start='2021-08-01', end='2021-08-01', freq='MS').strftime("%Y-%m-%d")
        self.process_monthly_data(month_range=valid_range, s3_path='Valid_Data/')


if __name__ == "__main__":

    data_process = ProcessData()
    data_process.main()