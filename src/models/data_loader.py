import os

import numpy as np
import pandas as pd

from src.models.io_handling import pickle_out_data
from src.models.s3_bucket import S3Bucket

s3_bucket = S3Bucket()
month_range = pd.date_range(start='2019-09-01', end='2021-08-01', freq='MS').strftime("%Y-%m-%d")
data_path = '{}/RL_Portfolio_Optimization/data/processed/'.format(os.environ['HOME'])

# Load month data into dataframe and loop through each day creating the x, y_prev, and y_gt arrays
# Concatenate arrays to create a month's array, save x, y_prev, and y_gt to a pickle file for each month
for m in range(len(month_range)):

    month_begin = month_range[m]
    print("Month: " + str(month_begin))
    data = s3_bucket.load_from_s3("data_{}.csv".format(month_begin))
    data_indices = np.unique(data.index)
    day_length = data.loc[0].shape[0]
    input_size = data.loc[0].shape[1] - 3

    T = 10
    ref_idx = np.array(range(day_length - T))

    x_month = np.zeros((len(ref_idx) * len(data_indices), T - 1, input_size))
    y_prev_month = np.zeros((len(ref_idx) * len(data_indices), T - 1))
    y_gt_month = np.array([])


    for i in data_indices:
        data_day = data.loc[i]
        X = data_day.loc[:, [x for x in data_day.columns.tolist() if x != 'QQQ' and x != 'Date' and x != 'Date Time']].to_numpy()
        y = np.array(data_day['QQQ'])

        x = np.zeros((len(ref_idx), T - 1, input_size))
        y_prev = np.zeros((len(ref_idx), T - 1))
        y_gt = y[ref_idx + T - 1]

        # format x into 3D tensor
        for bs in range(len(ref_idx)):
            x[bs, :, :] = X[ref_idx[bs]:(ref_idx[bs] + T - 1), :]
            y_prev[bs, :] = y[ref_idx[bs]: (ref_idx[bs] + T - 1)]

        x_month[len(ref_idx) * i:len(ref_idx) * (i+1)] = x
        y_prev_month[len(ref_idx) * i:len(ref_idx) * (i+1)] = y_prev
        y_gt_month = np.append(y_gt_month, y_gt)

    pickle_out_data(x_month, data_path + "x_{}".format(month_begin))
    pickle_out_data(y_prev_month, data_path + "y_prev_{}".format(month_begin))
    pickle_out_data(y_gt_month, data_path + "y_gt_{}".format(month_begin))

    s3_bucket.push_to_s3(data_path, "x_{}.pickle".format(month_begin))
    s3_bucket.push_to_s3(data_path, "y_prev_{}.pickle".format(month_begin))
    s3_bucket.push_to_s3(data_path, "y_gt_{}.pickle".format(month_begin))