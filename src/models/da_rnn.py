import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from s3_bucket import S3Bucket
import os


def read_data(input_path, debug=True):
    """
    Read nasdaq stocks data.

    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.

    """
    df = pd.read_csv(input_path, nrows=250 if debug else None)
    X = df.loc[:, [x for x in df.columns.tolist() if x != 'NDX']].to_numpy()
    y = np.array(df.NDX)

    return X, y


class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, T,
                 input_size,
                 encoder_num_hidden,
                 parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T - 1,
            out_features=1
        )

    def forward(self, X):
        """forward.

        Args:
            X: input data

        """
        # Initialization of X_tilde (transformed x_input) 
        # with dimensions of batch_size x lookback time x num_driving_series
        X_tilde = Variable(X.data.new(
            X.size(0), self.T - 1, self.input_size).zero_())
        # Initialization of h (X_encoded) 
        # with dimensions of batch_size x loockback time x num_hidden_units
        X_encoded = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # h_n, s_n: initial states with dimention hidden_size
        # Initialize h_{0} and s_{0} 
        h_n = self._init_states(X) # 1 x batch_size x num_hidden_units
        s_n = self._init_states(X) # 1 x batch_size x num_hidden_units

        # t here denotes the subscript for Eqs (2 - 11)
        for t in range(self.T - 1):
            # batch_size * input_size * (2 * hidden_size + T - 1)
            # Eq (8): Need to concatenate h_{t-1}, s_{t-1}, and full x
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

            # Eq (8): Compute e_{t} with attention layer
            # Reshaping array from batch_size x input_size x (2 * hidden_size + T - 1)
            # to (batch_size*input_size) x (2 * hidden_size + T - 1) with view function
            # returns (batch_size*input_size) x 1
            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.T - 1))

            # get weights by softmax
            # Eq (9): Computing softmax of energy units to get alpha_{t}
            # Reshaping array from (batch_size*input_size) x 1 to batch_size x input_size with view
            alpha = F.softmax(x.view(-1, self.input_size), dim=1)

            # get new input for LSTM
            # Eq (10): compute x_tilde_{t}
            x_tilde = torch.mul(alpha, X[:, t, :])

            # Fix the warning about non-contiguous memory
            # https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.encoder_lstm.flatten_parameters()

            # encoder LSTM
            # Eq (11): feeding x_tilde_{t} into LSTM with h_{t-1} and s_{t-1}
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))

            # Computing h_{t} and s_{t} as per Eq (11)
            h_n = final_state[0]
            s_n = final_state[1]

            # Storing computed values in X_tilde_t and X_encoded_t to pass to decoder
            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n

        return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden +
                      encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1)
        )

        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=decoder_num_hidden
        )
        
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)

        self.fc.weight.data.normal_()

    def forward(self, X_encoded, y_prev, d_n, c_n, initial=True):
        """forward."""
        if initial:
            # print("Initial prediction from history.")
            # # Initializing hidden state d_t=0 (d_n) with dimensions of h (x_encoded)
            # d_n = self._init_states(X_encoded)
            # # Initializing cell state s_t=0 (c_n) with dimensions of h (x_encoded)
            # c_n = self._init_states(X_encoded)

            # # t here denotes the subscript for Eqs (12 - 17)
            # # Cannot feed lstm cell d_{t-1} straight into d_{t},
            # # Need to combine lstm hidden cell d_{t-1} with temporal attention weights beta multiplied by h
            # # Must do this for each time step of RNN
            for t in range(self.T - 1):
                
                # Equation 12 in the paper. To compute l^{i}_{t},
                # where i is time index of encoded state (X_encoded)
                # need to feed in hidden and cell states
                # at t-1: d_{t-1}, s_{t-1} (d_n, c_n) with encoded state h (X_encoded)
                # Computing all beta^{i} values at the same time
                x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                            c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                            X_encoded), dim=2)

                # Computing Beta at time t, using the concatenation of hidden states, cell states
                # and encoded states
                # self.attn_layer computes l^{i}_{t} in Eq. (12)
                # softmax computes beta^{i}_{t} in Eq. (13)
                beta = F.softmax(self.attn_layer(
                    x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1), dim=1)

                # Eqn. 14: compute context vector at time t
                # batch_size * encoder_hidden_size
                # c_{t-1} is computed here via Eq. (14)
                context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]

                if t < self.T - 1:
                    # batch_size * 1
                    # Eqn. 15: Feed c_{t-1} (context) with y_{t-1} (y_previous) into Linear layer
                    # context weighted output: y_tilde
                    # dim = encoder_dimensions + 1 (y_previous)
                    y_tilde = self.fc(
                        torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))

                    # Eqn. 16 as well as 17-21: LSTM equations
                    self.lstm_layer.flatten_parameters()
                    # Feed in d_{t-1} (d_n), s_{t-1} (c_n) and context-weighted y_{t-1} into LSTM to get d_{t}
                    _, final_states = self.lstm_layer(
                        y_tilde.unsqueeze(0), (d_n, c_n))

                    # This computes d_{t} Eq 21 of LSTM
                    d_n = final_states[0]  # 1 * batch_size * decoder_num_hidden
                    c_n = final_states[1]  # 1 * batch_size * decoder_num_hidden

            # Eqn. 22: final output
            # dn[0] = d_n.squeeze(0) to get batch_size x decoder_dim
            # context is batch_size x encoder_dim
            y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))
        
        else:
            # Predicting future values using y_pred, d_n, c_n
            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                        c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                        X_encoded), dim=2)

            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1), dim=1)

            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]

            y_tilde = self.fc(torch.cat((context, y_prev[:, 0].unsqueeze(1)), dim=1))

            self.lstm_layer.flatten_parameters()
            _, final_states = self.lstm_layer(y_tilde.unsqueeze(0), (d_n, c_n))

            d_n = final_states[0]
            c_n = final_states[1]

            y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))

        return y_pred, d_n, c_n

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())


class StockDataset(Dataset):

    def __init__(self, x, y_prev, y_gt):
        super().__init__()
        self.x = x
        self.y_prev = y_prev
        self.y_gt = y_gt

        self.n_samples = y_gt.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y_prev[index], self.y_gt[index]

    def __len__(self):
        return self.n_samples


class DA_RNN(nn.Module):
    """Dual-Stage Attention-Based Recurrent Neural Network."""

    def __init__(self, X, y, T, T_predict,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 parallel=False,
                 sagemaker=True,
                 old_data=True):

        """initialization."""
        super(DA_RNN, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T
        self.T_predict = T_predict
        self.old_data = old_data

        self.s3_bucket = S3Bucket()
        
        if sagemaker:
            self.output_dir = os.environ['SM_OUTPUT_DATA_DIR'] + "/"
        else:
            self.output_dir = "./"

        self.X = X
        self.y = y

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        # Old Data Set
        if self.old_data:
            self.train_timesteps = int(self.X.shape[0] * 0.7)
            self.valid_timesteps = self.X.shape[0] - self.train_timesteps
            self.y_mean = np.mean(self.y[:self.train_timesteps])

            self.y = self.y - self.y_mean
            self.input_size = self.X.shape[1]

            # Training Set
            x_train, y_prev_train, y_gt_train = self.pre_process_old_data()
            self.train_dataset = StockDataset(x_train, y_prev_train, y_gt_train)
            self.train_samples = len(self.train_dataset)
            self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=False)

            # Validation Set
            x_valid, y_prev_valid, y_gt_valid = self.pre_process_old_data(train=False)
            self.valid_dataset = StockDataset(x_valid, y_prev_valid, y_gt_valid)
            self.valid_samples = len(self.valid_dataset)
            self.valid_dataloader = DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=False)

        else:
            # Training Set
            print("Pre-process training data: ")
            train_range = pd.date_range(start='2020-08-01', end='2021-02-01', freq='MS').strftime("%Y-%m-%d")
            self.y_train = self.compute_y_total(train_range) 
            self.y_mean = np.mean(self.y_train)

            x_train, y_prev_train, y_gt_train = self.pre_process_data(train_range, self.y_mean)

            self.x_train = x_train
            self.y_prev_train = y_prev_train
            self.y_gt_train = y_gt_train

            self.train_dataset = StockDataset(x_train, y_prev_train, y_gt_train)
            self.train_samples = len(self.train_dataset)
            self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=False)

            print("Loaded training data loader.\n")

            # Validation Set
            print("Pre-process validation data: ")
            valid_range = pd.date_range(start='2021-02-01', end='2021-03-01', freq='MS').strftime("%Y-%m-%d")

            x_valid, y_prev_valid, y_gt_valid = self.pre_process_data(valid_range, self.y_mean)

            self.x_valid = x_valid
            self.y_prev_valid = y_prev_valid
            self.y_gt_valid = y_gt_valid

            valid_timesteps = int(x_valid.shape[0] * 0.5)

            self.valid_dataset = StockDataset(x_valid[:valid_timesteps], y_prev_valid[:valid_timesteps], y_gt_valid[:valid_timesteps])
            self.valid_samples = len(self.valid_dataset)
            self.valid_dataloader = DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=False)

            print("Loaded validation data loader.\n")

            # Test Set
            self.test_dataset = StockDataset(x_valid[valid_timesteps:], y_prev_valid[valid_timesteps:], y_gt_valid[valid_timesteps:])
            self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)
            self.input_size = x_train.shape[2]
        
        self.Encoder = Encoder(input_size=self.input_size,
                        encoder_num_hidden=encoder_num_hidden,
                        T=T).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T).to(self.device)

        # Loss function
        self.criterion = nn.MSELoss()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate)

    def check_data_set(self, month_range):
        indices_dict = {}
        for m in range(len(month_range)):
            month_begin = month_range[m]
            data = self.s3_bucket.load_from_s3("Data/data_{}.csv".format(month_begin))
            X = data.loc[:, [x for x in data.columns.tolist() if x != 'Date' and x != 'Date Time']].to_numpy()

            if np.any(np.isnan(X)):
                indices_dict[month_begin] = np.where(np.isnan(X))[0]
        
        return indices_dict
    
    def pre_process_old_data(self, train=True):

        if train:
            x = np.zeros((self.train_timesteps, self.T - 1, self.input_size))
            y_prev = np.zeros((self.train_timesteps, self.T - 1))
            y_gt = np.zeros((self.train_timesteps, self.T_predict))

            for bs in range(self.train_timesteps):
                x[bs, :, :] = self.X[bs: (bs + self.T - 1), :]
                y_prev[bs, :] = self.y[bs: (bs + self.T - 1)]
                y_gt[bs, :] = self.y[bs + self.T - 1: bs + self.T - 1 + self.T_predict]

        else:
            x = np.zeros((self.valid_timesteps, self.T - 1, self.input_size))
            y_prev = np.zeros((self.valid_timesteps, self.T - 1))
            y_gt = np.zeros((self.valid_timesteps, self.T_predict))

            start = self.train_timesteps - self.T

            for bs in range(self.valid_timesteps):
                x[bs, :, :] = self.X[bs + self.train_timesteps - self.T: (bs + self.train_timesteps - 1), :]
                y_prev[bs, :] = self.y[bs + self.train_timesteps - self.T: (bs + self.train_timesteps - 1)]
                y_gt[bs, :] = self.y[bs + self.train_timesteps - 1: bs + self.train_timesteps - 1 + self.T_predict]
        
        return x, y_prev, y_gt
    
    def compute_y_total(self, month_range):
        y_total = np.array([])
        for m in range(len(month_range)):
            month_begin = month_range[m]
            data = self.s3_bucket.load_from_s3("Data/data_{}.csv".format(month_begin))
            y = np.array(data['QQQ'])
            y_total = np.append(y_total, y)
        
        return y_total
    
    # Modify here in prepocessing of data. Do differencing of both the Features X and the Target y.
    def process_daily_data(self, data, i, ref_idx, input_size, y_mean):
        data_day = data.loc[i]
        data_day = data_day.bfill()

        X = data_day.loc[:, [x for x in data_day.columns.tolist() if x != 'QQQ' and x != 'Date' and x != 'Date Time']].to_numpy()
        y = np.array(data_day['QQQ']) - y_mean

        x = np.zeros((len(ref_idx), self.T - 1, input_size))
        y_prev = np.zeros((len(ref_idx), self.T - 1))
        y_gt = np.zeros((len(ref_idx), self.T_predict))

        # format x into 3D tensor
        for bs in range(len(ref_idx)):
            x[bs, :, :] = X[ref_idx[bs]:(ref_idx[bs] + self.T - 1), :]
            y_prev[bs, :] = y[ref_idx[bs]:(ref_idx[bs] + self.T - 1)]
            y_gt[bs, :] = y[ref_idx[bs] + self.T - 1: ref_idx[bs] + self.T - 1 + self.T_predict]
        
        return x, y_prev, y_gt
        
    def pre_process_data(self, month_range, y_mean):
        # Load month data into dataframe and loop through each day creating the x, y_prev, and y_gt arrays

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

                x, y_prev, y_gt = self.process_daily_data(data, i, ref_idx, input_size, y_mean)

                x_month[len(ref_idx) * i:len(ref_idx) * (i+1)] = x
                y_prev_month[len(ref_idx) * i:len(ref_idx) * (i+1)] = y_prev
                y_gt_month[len(ref_idx) * i:len(ref_idx) * (i+1)] = y_gt

            print(f'x_month shape: {x_month.shape}')
            print(f'y_prev_month shape: {y_prev_month.shape}')
            print(f'y_gt_month shape: {y_gt_month.shape}')
            
            if m == 0:
                x_year = x_month
                y_prev_year = y_prev_month
                y_gt_year = y_gt_month
            
            else:
                x_year = np.concatenate((x_year, x_month))
                y_prev_year = np.concatenate((y_prev_year, y_prev_month))
                y_gt_year = np.concatenate((y_gt_year, y_gt_month))
    
        return x_year, y_prev_year, y_gt_year

    def train(self):
        
        """Training process."""

        iter_per_epoch = int(
            np.ceil(self.train_samples * 1. / self.batch_size))
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)

        valid_iter_per_epoch = int(
            np.ceil(self.valid_samples * 1. / self.batch_size))
        self.valid_iter_losses = np.zeros(self.epochs * valid_iter_per_epoch)
        self.valid_epoch_losses = np.zeros(self.epochs)

        n_iter = 0

        for epoch in range(self.epochs):
            idx = 0

            # Training Set
            for i, (x, y_prev, y_gt) in enumerate(self.train_dataloader):
                loss = self.train_forward(x, y_prev, y_gt)

                # if (i + 1) % 5 == 0:
                #     print(f'epochs = {epoch}/{self.epochs}, iterations = {i+1}/{iter_per_epoch}, training loss = {loss}')

                self.iter_losses[int(epoch * iter_per_epoch + idx / self.batch_size)] = loss

                idx += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(
                    epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])
                
            # Validation Set
            with torch.no_grad():
                valid_idx = 0

                for i, (x, y_prev, y_gt) in enumerate(self.valid_dataloader):

                    # HERES WHERE VALIDATION IS CALLED AFTER MODEL TRAINING
                    valid_loss = self.validate_forward(x, y_prev, y_gt)
                    
                    self.valid_iter_losses[int(epoch * valid_iter_per_epoch + valid_idx / self.batch_size)] = valid_loss

                    valid_idx += self.batch_size

                    self.valid_epoch_losses[epoch] = np.mean(self.valid_iter_losses[range(
                        epoch * valid_iter_per_epoch, (epoch + 1) * valid_iter_per_epoch)])

            if (epoch+1) % 5 == 0:
                # print("Epochs: ", epoch, " Iterations: ", n_iter,
                #       "Training Loss: ", self.epoch_losses[epoch],
                #       "Validation Loss: ", self.valid_epoch_losses[epoch])

                y_train_pred = self.test(self.train_dataset, self.train_dataloader)
                y_valid_pred = self.test(self.valid_dataset, self.valid_dataloader)

                if not self.old_data:
                    y_test_pred = self.test(self.test_dataset, self.test_dataloader)

                # plt.ioff()
                # plt.figure()

                if self.old_data:
                    plt.plot(range(1, 1 + len(self.y)), self.y + self.y_mean, label="Ground Truth")
                
                else:
                    plt.plot(range(1, 1 + len(self.y_gt_train)), self.y_gt_train[:, 0] + self.y_mean, 
                            label="True - Train")
                    plt.plot(range(1 + len(self.y_gt_train), 1 + len(self.y_gt_train) + len(self.y_gt_valid)), 
                            self.y_gt_valid[:, 0] + self.y_mean, label="True - Valid")

                if y_train_pred.shape[1] == 1:
                    
                    train_start = self.T
                    train_end = train_start + len(y_train_pred)
                    plt.plot(range(train_start, train_end), y_train_pred + self.y_mean, label='Prediction - Training')                   
                    plt.axvline(x=train_end, linestyle='--', color='red')

                    valid_start = train_end
                    valid_end = valid_start + len(y_valid_pred)
                    plt.plot(range(valid_start, valid_end), y_valid_pred + self.y_mean, label='Prediction - Validation')

                    if not self.old_data:
                        test_start = valid_end
                        test_end = test_start + len(y_test_pred)
                        plt.plot(range(test_start, test_end), y_test_pred + self.y_mean, label='Prediction - Test')
                
                plt.legend(loc='upper left')
                plt.xlabel("Time (Minutes)")
                plt.title(f'Epochs: {epoch}, Training Loss: {self.epoch_losses[epoch]:.2f}, Validation Loss: {self.valid_epoch_losses[epoch]:.2f}')

                # plt.xlim(0, 37000)

                if self.old_data:
                    plt.ylabel("Nasdaq-100 (NDX) ($)")
                else:
                    plt.ylabel("QQQ ($)")
                
                plt.show()

                # plt.title("Epochs: {}, N iters: {}, Loss: {}".format(epoch, n_iter, self.epoch_losses[epoch]))
                
                # print("Saving and pushing figures to S3 Bucket.")
                # plt.savefig(os.path.join(self.output_dir, "model_epoch_{}_iter_{}.png".format(epoch, n_iter)))
                # self.s3_bucket.push_to_s3(self.output_dir, "model_epoch_{}_iter_{}.png".format(epoch, n_iter))


    def validate_forward(self, x, y_prev, y_gt):
        valid_loss = 0
            
        _, input_encoded = self.Encoder(Variable(x.type(torch.FloatTensor).to(self.device)))
            
        d_n = self.Decoder._init_states(input_encoded)
        c_n = self.Decoder._init_states(input_encoded)

        y_prev = Variable(y_prev.type(torch.FloatTensor).to(self.device))

        for t in range(self.T_predict):
            if t == 0:
                y_pred, d_n, c_n = self.Decoder(input_encoded, y_prev, d_n, c_n, initial=True)

            else:
                y_pred, d_n, c_n = self.Decoder(input_encoded, y_prev, d_n, c_n, initial=False)
            
            y_true = Variable(y_gt[:, t].type(torch.FloatTensor).to(self.device))

            y_true = y_true.view(-1, 1)
            valid_loss += self.criterion(y_pred, y_true)

            y_prev = y_pred.detach()
        
        return valid_loss.item()

    # Train forward is defined okay here! 
    def train_forward(self, x, y_prev, y_gt):
        """Forward pass."""
        # set losses to zero
        loss = 0

        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.Encoder(Variable(x.type(torch.FloatTensor).to(self.device)))
        
        d_n = self.Decoder._init_states(input_encoded)
        c_n = self.Decoder._init_states(input_encoded)

        y_prev = Variable(y_prev.type(torch.FloatTensor).to(self.device))

        # Put below in for loop range(0, T_predict).
        # Need to check code below.
        for t in range(self.T_predict):
            if t == 0:
                y_pred, d_n, c_n = self.Decoder(input_encoded, y_prev, d_n, c_n, initial=True)
                    
            else:
                y_pred, d_n, c_n = self.Decoder(input_encoded, y_prev, d_n, c_n, initial=False)

            y_true = Variable(y_gt[:, t].type(torch.FloatTensor).to(self.device))

            y_true = y_true.view(-1, 1)
            loss += self.criterion(y_pred, y_true)

            y_prev = y_pred.detach()
        
        # Belongs outside of the for loop        
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()
    
    def test(self, dataset, dataloader):
        """Prediction."""
        y_prediction = np.zeros((len(dataset), self.T_predict))
        
        j = 0
        for i, (x, y_prev, y_gt) in enumerate(dataloader):

            y_prev = Variable(y_prev.type(torch.FloatTensor).to(self.device))
            
            _, input_encoded = self.Encoder(Variable(x.type(torch.FloatTensor).to(self.device)))
            
            d_n = self.Decoder._init_states(input_encoded)
            c_n = self.Decoder._init_states(input_encoded)

            for t in range(self.T_predict):
                if t == 0:
                    y_pred, d_n, c_n = self.Decoder(input_encoded, y_prev, d_n, c_n, initial=True)

                else:
                    y_pred, d_n, c_n = self.Decoder(input_encoded, y_prev, d_n, c_n, initial=False)
                
                y_prev = y_pred
                y_prediction[j:(j + self.batch_size), t] = y_prev.cpu().data.numpy()[:, 0]
                
            j += self.batch_size

        return y_prediction