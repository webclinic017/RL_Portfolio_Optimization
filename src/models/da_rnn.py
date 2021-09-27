"""
DA-RNN model architecture.

@author Zhenye Na 05/21/2018
@modified 11/05/2019

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).
    [2] Chandler Zuo. "A PyTorch Example to Use RNN for Financial Prediction" (2017).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable


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


class DA_RNN(nn.Module):
    """Dual-Stage Attention-Based Recurrent Neural Network."""

    def __init__(self, X, y, T, T_predict,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 parallel=False):
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

        self.X = X
        self.y = y

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.Encoder = Encoder(input_size=X.shape[1],
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

        # Training set
        self.train_timesteps = int(self.X.shape[0] * 0.7)
        self.y = self.y - np.mean(self.y[:self.train_timesteps])
        self.input_size = self.X.shape[1]

    def train(self):
        """Training process."""
        iter_per_epoch = int(
            np.ceil(self.train_timesteps * 1. / self.batch_size))
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)

        n_iter = 0

        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T))

            idx = 0

            while idx < self.train_timesteps:
                # get the indices of X_train
                indices = ref_idx[idx:(idx + self.batch_size)]
                x = np.zeros((len(indices), self.T - 1, self.input_size))
                y_prev = np.zeros((len(indices), self.T - 1))

                # Modify y_gt
                # Ground truth must have shape
                # (batch_size, T_predict): indices + self.T -1: indices + self.T -1 + T_predict
                y_gt = np.zeros((len(indices), self.T_predict))

                # Defines ground truth for only minute ahead forecasting                
                # y_gt = self.y[indices + self.T - 1]

                # format x into 3D tensor
                for bs in range(len(indices)):
                    x[bs, :, :] = self.X[indices[bs]:(indices[bs] + self.T - 1), :]
                    y_prev[bs, :] = self.y[indices[bs]: (indices[bs] + self.T - 1)]
                    y_gt[bs, :] = self.y[indices[bs] + self.T - 1: indices[bs] + self.T - 1 + self.T_predict]

                loss = self.train_forward(x, y_prev, y_gt)
                self.iter_losses[int(
                    epoch * iter_per_epoch + idx / self.batch_size)] = loss

                idx += self.batch_size
                n_iter += 1

                if n_iter % 10000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(
                    epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])

            if epoch % 10 == 0:
                print("Epochs: ", epoch, " Iterations: ", n_iter,
                      " Loss: ", self.epoch_losses[epoch])

            if epoch % 10 == 0:
                # NEED TO MODIFY PLOTS IF FORECASTING N_STEPS AHEAD
                y_train_pred = self.test(on_train=True)
                y_test_pred = self.test(on_train=False)

                # print("y_train_pred shape = " + str(y_train_pred.shape))
                # print("y_test_pred shape = " + str(y_test_pred.shape))

                # y_pred = np.concatenate((y_train_pred, y_test_pred))
                plt.ioff()
                plt.figure()
                plt.plot(range(1, 1 + len(self.y)), self.y, label="True")

                if y_train_pred.shape[1] == 1:

                    plt.plot(range(self.T, len(y_train_pred) + self.T),
                            y_train_pred, label='Predicted - Train')
                    plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1),
                            y_test_pred, label='Predicted - Test')
                    plt.legend(loc='upper left')
                    # plt.show()

                else:

                    for t1 in range(len(y_train_pred)):
                        if t1 % self.T_predict == 0:
                            x_values = [t1 + i for i in range(self.T, self.T + self.T_predict)]
                            plt.plot(x_values, y_train_pred[t1], label='Predicted - Train')

                    for t2 in range(len(y_test_pred)):
                        if t2 % self.T_predict == 0:
                            x_values = [t2 + i for i in range(self.train_timesteps, self.train_timesteps + self.T_predict)]
                            plt.plot(x_values, y_test_pred[t2], label='Predicted - Test')

                    # plt.legend(loc='upper left')
                    # plt.show()
                
                plt.title("Epochs: {}, N iters: {}, Loss: {}".format(epoch, n_iter, self.epoch_losses[epoch]))
                plt.savefig("model_epoch_{}_iter_{}.png".format(epoch, n_iter))

    def train_forward(self, X, y_prev, y_gt):
        """Forward pass."""
        # set losses to zero
        loss = 0

        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.Encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
        
        d_n = self.Decoder._init_states(input_encoded)
        c_n = self.Decoder._init_states(input_encoded)

        y_prev = Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device))

        # Put below in for loop range(0, T_predict).
        # Need to check code below.
        for t in range(self.T_predict):
            if t == 0:
                y_pred, d_n, c_n = self.Decoder(input_encoded, y_prev, d_n, c_n, initial=True)
                    
            else:
                y_pred, d_n, c_n = self.Decoder(input_encoded, y_prev, d_n, c_n, initial=False)

            y_true = Variable(torch.from_numpy(
                y_gt[:, t]).type(torch.FloatTensor).to(self.device))

            y_true = y_true.view(-1, 1)
            loss += self.criterion(y_pred, y_true)

            y_prev = y_pred.detach()
        
        # Belongs outside of the for loop        
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def test(self, on_train=False):
        """Prediction."""

        if on_train:
            y_prediction = np.zeros((self.train_timesteps - self.T + 1, self.T_predict))

        else:
            y_prediction = np.zeros((self.X.shape[0] - self.train_timesteps, self.T_predict))

        i = 0

        while i < len(y_prediction):
            batch_idx = np.array(range(len(y_prediction)))[i:(i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(
                        batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(
                        batch_idx[j], batch_idx[j] + self.T - 1)]
                else:
                    X[j, :, :] = self.X[range(
                        batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1), :]
                    y_history[j, :] = self.y[range(
                        batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1)]
            
            y_history = Variable(torch.from_numpy(
                y_history).type(torch.FloatTensor).to(self.device))
            
            _, input_encoded = self.Encoder(
                Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
            
            d_n = self.Decoder._init_states(input_encoded)
            c_n = self.Decoder._init_states(input_encoded)

            for t in range(self.T_predict):
                if t == 0:
                    y_pred, d_n, c_n = self.Decoder(input_encoded, y_history, d_n, c_n, initial=True)

                else:
                    y_pred, d_n, c_n = self.Decoder(input_encoded, y_history, d_n, c_n, initial=False)
                
                y_history = y_pred
                y_prediction[i:(i + self.batch_size), t] = y_history.cpu().data.numpy()[:, 0]
                
            i += self.batch_size

        return y_prediction
