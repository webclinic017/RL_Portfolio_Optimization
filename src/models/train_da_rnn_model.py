import argparse
import time

import numpy as np

from da_rnn import DA_RNN
from s3_bucket import S3Bucket
import torch
import os
import matplotlib.pyplot as plt

def train_da_rnn(batch_size,
                 encoder_num_hidden,
                 T,
                 T_predict,
                 learning_rate,
                 epochs,
                 parallel,
                 sagemaker=True):

    """DA RNN model"""
    start = time.time()

    # Retrieve and pre-process data
    s3_bucket = S3Bucket()
    data = s3_bucket.load_from_s3('test_model/nasdaq100_padding.csv', index=True)
    X = data.loc[:, [x for x in data.columns.tolist() if x != 'NDX']].to_numpy()
    y = np.array(data.NDX)

    # Build model
    da_rnn = DA_RNN(X=X,
                    y=y,
                    T=T,
                    T_predict=T_predict,
                    encoder_num_hidden=encoder_num_hidden,
                    decoder_num_hidden=encoder_num_hidden,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    parallel=parallel,
                    sagemaker=sagemaker)
    
    # Train model
    da_rnn.train()

    # Test model
    # Prediction
    y_pred = da_rnn.test()

    fig = plt.figure()
    plt.plot(da_rnn.y[da_rnn.train_timesteps:], label="True")
    plt.plot(y_pred, label='Predicted')
    plt.legend(loc='upper left')
    plt.xlabel("Time (minutes)")
    plt.ylabel("Nasdaq-100 (NDX) ($)")

    # Define output directory
    if sagemaker:
        output_dir = os.environ['SM_OUTPUT_DATA_DIR'] + "/"
    else:
        output_dir = "./"

    print("Saving and pushing figures to S3 Bucket.")
    plt.savefig(os.path.join(output_dir, "prediction.png"))
    s3_bucket.push_to_s3(output_dir, "prediction.png")

    end = time.time()

    # Define model directory
    if sagemaker:
        model_dir = os.environ['SM_MODEL_DIR'] + "/"
    else:
        model_dir = "./"

    torch.save(da_rnn.state_dict(), os.path.join(model_dir, 'model_checkpoint.pth'))
    s3_bucket.push_to_s3(model_dir, "model_checkpoint.pth")

    # Record final training loss
    loss = da_rnn.epoch_losses[-1]

    # mean_score = (batch_size - 25) ** 2 \
    #              - (encoder_num_hidden - 3) * (decoder_num_hidden - 5) + encoder_num_hidden + decoder_num_hidden

    duration = end - start

    print('{:.2f} training objective.'.format(loss))
    
    print('duration = {:.2f}'.format(duration))


if __name__ == "__main__":
    # Extract hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Size of batches to use (default: 128)')
    parser.add_argument('--encoder_num_hidden', type=int, default=128, help='Size of hidden layer for encoder (default: 128)')
    parser.add_argument('--T', type=int, default=10, help='Number of past minutes to use for prediction (default: 10)')
    parser.add_argument('--T_predict', type=int, default=1, help='Number of minutes ahead to predict (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate to use (default: 0.001')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to run (default: 50)')
    parser.add_argument('--parallel', type=bool, default=False, help='Whether to run in parallel (default: False)')
    args = parser.parse_args()

    print('Running model training.')
    print('Model parameters: {}'.format(args))

    # batchsize = 128
    # nhidden_encoder = 128
    # nhidden_decoder = 128
    # T = 10
    # T_predict = 1
    # lr = 0.001
    # epochs = 50

    train_da_rnn(batch_size=args.batch_size,
                 encoder_num_hidden=args.encoder_num_hidden,
                 T=args.T,
                 T_predict=args.T_predict,
                 learning_rate=args.learning_rate,
                 epochs=args.epochs,
                 parallel=args.parallel,
                 sagemaker=True)

    print('Done.')
