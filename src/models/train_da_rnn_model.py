import argparse
import time

import numpy as np

from da_rnn import DA_RNN


def train_da_rnn(encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 parallel):
    """DA RNN model"""
    start = time.time()

    # Retrieve and pre-process data
    x = np.zeros((10, 1))
    y = np.zeros((10, 1))
    t = 10

    # Train model
    da_rnn = DA_RNN(X=x,
                    y=y,
                    T=t,
                    encoder_num_hidden=encoder_num_hidden,
                    decoder_num_hidden=decoder_num_hidden,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    parallel=parallel)
    # da_rnn.train()
    end = time.time()

    # Record output
    mean_score = (batch_size - 25) ** 2 \
                 - (encoder_num_hidden - 3) * (decoder_num_hidden - 5) + encoder_num_hidden + decoder_num_hidden
    duration = end - start
    print('{:.2f} training objective.'.format(10 * duration - 1000.0 * mean_score))


if __name__ == "__main__":
    # Extract hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to run (default: 1000)')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='Learning rate to use (default: 0.5')
    parser.add_argument('--batch_size', type=int, default=10, help='Size of batches to use (default: 10)')
    parser.add_argument('--encoder_num_hidden', type=int, default=3,
                        help='Size of hidden layer for encoder (default: 3)')
    parser.add_argument('--decoder_num_hidden', type=int, default=3,
                        help='Size of hidden layer for decoder (default: 3)')
    parser.add_argument('--parallel', type=bool, default=False, help='Whether to run in parallel (default: False)')
    args = parser.parse_args()

    print('Running model training.')
    print('Model parameters: {}'.format(args))
    train_da_rnn(encoder_num_hidden=args.encoder_num_hidden,
                 decoder_num_hidden=args.decoder_num_hidden,
                 batch_size=args.batch_size,
                 learning_rate=args.learning_rate,
                 epochs=args.epochs,
                 parallel=args.parallel)
    print('Done.')
