# -*- coding: utf-8 -*-
"""Helper functions."""

import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = len(y)
    e = y-tx.dot(w)
    return -1/N*tx.T.dot(e)


def standardizeMatrix(tx):
    """Standardize the argument matrix"""
    for col in range(len(tx[0])):
        if col != 22:
            tx[:,col], mean, std = standardize(tx[:,col])
    return tx


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    if std_x != 0:
        x = x / std_x
    return x, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Returns:
        an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
