# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


## A modifier encore pour les tests
def cross_validation(y, x, k_indices, k, lambda_, degree):
    """ Cross Validation on our train sample """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    y_range = np.arange(len(y))
    train_ind = np.delete(y_range, k_indices[k])

    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree: TODO
    # ***************************************************

    train_tx = build_poly(x[train_ind], degree)
    test_tx = build_poly(x[k_indices[k]], degree)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    (loss_tr, w) = ridge_regression(y[train_ind], train_tx, lambda_)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    loss_te = compute_mse(y[k_indices[k]], test_tx, w)
    return loss_tr, loss_te
