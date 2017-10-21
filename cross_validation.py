# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np
from costs import *
import implementations as imp

def cross_validation_run(tx, y):
    seed = 1
    #degree = 7
    k_fold = 4
    #lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # cross validation:
    # ***************************************************
    #for lambda_ in lambdas:
    rmse_train = []
    rmse_test = []
    w = []
    for k in range(0, k_fold):
        loss_tr, loss_te, w_ = cross_validation(y, tx, k_indices, k)
        rmse_train.append(loss_tr)
        rmse_test.append(loss_te)
        w.append(w_)
        print(k, " loss train:", loss_tr, " loss test", loss_te, " weights:", w)
    rmse_tr.append(np.mean(rmse_train))
    rmse_te.append(np.mean(rmse_test))
    return

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
def cross_validation(y, x, k_indices, k):
    """ Cross Validation on our train sample """
    # ***************************************************
    # get k'th subgroup in test, others in train:
    # ***************************************************
    y_range = np.arange(len(y))
    train_ind = np.delete(y_range, k_indices[k])

    # ***************************************************
    # form data with polynomial degree:
    # ***************************************************

    #train_tx = build_poly(x[train_ind], degree)
    #test_tx = build_poly(x[k_indices[k]], degree)
    # ***************************************************
    # ridge regression:
    # ***************************************************
    #(loss_tr, w) = ridge_regression(y[train_ind], train_tx, lambda_)
    loss_tr, w = imp.least_squares(y[train_ind], x[train_ind])

    # ***************************************************
    # calculate the loss for train and test data:
    # ***************************************************
    loss_te = compute_mse(y[k_indices[k]], x[k_indices[k]], w)
    return loss_tr, loss_te, w
