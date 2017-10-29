# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np
from costs import *
import implementations as imp

def cross_validation_run(tx, y):
    """Perform the 4 groups cross-validation

    Returns
        An array of weights, the loss and the percentage of good predictions"""

    seed = 1
    k_fold = 4

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    rmse_te = []
    rmse_test = []
    w = []
    goodPer = []
    for k in range(0, k_fold):
        loss_te, w_, goodPerI = cross_validation(y, tx, k_indices, k)
        rmse_test.append(loss_te)
        w.append(w_)

        rmse_te.append(rmse_test)
        goodPer.append(goodPerI)

    return np.mean(w, axis=0), np.mean(rmse_te), np.mean(goodPer)


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Returns
        An array of indices corresponding to each fold"""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k):
    """ Cross Validation on our train sample using ridge regression

    Returns
        An array of weights, the loss and the percentage of good predictions"""

    y_range = np.arange(len(y))
    train_ind = np.delete(y_range, k_indices[k])

    lambda_ = np.logspace(-10, 0, 100)
    #lambda_ = np.logspace(-5, 0, 15)
    best_lambda = 0.0
    best_perc = 0.0

    for lambd in lambda_:
        try:
            w = imp.ridge_regression(y[train_ind], x[train_ind], lambd)
            goodPer = percentageGood(y[k_indices[k]], x[k_indices[k]], w)

            if best_perc < goodPer:
                best_perc = goodPer
                best_lambda = lambd
        except np.linalg.linalg.LinAlgError as err:
            pass

    loss_te = compute_mse(y[k_indices[k]], x[k_indices[k]], w)
    w = imp.ridge_regression(y[train_ind], x[train_ind], best_lambda)
    print("best lambda: ", best_lambda, "best perc: ", best_perc)
    return loss_te, w, best_perc
