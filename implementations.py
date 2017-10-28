# -*- coding: utf-8 -*-
# Authors : Matteo Giorla, Bastian Nanchen, Arnaud Pannatier
# EPFL Fall 2017
# Machine Learning Course
# Project : 1

"""Basic function of Machine Learning Project 1."""
import numpy as np
from costs import *
from linear_regression import *
from helper import *
from cross_validation import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent

    Returns
        An array of weights and the corresponding loss"""

    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        w = w - gamma * grad

    loss = compute_mse(y,tx,w)
    return (w, loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent

    Returns
        An array of weights and the corresponding loss"""

    w = initial_w
    batch_size = 1

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * gradient

    loss = compute_mse(y,tx,w)
    return (w, loss)


def least_squares(y, tx):
    """calculate mse and the least squares solution.

    Returns
        The least squares solution and an array of weights"""

    to_invert = tx.T.dot(tx)
    mul_y = tx.T.dot(y)
    w = np.linalg.solve(to_invert, mul_y)

    loss = compute_mse(y, tx, w)
    return loss, w


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Returns
        An array of weights"""

    first_part = tx.T.dot(tx) + (2 * tx.shape[0] * lambda_) * np.identity(tx.shape[1])
    sec_part = tx.T.dot(y)
    return np.linalg.solve(first_part, sec_part)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD

    Returns
        The loss and an array of weights"""

    w = initial_w
    losses = []
    threshold = 1e-8

    for i in range(max_iters):
        loss = neg_log_likelihood_loss(y, tx, w)
        gradient = compute_gradient_sig(y, tx, w)
        w = w - (gamma * gradient)
        #converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return neg_log_likelihood_loss(y, tx, w), w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD

    Returns
        The loss and an array of weights"""
        
    w = initial_w
    losses = []
    threshold = 1e-8

    for i in range(max_iters):
        loss = calculate_reg_loss(y, tx, w, lambda_)
        gradient = compute_gradient_sig(y, tx, w) + 2 * lambda_ * w
        w = w - (gamma * gradient)
        #converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return calculate_reg_loss(y, tx, w, lambda_), w
