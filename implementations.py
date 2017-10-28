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
    """Linear regression using gradient descent"""
    # ***************************************************
    # least_squares_GD: Give an estimation of the weights of the regression using gradient descent
    # returns mse, and optimal weights
    # ***************************************************
    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)

        # update w by gradient descent formula
        w = w-gamma*grad

    #Computation of the error
    loss = compute_mse(y,tx,w)
    return (w, loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    # ***************************************************
    # least_squares_SGD: Give an estimation of the weights of the regression using stochastic gradient descent
    # returns mse, and optimal weights
    # ***************************************************
    w = initial_w
    batch_size = 1

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * gradient

    #Computation of the error
    loss = compute_mse(y,tx,w)
    return (w, loss)

def least_squares(y, tx):
    """calculate mse and the least squares solution."""
    # w_star using least squares
    to_invert = tx.T.dot(tx)
    mul_y = tx.T.dot(y)
    w_star = np.linalg.solve(to_invert, mul_y)
    # MSE
    const_part = 1/(2*y.size)
    e = (y - (tx.dot(w_star)))
    e_squared = e.T.dot(e)
    return const_part * e_squared, w_star

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    first_part = tx.T.dot(tx) + (2 * tx.shape[0] * lambda_) * np.identity(tx.shape[1])
    sec_part = tx.T.dot(y)
    return np.linalg.solve(first_part, sec_part)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    w = initial_w
    losses = []
    threshold = 1e-8

    for i in range(max_iters):
        loss = calculate_neg_log_like_loss(y, tx, w)
        gradient = calculate_gradient_sig(y, tx, w)
        w = w - (gamma * gradient)
        # log info
        if i % 100 == 0:
           print("Current iteration={i}, loss={l}".format(i=i, l=loss))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        print("loss={l}".format(l=calculate_neg_log_like_loss(y, tx, w)))
    return calculate_neg_log_like_loss(y, tx, w), w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    w = initial_w
    losses = []
    threshold = 1e-8

    for i in range(max_iters):
        loss = calculate_neg_log_like_loss(y, tx, w) - (lambda_ * np.linalg.norm(w)**2)
        gradient = calculate_gradient_sig(y, tx, w)
        w = w - (gamma * gradient)

        # log info
        if i % 100 == 0:
           print("Current iteration={i}, loss={l}".format(i=i, l=loss))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        print("loss={l}".format(l=calculate_neg_log_like_loss(y, tx, w)))
    return calculate_neg_log_like_loss(y, tx, w), w
