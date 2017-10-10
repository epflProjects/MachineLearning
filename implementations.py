# -*- coding: utf-8 -*-
# Authors : Matteo Giorla, Bastian Nanchen, Arnaud Pannatier
# EPFL Fall 2017
# Machine Learning Course
# Project : 1
"""Basic function of Machine Learning Project 1."""
import numpy as np

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
    """Least squares regression using normal equations"""
    # ***************************************************
    # least squares: Give an estimation of the weights of the regression using normal equations.
    # returns mse, and optimal weights
    # ***************************************************
    #Computation of the optimal weights
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    
    #Computation of the error by MSE
    loss = compute_mse(y, tx, w)

    return (w, loss)

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    # ***************************************************
    # Ridge : Give an estimation of the weights of the regression using normal equations.
    # returns mse, and optimal weights
    # ***************************************************
        
    #Computation of the optimal weights
    N = len(y)
    lenX = len(tx.T)
    
    w = np.linalg.inv(tx.T.dot(tx)+2*N*lambda_*np.eye(lenX)).dot(tx.T).dot(y);
    
    #Computation of the error by MSE
    loss = compute_mse(y, tx, w)
    
    return (w, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    raise NotImplementedError
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    raise NotImplementedError

## Useful functions.################################################################################################################################################
def compute_mse(y, tx, w):
    # ***************************************************
    # compute loss by MSE
    # ***************************************************
    e = y-tx.dot(w)
    N = len(y)
    return 1/(2*N)*e.T.dot(e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # Compute gradient
    # ***************************************************
    N = len(y)
    e = y-tx.dot(w)
    return -1/N*tx.T.dot(e)

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
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





     