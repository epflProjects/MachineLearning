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
    loss_te=compute_mse(y[k_indices[k]], test_tx, w)
    return loss_tr, loss_te





     