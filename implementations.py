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
        #Stochastic choice
        random_index = np.random.choice(range(len(y)), batch_size)

        yp = y[random_index]
        txp = tx[random_index]

        #Gradient Descent
        grad = compute_gradient(yp,txp,w)
        
        w = w-gamma*grad
    
    #Computation of the error    
    loss = compute_mse(y,tx,w)    
    return (w, loss)    
    

def least_squares(y, tx):
    """Least squares regression using normal equations"""
    # ***************************************************
    # least squares: Give an estimation of the weights of the regression using normal equations.
    # returns mse, and optimal weights
    # ***************************************************
    #Calcul of the optimal weights
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    
    #Calcul of the error by MSE
    loss = compute_mse(y, tx, w)

    return (w, loss)

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    raise NotImplementedError

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    raise NotImplementedError
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    raise NotImplementedError

## Useful functions.
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





     