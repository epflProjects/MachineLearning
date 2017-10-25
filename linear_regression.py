# -*- coding: utf-8 -*-
"""
Linear regression useful functions
"""

import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    return 0.5*np.tanh(t/2)+0.5

def calculate_neg_log_like_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1+np.exp(tx.dot(w)))-y.T.dot(tx.dot(w)))

def calculate_reg_loss(y, tx, w, lambda_):
    """compute the cost by negative log likelihood."""
    return calculate_neg_log_like_loss(y, tx, w)+(lambda_ * np.linalg.norm(w)**2)

def calculate_gradient_sig(y, tx, w):
    """compute the gradient of loss."""
    s = sigmoid(tx.dot(w))
    return tx.T.dot(s - y)
