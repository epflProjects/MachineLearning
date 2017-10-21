# -*- coding: utf-8 -*-
"""
Linear regression useful functions
"""

import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    return (1/(1 + np.exp(-t)))

def calculate_neg_log_like_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    s = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(s)) + (1 - y).T.dot(np.log(1 - s))
    loss = -loss[0][0]
    return loss

def calculate_gradient_sig(y, tx, w):
    """compute the gradient of loss."""
    s = sigmoid(tx.dot(w))
    return tx.T.dot(s - y)
