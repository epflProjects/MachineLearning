# -*- coding: utf-8 -*-
"""A function to compute the cost."""
from proj1_helpers import *

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def percentageGood(y, tx,w):
    """ Compute the percentage of good prediction"""
    y_pred = predict_labels(w,tx)
    N = len(y)
    e = (y+y_pred)/2
    return e.dot(e) / N
