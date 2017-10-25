# -*- coding: utf-8 -*-
"""regression using generalized linear model in case of poissonnian distribution"""
import numpy as np
from costs import *

def reg_poisson_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    w = initial_w
    losses = []
    threshold = 1e-8

    for i in range(max_iters):
        per = percentageGood(y, tx, w)
        gradient = calculate_poisson_grad(y, tx, w) + 2*lambda_*w
        w = w - (gamma * gradient)


        # log info
        # if i % 100 == 0:
        #     per = percentageGood(y, tx,w)
        #     print("Current iteration={i}, percentage={l}".format(i=i, l=per))
        losses.append(per)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return per, w


def calculate_poisson_grad(y, tx, w):
    """compute the gradient of loss."""
    s = np.exp(tx.dot(w))
    return tx.T.dot(s - y)