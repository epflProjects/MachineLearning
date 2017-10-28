import numpy as np

def prepro(M,c):
    return add_product_between_columns(addColumns(M, c), M)

def addColumns(tx,n):
    """Extends the data matrix by adding the power of the columns"""
    ret = tx

    for i in range(1,n):

        if i >1:
            ret = np.hstack((ret,tx**i))
        else:
         	ret = np.hstack([ret, np.ones((len(tx),1))])
    return ret


def add_product_between_columns(M, tx):
    """Extend the data matrix by adding the product between columns"""
    ret = M
    n = len(tx[0])

    for i in range(n):
        ret = np.hstack((ret, np.multiply(tx[:, i], tx.T).T))
    return ret
