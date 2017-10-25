import numpy as np


def selectColumns(tx, n):
    if n >= len(tx):
        return tx
    else:
        return tx[:,:n]

def addColumns(tx,n):
    ret = tx

    for i in range(1,n):

        if i >1:
            ret = np.hstack((ret,tx**i))
        else:
         	ret = np.hstack([ret, np.ones((len(tx),1))])
    return ret

def addFunckyThings(M, tx):
	ret = M
	#logTx = tx+1+np.abs(np.min(tx))
	#ret = np.hstack((ret,np.log(logTx)))
	#ret = np.hstack((ret,np.exp(tx)))
	ret = np.hstack((ret,np.sin(tx)))
	return ret
	
