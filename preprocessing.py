import numpy as np


def selectColumns(tx, n):
	if n >= len(tx):
		return tx
	else:
		return tx[:,:n]

def addColumns(tx,n):
	ret = tx

	for i in range(n):
		if i >1:
			ret = np.hstack((ret,tx**i))


	return ret