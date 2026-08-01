import numpy as np
from numpy import ndarray

def to_categorical(x: ndarray, n_col=None):
	# Your code here
	# array -> 0 1 2 1 0
	# index -> 0 1 2 3 4

	if n_col is None:
		n_col= np.max(x) + 1

	output = np.zeros((len(x),n_col))

	for i in range(len(x)):
		cat = x[i]
		output[i][cat] = 1

	return output