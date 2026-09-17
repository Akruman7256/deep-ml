import numpy as np

def make_diagonal(x):
	"""
	x=np.array(x)
	x=x.astype('float64')
	length = len(x)
	result = np.zeros((length,length), dtype = x.dtype)

	for i in range(0,len(x)):
		result[i][i] = x[i]
	
	return result


	(My code runs fine, the Compiler is just stupid to NOT UNDERSTAND that +0 and -0 are the SAME values, at least in maths they r.)
	"""
	
    x = np.asarray(x, dtype=np.float64)
    return np.eye(len(x), dtype=np.float64) * x