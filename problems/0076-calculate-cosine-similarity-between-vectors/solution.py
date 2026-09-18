import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	# Implement your code here
	
	v1_mag = np.sqrt(np.sum(v1**2))
	v2_mag = np.sqrt(np.sum(v2**2))

	# Cos(a) = A.B / (|A| * |B|)

	cosine = (v1 @ v2) / (v1_mag*v2_mag)
	angle = np.arccos(cosine)
	return cosine
	