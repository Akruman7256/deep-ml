import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
	 
	#Standardization: (x - mean) / std_dev
    #Min-Max: (x - min) / (max - min)

	#Quick mental trick for future: axis=0 = "go down" (rows collapse), axis=1 = "go across" (columns collapse). The axis number tells you which dimension shrinks to 1.

    standardized_data = (data - mean) / std
    
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    
	

    return np.round(standardized_data,decimals=4), np.round(normalized_data,decimals=4)


