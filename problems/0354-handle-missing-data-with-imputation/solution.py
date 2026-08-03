import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    # Your code here
    result = data.copy()

    for i in range(data.shape[1]):
        value = result[:,i]
        missing = np.isnan(value)      #provides output as True/False 
        non_missing = value[~missing]  #negation of missing values

        if np.all(missing):
            continue

        if strategy == 'mean':
            fill = np.mean(non_missing)
        elif strategy == 'median':
            fill = np.median(non_missing)
        elif strategy == 'mode':
            values,counts = np.unique(non_missing, return_counts=True)
            fill = values[np.argmax(counts)]
        
        value[missing] = fill

    return result