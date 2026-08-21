import numpy as np

def shuffle_data(X, y, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    X = X.copy()
    y = y.copy()
    n = X.shape[0]
    
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        X[[i, j]] = X[[j, i]]
        y[[i, j]] = y[[j, i]]
    
    return X, y