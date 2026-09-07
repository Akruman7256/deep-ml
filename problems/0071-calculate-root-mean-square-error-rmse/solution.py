import numpy as np

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # This single line works perfectly for 1D, 2D, or higher dimensions
    mse = np.mean((y_true - y_pred) ** 2)
    rmse_res = np.sqrt(mse)
    
    return np.round(rmse_res, 3)
