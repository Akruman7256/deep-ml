import numpy as np
def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here


    m = X.shape[0]  # number of samples

    for epoch in range(n_epochs):
        if method == 'batch':
            y_pred = X @ weights
            error = y_pred - y
            gradient = (2/m) * (X.T @ error)
            weights = weights - learning_rate * gradient
            
        elif method == 'stochastic':
            for i in range(m):
                X_i = X[i:i+1]
                y_i = y[i:i+1]
                y_pred_i = X_i @ weights
                error_i = y_pred_i - y_i
                grad_i = (2/1) * (X_i.T @ error_i)
                weights = weights - learning_rate* grad_i
            ...
        elif method == 'mini_batch':
            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
                b = X_batch.shape[0]   # actual number of samples in this chunk (last chunk might be smaller!)
                y_pred = X_batch @ weights
                error = y_pred - y_batch
                gradient = (2/b) * (X_batch.T @ error)
                weights = weights - learning_rate * gradient

    return weights
