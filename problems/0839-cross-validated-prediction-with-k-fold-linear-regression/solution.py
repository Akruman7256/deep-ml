import numpy as np

def cross_val_predict(X, y, k):
    """
    Return out-of-fold predictions for every sample using k-fold
    cross-validation with a linear regression model (with intercept).
    """
    # Your code here
    X=np.array(X)
    y=np.array(y)

    n = X.shape[0]
    indices = list(range(n))

    base = n // k
    leftover = n % k
    
    fold_sizes = []
    for i in range(k):
        if i < leftover:
            fold_sizes.append(base + 1)
        else:
            fold_sizes.append(base)

    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start:start+size])
        start+=size
        
    test_indices = []
    result = []
    predictions = np.zeros(n)  # placeholder for final predictions, in original order
    
    for i in range(k):
        test_indices = folds[i]
        train_indices = []
        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        
        # add bias column (column of 1s) to both train and test
        X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        
        # Normal Equation
        theta = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
        
        # predict on validation fold
        preds = X_test_b @ theta
        
        # place predictions back at their ORIGINAL indices
        for idx, pred in zip(test_indices, preds):
            predictions[idx] = pred
    
    return predictions.tolist()
        
