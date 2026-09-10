import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features) data = [row1:rown,col1-coln]
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here

    data_std = np.zeros_like(data)

    for j in range(data.shape[1]):
        mean_j = np.mean(data[:,j])
        std_j = np.std(data[:,j])
        data_std[:,j] = (data[:,j]-mean_j)/std_j
    
    cov_matrix = np.cov(data_std, rowvar = False)
    eiganval , eiganvect = np.linalg.eigh(cov_matrix)
    sorted_eiganval = np.argsort(eiganval)[::-1]           
    sorted_eiganvect = eiganvect[:,sorted_eiganval]

    top_k_sorted = sorted_eiganvect[:, :k]

    for i in range(k):
        col = top_k_sorted[:,i]
        nonzero_indices = np.where(np.abs(col)>=1e-10)[0]

        if len(nonzero_indices) > 0:
            first_idx = nonzero_indices[0]
            # If that first non-zero element is negative, flip the column
            if col[first_idx] < 0:
                top_k_sorted[:, i] *= -1 

    return top_k_sorted
     


