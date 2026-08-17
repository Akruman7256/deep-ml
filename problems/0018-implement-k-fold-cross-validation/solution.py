import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    indices = list(range(n_samples))
    if shuffle:
        np.random.shuffle(indices)
    
    base = n_samples//k
    leftover = n_samples%k

    fold_size = []

    for i in range(k):
        if(i<leftover):
            fold_size.append(base + 1)
        else:
            fold_size.append(base)

    folds = []
    start = 0
    for size in fold_size:
        folds.append(indices[start:start+size])
        start+=size

    result = []
    for i in range(k):
        test_indices = folds[i]
        train_indices = []
        for j in range(k):
            if j != i:
                train_indices.extend(folds[j])
        result.append((train_indices, test_indices))

    return result

    