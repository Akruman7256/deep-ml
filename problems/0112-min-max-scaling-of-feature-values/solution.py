import numpy as np
def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    x=np.array(x)
    min_val = min(x)
    max_val = max(x)

    result = []
    norm = 0
    for i in range(len(x)):
        norm = (x[i]- min_val)/(max_val-min_val) 
        result.append(norm)

    return np.array(result)

    pass