import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """

    points = np.array(points)
    query_point = np.array(query_point)
    result = []
    x = query_point[0]
    y = query_point[1]
    xn = [p[0] for p in points]
    yn = [p[1] for p in points]

    for i in range(len(points)):
        distance_i = np.sqrt(np.square(xn[i] - x) + np.square(yn[i] - y))
        result.append((distance_i, tuple(points[i])))
    
    result.sort(key=lambda pair: pair[0])

    final = [point for dist, point in result[:k]]
    
    return final
    
