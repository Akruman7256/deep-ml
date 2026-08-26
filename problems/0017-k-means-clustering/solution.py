import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    # Your code here

    def euclidean_distance(p1, p2):
        return np.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))
    
    def assign_clusters(points, centroids):
        assignments = []
        for point in points:                                          # loop over EACH point
            distances = [euclidean_distance(point, c) for c in centroids]  # distance to EACH centroid
            nearest = np.argmin(distances)                                # This calculates the distance of EACH data point w.r.t EVERY initial centroid, and then takes the minimum values out of them as the closest centroid
            assignments.append(nearest)
        return assignments

    def update_centroids(points, assignments, k, old_centroids):
        new_centroids = []
        for cluster_idx in range(k):
            cluster_points = [points[i] for i in range(len(points)) if assignments[i] == cluster_idx]
            
            if len(cluster_points) == 0:
                new_centroids.append(old_centroids[cluster_idx])  # keep old, per the hint
            else:
                new_centroid = tuple(np.mean(cluster_points, axis=0))
                new_centroids.append(new_centroid)
            
        return new_centroids

    centroids = initial_centroids
        
    for _ in range(max_iterations):
        assignments = assign_clusters(points, centroids)
        centroids = update_centroids(points, assignments, k, centroids)
        
    return [tuple(round(x, 4) for x in c) for c in centroids]