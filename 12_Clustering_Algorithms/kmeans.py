"""
K Means Algorithm:
    1. Decide number of clusters
    2. Select random centroids
    3. Assign clusters
    4. Move centroids
    5. Check Finish
"""
import random
class KMeans:
    def __init__(self, number_of_clusters = 2, max_iteration = 100):
        self.number_of_clusters = number_of_clusters
        self.max_iteration = max_iteration
        self.centroids = None
    
    def fit_predict(self, X):
        random_index = random.sample(range(0,X.shape[0]),self.number_of_clusters)
        self.centroids = X[random_index]
        for i in range(self.max_iteration):
            # Assign Clusters
            cluster_group = self.assign_clusters(X)

            # Move Centroids
            # Check Finish
    
    def assign_clusters(self, X):
        cluster_group = []
        distances = []
        for row in X:
            for centroid in self.centroids:

        return cluster_group


