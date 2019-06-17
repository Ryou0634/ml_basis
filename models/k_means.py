import numpy as np

from .modules.distance_utils import get_euclid_distance, get_cosine_distance


class KMeans:

    def __init__(self,
                 n_clusters: int,
                 n_dim: int,
                 distance_metric: str = 'euclid'):
        self.n_clusters = n_clusters
        self.means = np.random.random((n_clusters, n_dim))
        self.distance_metric = distance_metric

    def __call__(self, xs: np.ndarray):
        return self.predict(xs)

    def predict(self, xs: np.ndarray):

        if self.distance_metric == 'euclid':
            distance_matrix = get_euclid_distance(xs, self.means, for_all_combinations=True)
        elif self.distance_metric == 'cosine':
            distance_matrix = get_cosine_distance(xs, self.means, for_all_combinations=True)

        assigned_clusters = distance_matrix.argmin(axis=1)
        return assigned_clusters

    def fit(self, xs: np.ndarray, max_iter: int = 100):
        prev_clusters = None
        for _ in range(max_iter):
            # assignment step
            assigned_clusters = self.predict(xs)

            if prev_clusters is not None and (assigned_clusters == prev_clusters).all():
                break
            prev_clusters = assigned_clusters

            # update step
            for i in range(self.n_clusters):
                self.means[i] = xs[assigned_clusters == i].mean(axis=0)
