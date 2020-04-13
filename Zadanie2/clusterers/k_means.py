from sklearn.cluster import KMeans

from clusterers.clusterer import Clusterer


class Kmeans(Clusterer, KMeans):
    name = "K-means"

    def __init__(self, data, n_clusters=8, n_init=10, max_iter= 300, random_state=None):
        super().__init__(data)
        self.model = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )

    def get_labels(self):
        return self.model.labels_