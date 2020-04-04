from sklearn.cluster import KMeans

from clusterers.clusterer import Clusterer


class Kmeans(Clusterer, KMeans):
    name = "K-means"

    def __init__(self, data):
        super().__init__(data)
