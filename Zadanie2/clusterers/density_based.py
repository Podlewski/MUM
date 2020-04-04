from sklearn.cluster import DBSCAN

from clusterers.clusterer import Clusterer


class DensityBased(Clusterer, DBSCAN):
    name = "Density-Based Spatial Clustering of Applications with Noise"

    def __init__(self, data):
        super().__init__(data)
