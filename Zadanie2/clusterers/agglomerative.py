from sklearn.cluster import AgglomerativeClustering

from clusterers.clusterer import Clusterer


class Agglomerative(Clusterer):
    name = "Agglomerative hierarchical clustering"

    def __init__(self, data, n_clusters=2, linkage='ward', affinity='euclidean'):
        super().__init__(data)
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            affinity=affinity
        )
