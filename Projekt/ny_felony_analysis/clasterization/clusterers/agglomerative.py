from sklearn.cluster import AgglomerativeClustering

from ny_felony_analysis.clasterization.clusterers.clusterer import Clusterer


class Agglomerative(Clusterer):
    name = "Agglomerative hierarchical clustering"

    def __init__(self, data, n_clusters, args):
        super().__init__(data)
        try:
            linkage = str(args[0])      # ward, complete, average, single
            affinity = str(args[1])     # euclidean, manhattan
            self.model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                affinity=affinity
            )
        except:
            self.model = AgglomerativeClustering(
                n_clusters=n_clusters
            )
