import sys
from sklearn.cluster import SpectralClustering

from clusterers.clusterer import Clusterer


class Spectral(Clusterer, SpectralClustering):
    name = "K-means"

    def __init__(self, data, n_clusters, args):
        super().__init__(data)

        try:
            n_clusters = n_clusters if n_clusters > 0 else 8

            n_init = int(args[0])
            n_init = n_init if n_init > 0 else 10

            random_state = args[2]
            if random_state is not 'None':
                random_state = int(args[2])
                random_state = random_state if random_state > 0 else None

            self.model = SpectralClustering(
                n_clusters=n_clusters,
                n_init=n_init,
                random_state=random_state,
                n_neighbors=10,
                affinity='rbf'
            )

        except:
            sys.exit("Incorrect parameters")

    def get_inertia_for_elbow(self):
        self.model.fit(self.data)
        return self.model.inertia_
