import sys
from sklearn.cluster import KMeans

from clusterers.clusterer import Clusterer


class Kmeans(Clusterer, KMeans):
    name = "K-means"

    def __init__(self, data, n_clusters, args):
        super().__init__(data)

        try:
            n_clusters = n_clusters if n_clusters > 0 else 5

            n_init = int(args[0])
            n_init = n_init if n_init > 0 else 10

            max_iter = int(args[1])
            max_iter = max_iter if max_iter > 0 else 300

            random_state = args[2]
            if random_state is not 'None':
                random_state = int(args[2])
                random_state = random_state if random_state > 0 else None

            self.model = KMeans(
                n_clusters=n_clusters,
                n_init=n_init,
                max_iter=max_iter,
                random_state=random_state
            )

        except:
            sys.exit("Incorrect parameters")

    def get_labels(self):
        print(self.model.fit_predict(self.data))
        return self.model.fit_predict(self.data)

    def get_inertia_for_elbow(self):
        self.model.fit(self.data)
        return self.model.inertia_
