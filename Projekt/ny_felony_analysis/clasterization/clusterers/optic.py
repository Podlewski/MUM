from sklearn.cluster import OPTICS

from ny_felony_analysis.clasterization.clusterers.clusterer import Clusterer


class Optic(Clusterer, OPTICS):
    name = "Optics"

    def __init__(self, data, args):
        super().__init__(data)
        min_samples = int(args[0])
        min_samples = min_samples if min_samples > 0 else 20
        xi = int(args[1])
        xi = xi if xi > 0 else 0.05
        min_cluster_size = int(args[2])
        min_cluster_size = min_cluster_size if min_cluster_size > 0 else 0.05

        self.model = OPTICS(
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size
        )

    def get_inertia_for_elbow(self):
        self.model.fit(self.data)
        return self.model.inertia_
