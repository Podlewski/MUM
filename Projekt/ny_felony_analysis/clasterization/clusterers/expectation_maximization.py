from sklearn.mixture import GaussianMixture

from ny_felony_analysis.clasterization.clusterers.clusterer import Clusterer

covariance_type = {0: "full", 1: "tied", 2: "diag", 3: "spherical"}


class ExpectationMaximization(Clusterer):
    name = "Expectation-Maximization"
    nmb_clusters = 5
    max_iterations = 100

    def __init__(self, data, n_clusters, args):
        super().__init__(data)

        try:
            type_number = int(args[0])
            iterations = int(args[1])

            self.n_clusters = n_clusters if n_clusters > 0 else 5
            self.max_iterations = iterations if iterations > 0 else 100

            self.model = GaussianMixture(n_components=self.n_clusters,
                                         covariance_type=covariance_type[type_number],
                                         max_iter=self.max_iterations)
        except:
            self.model = GaussianMixture()
