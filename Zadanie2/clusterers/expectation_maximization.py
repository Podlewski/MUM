from sklearn.mixture import GaussianMixture

from clusterers.clusterer import Clusterer


class ExpectationMaximization(Clusterer):
    name = "Expectation-Maximization"
    nmb_clusters = 5
    tolerance = 0.001
    max_iterations = 100

    def __init__(self, data, n_clusters, args):
        super().__init__(data)

        try:
            tol = float(args[0])
            iterations = int(args[1])

            self.n_clusters = n_clusters if n_clusters > 0 else 5
            self.tolerance = tol if tol > 0 else 0.001
            self.max_iterations = iterations if iterations > 0 else 100

            self.model = GaussianMixture(n_components=self.n_clusters,
                                         tol=self.tolerance,
                                         max_iter=self.max_iterations)
        except:
            self.model = GaussianMixture()
