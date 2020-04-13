from sklearn.mixture import GaussianMixture

from clusterers.clusterer import Clusterer


class ExpectationMaximization(Clusterer):
    name = "Expectation-Maximization"
    n_clusters = 5
    tolerance = 0.001
    max_iterations = 100

    def __init__(self, data, args):
        super().__init__(data)

        try:
            n_comp = int(args[0])
            tol = float(args[1])
            iterations = int(args[2])

            self.n_clusters = n_comp if n_comp > 0 else 5
            self.tolerance = tol if tol > 0 else 0.001
            self.max_iterations = iterations if iterations > 0 else 100

            self.model = GaussianMixture(n_components=self.n_clusters,
                                         tol=self.tolerance,
                                         max_iter=self.max_iterations)
        except:
            self.model = GaussianMixture()

    def get_labels(self):
        self.model.fit(self.data)
        print(self.model.predict(self.data))
        return self.model.predict(self.data)
