from sklearn.mixture import GaussianMixture

from clusterers.clusterer import Clusterer


class ExpectationMaximization(Clusterer, GaussianMixture):
    name = "Expectation-Maximization"

    def __init__(self, data):
        super().__init__(data)
