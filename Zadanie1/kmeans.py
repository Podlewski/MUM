from sklearn import cluster as cl

from classifier import Classifier


class KMeans(Classifier):

    def __init__(self, data, training_fraction=0.3):
        super().__init__(data, training_fraction)
        self.model = cl.KMeans()
