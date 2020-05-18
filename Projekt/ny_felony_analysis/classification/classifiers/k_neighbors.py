from sklearn import neighbors as n

from classifiers.classifier import Classifier


class KNeighbors(Classifier):
    name = "k-nearest neighbors algorithm"
    short_name = "k-NN"

    def __init__(self, data, labels, training_fraction):
        super().__init__(data, labels, training_fraction)
        self.model = n.KNeighborsClassifier()
