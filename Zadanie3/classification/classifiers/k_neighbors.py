from sklearn import neighbors as n

from classifiers.classifier import Classifier


class KNeighbors(Classifier):
    name = "k-nearest neighbors algorithm"
    short_name = "k-NN"

    def __init__(self, data, labels, training_fraction, arguments):
        super().__init__(data, labels, training_fraction)
        # n_neighbors = int(input("Choose a number of neighbors:"))
        # self.model = n.KNeighborsClassifier(n_neighbors)
        self.model = n.KNeighborsClassifier()
