from sklearn import neighbors as n

from classifiers.classifier import Classifier


class KNeighbors(Classifier):
    name = "k-nearest neighbors algorithm"

    def __init__(self, data, lr, labels, training_fraction, arguments):
        super().__init__(data, lr, labels, training_fraction, arguments)
        n_neighbors = int(input("Choose a number of neighbors:"))
        self.model = n.KNeighborsClassifier(n_neighbors)
