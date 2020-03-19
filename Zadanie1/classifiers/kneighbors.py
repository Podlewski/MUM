from sklearn import neighbors as n

from classifiers.classifier import Classifier


class KNeighbors(Classifier):

    def __init__(self, data, lr, labels, training_fraction=0.3):
        super().__init__(data, lr, labels, training_fraction)
        self.model = n.KNeighborsClassifier()
