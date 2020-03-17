from sklearn import neighbors as n

from classifier import Classifier


class KNeighbors(Classifier):

    def __init__(self, data, training_fraction=0.3):
        super().__init__(data, training_fraction)
        self.model = n.KNeighborsClassifier()
