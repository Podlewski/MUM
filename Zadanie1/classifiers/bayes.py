from sklearn import naive_bayes as nb

from classifiers.classifier import Classifier


class Bayes(Classifier):

    def __init__(self, data, lr, labels, training_fraction=0.3):
        super().__init__(data, lr, labels, training_fraction)
        self.model = nb.GaussianNB()
