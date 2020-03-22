from sklearn import naive_bayes as nb

from classifiers.classifier import Classifier


class Bayes(Classifier):

    name = "Naive Bayes classifier"

    def __init__(self, data, lr, labels, training_fraction, arguments):
        super().__init__(data, lr, labels, training_fraction, arguments)
        self.model = nb.GaussianNB()
