from sklearn import naive_bayes as nb

from classifiers.classifier import Classifier


class Bayes(Classifier):
    name = "Naive Bayes classifier"
    short_name = "Bayes"

    def __init__(self, data, labels, training_fraction):
        super().__init__(data, labels, training_fraction)
        self.model = nb.GaussianNB()
