from sklearn import naive_bayes as nb

from classifier import Classifier


class Bayes(Classifier):

    def __init__(self, data, training_fraction=0.3):
        super().__init__(data, training_fraction)
        self.model = nb.GaussianNB()
