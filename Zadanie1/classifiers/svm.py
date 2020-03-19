from sklearn import svm

from classifiers.classifier import Classifier


class SVM(Classifier):

    def __init__(self, data, lr, labels, training_fraction=0.2):
        super().__init__(data, lr, labels, training_fraction)
        self.model = svm.SVC()
