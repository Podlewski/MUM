from sklearn import svm
import pandas

from classifier import Classifier


class SVM(Classifier):

    def __init__(self, data, training_fraction=0.2):
        super().__init__(data, training_fraction)
        self.model = svm.SVC()
