from sklearn import svm

from classifiers.classifier import Classifier


class SVM(Classifier):
    name = "Support-vector machine"
    short_name = "SVM"

    def __init__(self, data, labels, training_fraction):
        super().__init__(data, labels, training_fraction)
        self.model = svm.SVC(max_iter=5000)
