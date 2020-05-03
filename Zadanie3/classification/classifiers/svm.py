from sklearn import svm

from classifiers.classifier import Classifier


class SVM(Classifier):
    name = "Support-vector machine"
    short_name = "SVM"

    def __init__(self, data, labels, training_fraction, arguments):
        super().__init__(data, labels, training_fraction)

        try:
            self.regularization = float(arguments[0])
            self.kernel = str(arguments[1])
            self.model = svm.SVC(C=self.regularization,
                                 kernel=self.kernel)
        except:
            self.model = svm.SVC()
