from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import BaggingClassifier

from classifiers.classifier import Classifier


class Bagging(Classifier):
    name = "Bagging classifier"
    short_name = "Bagging"

    def __init__(self, data, labels, unique, training_fraction):
        super().__init__(data, labels, unique, training_fraction)
        self.model = BaggingClassifier(
            base_estimator=tree.DecisionTreeClassifier(),
            n_estimators=20,
            random_state=1,
            n_jobs= -1)
