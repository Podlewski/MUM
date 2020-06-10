from sklearn import naive_bayes as nb
from sklearn import neighbors as n
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from classifiers.classifier import Classifier


class Stacking(Classifier):
    name = "Stacking classifier"
    short_name = "Stacking"

    def __init__(self, data, labels, unique, training_fraction):
        super().__init__(data, labels, unique, training_fraction)
        self.model = StackingClassifier(
                estimators=[('Tree', tree.DecisionTreeClassifier()),
                            ('Bayes', nb.GaussianNB()),
                            ('KNN', n.KNeighborsClassifier())],
                final_estimator=LogisticRegression())
