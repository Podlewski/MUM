from sklearn import tree

from classifiers.classifier import Classifier


class DecisionTree(Classifier):
    name = "Decision trees algorithm"

    def __init__(self, data, lr, labels, training_fraction, arguments):
        super().__init__(data, lr, labels, training_fraction, arguments)
        self.model = tree.DecisionTreeClassifier()
