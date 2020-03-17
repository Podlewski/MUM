from sklearn import tree

from classifier import Classifier


class DecisionTree(Classifier):

    def __init__(self, data, training_fraction=0.3):
        super().__init__(data, training_fraction)
        self.model = tree.DecisionTreeClassifier()
