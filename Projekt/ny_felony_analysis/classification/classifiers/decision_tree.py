from sklearn import tree
from classifiers.classifier import Classifier

criterion = {0: "gini", 1: "entropy"}


class DecisionTree(Classifier):

    chosen_criterion = criterion[0]
    max_depth = None
    max_leaf_nodes = None

    name = "Decision trees algorithm"
    short_name = "Tree"

    def __init__(self, data, labels, training_fraction):
        super().__init__(data, labels, training_fraction)
        self.model = tree.DecisionTreeClassifier()
