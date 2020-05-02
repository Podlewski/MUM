from sklearn import tree
from classifiers.classifier import Classifier

criterion = {0: "gini", 1: "entropy"}


class DecisionTree(Classifier):

    chosen_criterion = criterion[0]
    max_depth = None
    max_leaf_nodes = None

    name = "Decision trees algorithm"

    def __init__(self, data, labels, training_fraction, arguments):
        super().__init__(data, labels, training_fraction)

        try:
            self.chosen_criterion = criterion[arguments[0]]
            self.max_depth = int(arguments[1])
            self.max_leaf_nodes = int(arguments[2])
            self.model = tree.DecisionTreeClassifier(criterion=self.chosen_criterion,
                                                     max_depth=self.max_depth,
                                                     max_leaf_nodes=self.max_leaf_nodes)
        except:
            self.model = tree.DecisionTreeClassifier()

    # def print_stats(self, dataset_name, basic=True):
    #     print(f"Criterion:\t\t\t{self.chosen_criterion}")
    #     print(f"Max depth:\t\t\t{self.max_depth}")
    #     print(f"Max leaf nodes:\t\t\t{self.max_leaf_nodes}")
    #     print(self.get_metrics())
