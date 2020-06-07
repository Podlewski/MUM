from sklearn.ensemble import GradientBoostingClassifier

from classifiers.classifier import Classifier


class Boosting(Classifier):
    name = "Boosting classifier"
    short_name = "Boosting"

    def __init__(self, data, labels, unique, training_fraction):
        super().__init__(data, labels, unique, training_fraction)
        self.model = GradientBoostingClassifier(
            random_state=0,
            n_estimators=10,
            learning_rate=0.1,
            max_depth=3)
