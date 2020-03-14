from copy import deepcopy

from classifier import Classifier


class Bayes(Classifier):
    training_part = 0.7

    def __init__(self, data):
        super().__init__(data)
